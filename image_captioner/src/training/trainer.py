import os
import torch
import torch.nn as nn
import random
import time
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any, Union, Set
from torch.utils.data import DataLoader

from src.models.decoder import Transformer_Decoder, sample_caption
from src.models.beam_search import beam_search_caption
from src.data.datasets import CaptionFeatureDataset, CaptionEvaluationDataset
from src.training.scheduler import create_optimizer_and_scheduler
from src.evaluation.clip_scorer import CLIPCalculator, compute_clip_reward_loss, evaluate_model_with_clip_score, load_references
from src.utils.file_utils import save_model, load_model, find_best_model
from src.utils.visualization import plot_training_curves, plot_training_phases_comparison
from src.config import DEVICE


def train_model_enhanced(
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        word2idx: Dict[str, int], 
        idx2word: Dict[int, str],
        embed_size: int = 256, 
        hidden_size: int = 512, 
        num_layers: int = 4,
        learning_rate: float = 0.0003, 
        num_epochs: int = 20,
        early_stopping_patience: int = 5, 
        checkpoint_frequency: int = 1,
        model_path: Optional[str] = None, 
        output_path: Optional[str] = None, 
        feature_dim: int = 512,
        clip_loss_weight: float = 0.5, 
        clip_batch_size: int = 16,
        clip_eval_frequency: int = 50, 
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01) -> Tuple[Transformer_Decoder, torch.nn.Module, Dict[str, Any]]:
    """
    enhanced training function that integrates CLIP scores using policy gradients

    Args:
        train_loader: dataloader for training
        val_loader: dataloader for validation
        word2idx, idx2word: vocab mappings
        embed_size, hidden_size, num_layers: model architecture parameters
        learning_rate: lr
        num_epochs: nr of epochs
        early_stopping_patience: nr of epochs to wait before early stopping
        checkpoint_frequency: how often to save checkpoints (in epochs)
        model_path: path to load pre-trained model (to resume training)
        output_path: where to save models
        feature_dim: dimension of image features (512 for ResNet18, 2048 for ResNet50!!!)
        clip_loss_weight: weight for the CLIP-based loss component (0-1)
        clip_batch_size: nr of samples for CLIP evaluation in each batch
        clip_eval_frequency: wow often to evaluate CLIP loss (in training steps)
        warmup_ratio: portion of training to use for warmup
        weight_decay: weight decay coefficient

    Returns:
        trained decoder and project_features models, training history
    """

    history = {
        'train_loss': [],
        'val_loss': [],
        'clip_scores': [],
        'clip_batch_scores': [],
        'learning_rates': [],
        'eval_epochs': [],
        'best_val_loss': float('inf'),
        'best_clip_score': 0.0,
        'epochs_without_improvement': 0,
        'start_time': time.time(),
        'total_training_time': 0
    }

    vocab_size = len(word2idx)


    clip_calculator = CLIPCalculator()

    # cache reference captions and image paths
    print("Loading and caching reference captions...")
    train_image_ids = set(train_loader.dataset.image_names)
    val_image_ids = set(val_loader.dataset.image_names)
    all_used_image_ids = train_image_ids.union(val_image_ids)
    

    first_img_name = train_loader.dataset.image_names[0]
    
    from src.config import PROJECT_ROOT

    # absolute paths based on the project root
    captions_file = os.path.join(PROJECT_ROOT, "data", "captions", "filtered_captions.tsv")
    image_folder = os.path.join(PROJECT_ROOT, "data", "images")

    # make sure the file exists before trying to use it
    if not os.path.exists(captions_file):
        # try the filtered version as a fallback
        captions_file = os.path.join(PROJECT_ROOT, "data", "captions", "filtered_captions_matched.tsv")
        if not os.path.exists(captions_file):
            raise FileNotFoundError(f"cant find captions file at {captions_file}")

    if not os.path.exists(image_folder):
        raise FileNotFoundError(f"cant find image folder at {image_folder}")

    print(f"using captions file: {captions_file}")
    print(f"using image folder: {image_folder}")
    
    cached_references, cached_image_paths = load_references(
        captions_file, image_folder, filter_ids=all_used_image_ids)
    print(f"cached {len(cached_references)} reference captions")

    # initialize or load models
    if model_path:
        try:
            decoder, project_features, metadata = load_model(
                model_path, embed_size, vocab_size, hidden_size, num_layers, DEVICE
            )

            # check if feature dimensions match
            if project_features.weight.size(1) != feature_dim:
                print(f"WARNING: model expects {project_features.weight.size(1)}-dim features, but we have {feature_dim}-dim features")
                print("re-initializing projection layer...")
                project_features = nn.Linear(feature_dim, embed_size).to(DEVICE)

            # load training history e
            if 'metrics' in metadata and isinstance(metadata['metrics'], dict):
                for key in history:
                    if key in metadata['metrics']:
                        history[key] = metadata['metrics'][key]
                print(f"continue training from epoch {len(history['train_loss'])+1}")
        except Exception as e:
            print(f"error loading model: {e}")
            print("initializing new model...")
            decoder = Transformer_Decoder(embed_size, vocab_size, hidden_size, num_layers).to(DEVICE)
            project_features = nn.Linear(feature_dim, embed_size).to(DEVICE)
    else:
        # initialize new models
        decoder = Transformer_Decoder(embed_size, vocab_size, hidden_size, num_layers).to(DEVICE)
        project_features = nn.Linear(feature_dim, embed_size).to(DEVICE)
        print("initialized new model")

    # optimizer and scheduler
    total_steps = len(train_loader) * num_epochs
    optimizer, scheduler = create_optimizer_and_scheduler(
        list(decoder.parameters()) + list(project_features.parameters()),
        learning_rate=learning_rate,
        num_training_steps=total_steps,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay
    )

    # cross-entropy loss for supervised training
    criterion = nn.CrossEntropyLoss(ignore_index=word2idx["<PAD>"])


    # Get features from validation dataset
    features_dict = val_loader.dataset.features_dict
    eval_dataset = CaptionEvaluationDataset(features_dict)
    eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=False, num_workers=2)

    print(f"starting training with {num_epochs} epochs")
    print(f"feature dimension: {feature_dim}")
    print(f"architecture: {num_layers} layers, {hidden_size} hidden dim, {embed_size} embed dim")
    print(f"CLIP loss weight: {clip_loss_weight}")
    print(f"evaluating CLIP every {clip_eval_frequency} steps")

    start_epoch = len(history['train_loss'])
    global_step = 0

    for epoch in range(start_epoch, start_epoch + num_epochs):
        epoch_start_time = time.time()

        # training phase
        decoder.train()
        project_features.train()
        total_ce_loss = 0
        total_clip_loss = 0
        total_combined_loss = 0
        epoch_clip_scores = []
        num_clip_batches = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{start_epoch+num_epochs} [Train]")
        for batch_idx, (features, captions) in enumerate(progress_bar):
            features, captions = features.to(DEVICE), captions.to(DEVICE)

            # standard cross-entropy loss
            optimizer.zero_grad()
            projected = project_features(features)
            output = decoder(projected, captions[:, :-1])
            ce_loss = criterion(output.reshape(-1, vocab_size), captions[:, 1:].reshape(-1))

            # determine whether to compute CLIP reward loss
            compute_clip = (clip_loss_weight > 0) and (
            (global_step % clip_eval_frequency == 0)  
            )
            clip_loss = torch.tensor(0.0, device=DEVICE)
            clip_score = 0.0

            # compute policy gradient loss with CLIP rewards (occasionally)
            if compute_clip:
                # select a subset of images for CLIP evaluation
                clip_eval_indices = random.sample(
                    range(len(features)), min(clip_batch_size, len(features)))

                clip_features = features[clip_eval_indices]
                img_ids = [train_loader.dataset.image_names[batch_idx * train_loader.batch_size + i]
                          for i in clip_eval_indices]

                # ccalculate policy gradient loss with CLIP as reward
                clip_loss, clip_score = compute_clip_reward_loss(
                    clip_features, img_ids, decoder, project_features,
                    word2idx, idx2word, clip_calculator,
                    cached_references, cached_image_paths, DEVICE
                )

                if clip_score > 0:
                    epoch_clip_scores.append(clip_score)
                    num_clip_batches += 1

            # combine losses
            if compute_clip and clip_score > 0:
                combined_loss = (1 - clip_loss_weight) * ce_loss + clip_loss_weight * clip_loss
                total_clip_loss += clip_loss.item()
            else:
                combined_loss = ce_loss

            combined_loss.backward()

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(project_features.parameters(), max_norm=1.0)

            # update weights
            optimizer.step()
            scheduler.step()

            # track current learning rate
            current_lr = scheduler.get_last_lr()[0]

            # update progress bar and tracking variables
            total_ce_loss += ce_loss.item()
            total_combined_loss += combined_loss.item()

            if compute_clip and clip_score > 0:
                progress_bar.set_postfix(
                    CE_Loss=f"{ce_loss.item():.4f}",
                    CLIP_Loss=f"{clip_loss.item():.4f}",
                    CLIP_Score=f"{clip_score:.4f}",
                    LR=f"{current_lr:.6f}"
                )
            else:
                progress_bar.set_postfix(
                    CE_Loss=f"{ce_loss.item():.4f}",
                    LR=f"{current_lr:.6f}"
                )

            global_step += 1

        # average losses for the epoch
        avg_ce_loss = total_ce_loss / len(train_loader)
        avg_combined_loss = total_combined_loss / len(train_loader)


        # Calculate average CLIP score if available
        avg_clip_score = sum(epoch_clip_scores) / num_clip_batches if num_clip_batches > 0 else 0.0


        history['train_loss'].append(avg_ce_loss)
        history['clip_batch_scores'].append(avg_clip_score)
        history['learning_rates'].append(current_lr)

        # validation phase
        decoder.eval()
        project_features.eval()
        total_val_loss = 0

        with torch.no_grad():
            for features, captions in tqdm(val_loader, desc=f"Epoch {epoch+1}/{start_epoch+num_epochs} [Val]"):
                features, captions = features.to(DEVICE), captions.to(DEVICE)
                projected = project_features(features)
                output = decoder(projected, captions[:, :-1])
                loss = criterion(output.reshape(-1, vocab_size), captions[:, 1:].reshape(-1))
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)

        # epoch time
        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - history['start_time']
        history['total_training_time'] = total_time

        # Print progress summary
        print(f"epoch {epoch+1}/{start_epoch+num_epochs} - "
              f"train CE Loss: {avg_ce_loss:.4f}, "
              f"val Loss: {avg_val_loss:.4f}, "
              f"CLIP Score: {avg_clip_score:.4f}, "
              f"time: {epoch_time:.1f}s, "
              f"total: {timedelta(seconds=int(total_time))}")

        # save checkpoint sometimes
        if (epoch + 1) % checkpoint_frequency == 0:
            save_model(
                decoder, project_features, history, output_path,
                model_type=f"checkpoint"
            )

        # track best validation loss
        if avg_val_loss < history['best_val_loss']:
            history['best_val_loss'] = avg_val_loss
            history['epochs_without_improvement'] = 0
            save_model(
                decoder, project_features, history, output_path,
                model_type="best_val"
            )
            print(f"new best model saved based on validation loss: {avg_val_loss:.4f}!!")
        else:
            history['epochs_without_improvement'] += 1

        # evaluate with CLIP score every few epochs or at the end
        evaluate_clip = ((epoch + 1) % 2 == 0) or (epoch == start_epoch + num_epochs - 1)

        if evaluate_clip:
            print("evaluating with CLIP score...")
            clip_score, generated_captions = evaluate_model_with_clip_score(
            decoder, project_features, eval_loader, word2idx, idx2word,
            clip_calculator, captions_file, image_folder,
            train_loader.dataset.image_names,  # Pass train image names
            val_loader.dataset.image_names,    # Pass validation image names
            max_eval_images=100
            )

            history['clip_scores'].append(clip_score)
            history['eval_epochs'].append(epoch)

            # Save model if CLIP score improved
            if clip_score > history['best_clip_score']:
                history['best_clip_score'] = clip_score
                save_model(
                    decoder, project_features, history, output_path,
                    model_type="best_clip"
                )
                print(f"new best model saved based on CLIP score: {clip_score:.4f}!!")

        # sample some captions every few epochs
        if (epoch + 1) % 2 == 0:
            print("\nSample captions:")
            sample_features, _ = next(iter(val_loader))
            for i in range(min(3, len(sample_features))):
                feature = sample_features[i].unsqueeze(0).to(DEVICE)
                caption = beam_search_caption(
                    feature, decoder, project_features,
                    word2idx, idx2word, DEVICE, beam_width=3
                )
                print(f"sample {i+1}: {caption}")
            print()

        # early stopping check
        if history['epochs_without_improvement'] >= early_stopping_patience:
            print(f"early stopping triggered after {epoch+1} epochs")
            break

    # plot training curves
    plots_dir = os.path.join(os.path.dirname(output_path), "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, "training_curves.png")
    plot_training_curves(history, save_path=plot_path)

    # load the best model (using CLIP score)
    best_model_path = find_best_model(output_path, "best_clip")
    if best_model_path:
        print(f"loading best model from {best_model_path}")
        decoder, project_features, _ = load_model(
            best_model_path, embed_size, vocab_size, hidden_size, num_layers, DEVICE
        )

    return decoder, project_features, history


def train_in_phases(train_loader: DataLoader, 
                   val_loader: DataLoader, 
                   word2idx: Dict[str, int], 
                   idx2word: Dict[int, str], 
                   config: Dict[str, Any]) -> Tuple[Transformer_Decoder, torch.nn.Module, List[Dict[str, Any]]]:
    """
    train a model in multiple phases with different objectives

    Args:
        train_loader: dataloader for training
        val_loader: dataloader for validation
        word2idx, idx2word: vocab mappings
        config: base configuration for training

    Returns:
        tuple of (decoder, project_features, histories)
    """
    import os
    from datetime import datetime

    # make unique experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(os.path.dirname(config['output_path']), f"experiment_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)

    print(f"Starting multi-phase training in {experiment_dir}")

    # save configuration
    with open(os.path.join(experiment_dir, "config.txt"), "w") as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")

    histories = []

    # phase 1: initial training with cross-entropy loss only
    print("\n" + "="*50)
    print("PHASE 1: Cross-Entropy Training")
    print("="*50)

    phase1_config = config.copy()
    phase1_config.update({
        'num_epochs': 10,
        'clip_loss_weight': 0.0,  # No CLIP loss
        'learning_rate': 0.0003,
        'output_path': os.path.join(experiment_dir, "phase1_model")
    })

    # train with cross-entropy only
    decoder, project_features, history1 = train_model_enhanced(
        train_loader=train_loader,
        val_loader=val_loader,
        word2idx=word2idx,
        idx2word=idx2word,
        embed_size=phase1_config['embed_size'],
        hidden_size=phase1_config['hidden_size'],
        num_layers=phase1_config['num_layers'],
        learning_rate=phase1_config['learning_rate'],
        num_epochs=phase1_config['num_epochs'],
        early_stopping_patience=phase1_config['early_stopping_patience'],
        checkpoint_frequency=phase1_config['checkpoint_frequency'],
        model_path=None,  # Start from scratch
        output_path=phase1_config['output_path'],
        feature_dim=phase1_config['feature_dim'],
        clip_loss_weight=phase1_config['clip_loss_weight'],
        clip_batch_size=phase1_config['clip_batch_size'],
        clip_eval_frequency=phase1_config['clip_eval_frequency']
    )

    histories.append(history1)

    # save phase 1 training curves
    plot_path = os.path.join(experiment_dir, "phase1_curves.png")
    plot_training_curves(history1, save_path=plot_path)

    # find best model from phase 1
    phase1_best_model = find_best_model(phase1_config['output_path'], "best_val")

    # phase 2: balanced training with both losses
    print("\n" + "="*50)
    print("PHASE 2: Combined Cross-Entropy and CLIP Training")
    print("="*50)

    phase2_config = config.copy()
    phase2_config.update({
        'num_epochs': 10,
        'clip_loss_weight': 0.3,  # moderate CLIP influence
        'learning_rate': 0.0001,  # lower learning rate for fine-tuning
        'output_path': os.path.join(experiment_dir, "phase2_model")
    })

    # train with combined loss
    decoder, project_features, history2 = train_model_enhanced(
        train_loader=train_loader,
        val_loader=val_loader,
        word2idx=word2idx,
        idx2word=idx2word,
        embed_size=phase2_config['embed_size'],
        hidden_size=phase2_config['hidden_size'],
        num_layers=phase2_config['num_layers'],
        learning_rate=phase2_config['learning_rate'],
        num_epochs=phase2_config['num_epochs'],
        early_stopping_patience=phase2_config['early_stopping_patience'],
        checkpoint_frequency=phase2_config['checkpoint_frequency'],
        model_path=phase1_best_model,  # continue from phase 1
        output_path=phase2_config['output_path'],
        feature_dim=phase2_config['feature_dim'],
        clip_loss_weight=phase2_config['clip_loss_weight'],
        clip_batch_size=phase2_config['clip_batch_size'],
        clip_eval_frequency=phase2_config['clip_eval_frequency']
    )

    histories.append(history2)

    # save phase 2 training curves
    plot_path = os.path.join(experiment_dir, "phase2_curves.png")
    plot_training_curves(history2, save_path=plot_path)

    # best model from phase 2
    phase2_best_model = find_best_model(phase2_config['output_path'], "best_clip")

    # phase 3: CLIP fine-tuning
    print("\n" + "="*50)
    print("PHASE 3: CLIP-Only Fine-Tuning")
    print("="*50)

    phase3_config = config.copy()
    phase3_config.update({
        'num_epochs': 8,
        'clip_loss_weight': 0.9,  # very high CLIP influence
        'learning_rate': 5e-5,  # very low learning rate for fine-tuning
        'output_path': os.path.join(experiment_dir, "phase3_model")
    })


    decoder, project_features, history3 = train_model_enhanced(
        train_loader=train_loader,
        val_loader=val_loader,
        word2idx=word2idx,
        idx2word=idx2word,
        embed_size=phase3_config['embed_size'],
        hidden_size=phase3_config['hidden_size'],
        num_layers=phase3_config['num_layers'],
        learning_rate=phase3_config['learning_rate'],
        num_epochs=phase3_config['num_epochs'],
        early_stopping_patience=phase3_config['early_stopping_patience'],
        checkpoint_frequency=phase3_config['checkpoint_frequency'],
        model_path=phase2_best_model,  # Continue from phase 2
        output_path=phase3_config['output_path'],
        feature_dim=phase3_config['feature_dim'],
        clip_loss_weight=phase3_config['clip_loss_weight'],
        clip_batch_size=phase3_config['clip_batch_size'],
        clip_eval_frequency=phase3_config['clip_eval_frequency']
    )

    histories.append(history3)

    # save phase 3 training curves
    plot_path = os.path.join(experiment_dir, "phase3_curves.png")
    plot_training_curves(history3, save_path=plot_path)

    # find the final best model
    final_best_model = find_best_model(phase3_config['output_path'], "best_clip")

    # if no phase 3 model is better, use phase 2's best
    if not final_best_model:
        final_best_model = phase2_best_model

    # load the best overall model
    decoder, project_features, _ = load_model(
        final_best_model,
        phase3_config['embed_size'],
        len(word2idx),
        phase3_config['hidden_size'],
        phase3_config['num_layers'],
        DEVICE
    )

    # final evaluation with CLIP score
    print("\n" + "="*50)
    print("FINAL EVALUATION!")
    print("="*50)

    # get datasets & paths for evaluation
    train_image_ids = set(train_loader.dataset.image_names)
    val_image_ids = set(val_loader.dataset.image_names)
    
    # use the same paths as in training
    eval_dataset = val_loader.dataset
    eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=False, num_workers=2)

    if hasattr(train_loader.dataset, 'captions_file') and hasattr(train_loader.dataset, 'image_dir'):
        captions_file = train_loader.dataset.captions_file
        image_folder = train_loader.dataset.image_dir
    else:
        captions_file = os.path.join(os.path.dirname(phase3_config['output_path']), "filtered_captions.tsv")
        image_folder = os.path.dirname(os.path.dirname(phase3_config['output_path']))
        print(f"Using derived paths: {captions_file}, {image_folder}")

    clip_calculator = CLIPCalculator()
    final_clip_score, _ = evaluate_model_with_clip_score(
        decoder, project_features, eval_loader, word2idx, idx2word,
        clip_calculator, captions_file, image_folder,
        train_image_ids, val_image_ids, max_eval_images=100
    )

    print(f"final CLIP Score: {final_clip_score:.4f}")

    plot_training_phases_comparison(histories, experiment_dir)

    # save final model
    final_model_path = os.path.join(experiment_dir, "final_model.pt")
    save_dict = {
        'decoder': decoder.state_dict(),
        'project_features': project_features.state_dict(),
        'final_clip_score': final_clip_score,
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'type': "final"
    }
    torch.save(save_dict, final_model_path)
    print(f"final model saved to {final_model_path}")

    return decoder, project_features, histories