import os
import sys
import argparse
import torch
import random
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.config import DATA_DIR, IMAGE_DIR, CAPTIONS_DIR, MODELS_DIR, DEVICE, update_config
from src.data.preparation import prepare_data_with_cache, filter_captions_file
from src.data.datasets import CaptionFeatureDataset, CaptionEvaluationDataset
from src.training.trainer import train_model_enhanced, train_in_phases


def parse_args():
    """
    command line arguments
    """
    parser = argparse.ArgumentParser(description="Train an image captioning model")
    
    # paths
    parser.add_argument("--captions-file", type=str, default=os.path.join(CAPTIONS_DIR, "filtered_captions.tsv"),
                       help="Path to captions TSV file")
    parser.add_argument("--image-dir", type=str, default=os.path.join(IMAGE_DIR),
                       help="Path to directory containing images")
    parser.add_argument("--output-dir", type=str, default=os.path.join(MODELS_DIR),
                       help="Directory to save trained models")
    parser.add_argument("--model-path", type=str, default=None,
                       help="Path to pre-trained model to continue training (optional)")
    
    # parameters
    parser.add_argument("--max-images", type=int, default=10000,
                       help="Maximum number of images to use")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=20,
                       help="Number of epochs to train")
    parser.add_argument("--embed-size", type=int, default=256,
                       help="Size of word embeddings")
    parser.add_argument("--hidden-size", type=int, default=768,
                       help="Size of hidden layers")
    parser.add_argument("--num-layers", type=int, default=6,
                       help="Number of transformer layers")
    parser.add_argument("--learning-rate", type=float, default=0.0003,
                       help="Initial learning rate")
    parser.add_argument("--clip-loss-weight", type=float, default=0.3,
                       help="Weight for CLIP loss component")
    parser.add_argument("--multi-phase", action="store_true",
                       help="Use multi-phase training strategy")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--force-reload", action="store_true",
                       help="Force reloading and re-extracting features")
    parser.add_argument("--workers", type=int, default=4,
                       help="Number of worker threads for data loading")
    
    return parser.parse_args()


def main():
    """
    main training function
    """
    args = parse_args()
    
    # output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    print("\n" + "="*60)
    print(f"starting image captioning training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"using: {DEVICE}")
    print("="*60)
    
    # captions file exists?
    if not os.path.exists(args.captions_file):
        raise FileNotFoundError(f"captions file not found at {args.captions_file}")
    
    # image directory exists?
    if not os.path.exists(args.image_dir):
        raise FileNotFoundError(f"Image directory not found at {args.image_dir}")
    
    # filter captions to match available images
    filtered_captions_file = os.path.join(CAPTIONS_DIR, "filtered_captions_matched.tsv")
    filter_captions_file(
        captions_file=args.captions_file,
        image_folder=args.image_dir,
        output_file=filtered_captions_file
    )
    
    # prep data
    print("\nPreparing data...")
    cache_path = os.path.join(args.output_dir, "cached_features.pt")
    
    features_dict, captions_dict, word2idx, idx2word = prepare_data_with_cache(
        filtered_captions_file,
        args.image_dir,
        max_images=args.max_images,
        cache_path=cache_path,
        force_reload=args.force_reload,
        num_workers=args.workers,
        batch_size=args.batch_size
    )
    
    # datasets and data loaders
    train_dataset = CaptionFeatureDataset(
        features_dict, captions_dict, word2idx, max_len=22, split='train'
    )
    val_dataset = CaptionFeatureDataset(
        features_dict, captions_dict, word2idx, max_len=22, split='val'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True if DEVICE == "cuda" else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True if DEVICE == "cuda" else False
    )
    
    print(f"training set: {len(train_dataset)} samples")
    print(f"validation set: {len(val_dataset)} samplees")
    
    # configuration with command line arguments
    config = update_config(args)
    
    # additional parameters to config
    sample_feature = next(iter(features_dict.values()))
    feature_dim = sample_feature.size(0)
    print(f"detected feature dimension: {feature_dim}")
    
    config.update({
        'feature_dim': feature_dim,
        'num_epochs': args.epochs,
        'output_path': os.path.join(args.output_dir, f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    })
    
    # train model
    if args.multi_phase:
        print("\nrunning multi-phase training...")
        decoder, project_features, histories = train_in_phases(
            train_loader, val_loader, word2idx, idx2word, config
        )
    else:
        print("\nrunning single-phase training...")
        decoder, project_features, history = train_model_enhanced(
            train_loader=train_loader,
            val_loader=val_loader,
            word2idx=word2idx,
            idx2word=idx2word,
            embed_size=config['embed_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            learning_rate=config['learning_rate'],
            num_epochs=config['num_epochs'],
            early_stopping_patience=config['early_stopping_patience'],
            model_path=args.model_path,
            output_path=config['output_path'],
            feature_dim=config['feature_dim'],
            clip_loss_weight=config['clip_loss_weight']
        )
    
    
    print("\n" + "="*60)
    print("training complete! :)")
    print("="*60)


if __name__ == "__main__":
    main()