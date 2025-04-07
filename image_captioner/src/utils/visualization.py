import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Optional
import os


def plot_training_curves(history: Dict[str, Any], save_path: Optional[str] = None) -> None:
    """
    plot enhanced training curves with multiple metrics

    Args:
        history: training history dictionary
        save_path: optional path to save the plot
    """
    fig = plt.figure(figsize=(18, 12))

    # 1: loss curves
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(history['train_loss'], 'b-', label='Training CE Loss')
    ax1.plot(history['val_loss'], 'r-', label='Validation Loss')

    if 'best_val_loss' in history:
        ax1.axhline(y=history['best_val_loss'], color='r', linestyle='--',
                   label=f'Best Val Loss: {history["best_val_loss"]:.4f}')

    ax1.set_title('Loss Curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # 2: CLIP scores
    ax2 = fig.add_subplot(2, 2, 2)

    # plot CLIP batch scores if available
    if 'clip_batch_scores' in history and any(history['clip_batch_scores']):
        clip_batch_x = list(range(len(history['clip_batch_scores'])))
        ax2.plot(clip_batch_x, history['clip_batch_scores'], 'g-', alpha=0.5,
                label='Training CLIP Scores')

    # plot evaluation CLIP scores if available
    if 'clip_scores' in history and 'eval_epochs' in history and history['clip_scores']:
        ax2.plot(history['eval_epochs'], history['clip_scores'], 'g-o',
                label='Evaluation CLIP Scores')

        if 'best_clip_score' in history:
            ax2.axhline(y=history['best_clip_score'], color='r', linestyle='--',
                       label=f'Best CLIP Score: {history["best_clip_score"]:.4f}')

    ax2.set_title('CLIP Score Progression')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('CLIP Score')
    ax2.legend()
    ax2.grid(True)

    # 3: lr
    ax3 = fig.add_subplot(2, 2, 3)
    if 'learning_rates' in history and history['learning_rates']:
        ax3.plot(history['learning_rates'], 'c-')
        ax3.set_title('Learning Rate Schedule')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')  # log scale for better visualization
        ax3.grid(True)

    # 4: combined metrics (optional)
    ax4 = fig.add_subplot(2, 2, 4)

    # extra axis for CLIP score
    if ('clip_scores' in history and history['clip_scores'] and
        'eval_epochs' in history and 'val_loss' in history):

        # plot validation loss on primary axis
        epochs = list(range(len(history['val_loss'])))
        line1 = ax4.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Validation Loss', color='r')
        ax4.tick_params(axis='y', labelcolor='r')

        # extra axis for CLIP score
        ax4_twin = ax4.twinx()
        line2 = ax4_twin.plot(history['eval_epochs'], history['clip_scores'], 'g-o',
                             label='CLIP Score')
        ax4_twin.set_ylabel('CLIP Score', color='g')
        ax4_twin.tick_params(axis='y', labelcolor='g')

        # legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc='upper right')

        ax4.set_title('Validation Loss vs CLIP Score')
        ax4.grid(True)

    plt.tight_layout()
    
    # save the figure 
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Saved training curves to {save_path}")
    
    plt.show()


def plot_training_phases_comparison(histories: List[Dict[str, Any]], output_dir: str) -> None:
    """
    create a plot comparing metrics across training phases

    Args:
        histories: list of training histories for each phase
        output_dir: directory to save the plot
    """

    plt.figure(figsize=(15, 12))

    # 1: training loss across phases
    ax1 = plt.subplot(2, 2, 1)
    colors = ['b', 'g', 'r']

    for i, history in enumerate(histories):
        if 'train_loss' in history and history['train_loss']:
            
            epochs = np.arange(len(history['train_loss']))

            if i > 0:
                offset = sum(len(h['train_loss']) for h in histories[:i])
                epochs = epochs + offset

            ax1.plot(epochs, history['train_loss'], f'{colors[i]}-',
                    label=f'Phase {i+1} Training Loss')

    ax1.set_title('Training Loss Across Phases')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # 2: validation loss across phases
    ax2 = plt.subplot(2, 2, 2)

    for i, history in enumerate(histories):
        if 'val_loss' in history and history['val_loss']:

            epochs = np.arange(len(history['val_loss']))

            if i > 0:
                offset = sum(len(h['val_loss']) for h in histories[:i])
                epochs = epochs + offset

            ax2.plot(epochs, history['val_loss'], f'{colors[i]}-',
                    label=f'Phase {i+1} Validation Loss')

    ax2.set_title('Validation Loss Across Phases')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    # 3: CLIP scores across phases
    ax3 = plt.subplot(2, 2, 3)

    for i, history in enumerate(histories):
        if 'clip_scores' in history and history['clip_scores'] and 'eval_epochs' in history:
           
            eval_epochs = history['eval_epochs']

            if i > 0:
                offset = sum(len(h['train_loss']) for h in histories[:i])
                eval_epochs = [e + offset for e in eval_epochs]

            ax3.plot(eval_epochs, history['clip_scores'], f'{colors[i]}-o',
                    label=f'Phase {i+1} CLIP Score')

    ax3.set_title('CLIP Scores Across Phases')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('CLIP Score')
    ax3.legend()
    ax3.grid(True)

    # 4: lr across phases
    ax4 = plt.subplot(2, 2, 4)

    for i, history in enumerate(histories):
        if 'learning_rates' in history and history['learning_rates']:
            steps = np.arange(len(history['learning_rates']))

            if i > 0:
                offset = sum(len(h['learning_rates']) for h in histories[:i])
                steps = steps + offset

            ax4.plot(steps, history['learning_rates'], f'{colors[i]}-',
                    label=f'Phase {i+1} Learning Rate')

    ax4.set_title('Learning Rate Schedule Across Phases')
    ax4.set_xlabel('Training Step')
    ax4.set_ylabel('Learning Rate')
    ax4.set_yscale('log')
    ax4.legend()
    ax4.grid(True)

    # save plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'all_phases_comparison.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved phase comparison plot to {output_path}")
    plt.close()