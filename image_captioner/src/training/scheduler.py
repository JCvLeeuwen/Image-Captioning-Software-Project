import torch
import math
from typing import List, Callable


def get_cosine_schedule_with_warmup(optimizer: torch.optim.Optimizer, 
                                   num_warmup_steps: int, 
                                   num_training_steps: int, 
                                   min_lr_ratio: float = 0.1) -> torch.optim.lr_scheduler.LambdaLR:
    """
    create a schedule with a learning rate that decreases following the values of the
    cosine function between the initial lr and 0, with a warmup period at the beginning

    Args:
        optimizer: optimizer for which to schedule the learning rate
        num_warmup_steps: nr of steps for the warmup phase
        num_training_steps: total nr of training steps
        min_lr_ratio: min learning rate ratio compared to the initial lr

    Returns:
        lr scheduler
    """
    def lr_lambda(current_step: int) -> float:
        # warmup 
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # cosine decay 
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        decayed = (1 - min_lr_ratio) * cosine_decay + min_lr_ratio

        return decayed

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def create_optimizer_and_scheduler(model_params: List[torch.nn.parameter.Parameter], 
                                  learning_rate: float, 
                                  num_training_steps: int,
                                  warmup_ratio: float = 0.1, 
                                  weight_decay: float = 0.01) -> tuple:
    """
    create an optimizer and learning rate scheduler with warmup

    Args:
        model_params: parameters of the model to optimize
        learning_rate: max lr
        num_training_steps: total nr of training steps
        warmup_ratio: portion of training to use for warmup
        weight_decay: weight decay coefficient

    Returns:
        tuple of (optimizer, scheduler)
    """
    # optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model_params,
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # scheduler with warmup
    num_warmup_steps = int(num_training_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    return optimizer, scheduler