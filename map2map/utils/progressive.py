import math


def get_progressive_alpha(epoch, total_epochs_per_scale=100, fade_in_ratio=0.5):
    """Calculate progressive alpha value for smooth layer transitions.
    
    Args:
        epoch: Current epoch within this scale
        total_epochs_per_scale: Total epochs to train at this scale
        fade_in_ratio: Fraction of epochs to use for fading in (0.5 = half)
    
    Returns:
        alpha: Value between 0 and 1 for blending
    """
    fade_in_epochs = int(total_epochs_per_scale * fade_in_ratio)
    
    if epoch < fade_in_epochs:
        # Linear fade in from 0 to 1
        alpha = epoch / fade_in_epochs
    else:
        # Fully faded in
        alpha = 1.0
        
    return alpha


def get_progressive_schedule(max_scale_factor, base_epochs=100, scale_multiplier=1.5):
    """Generate training schedule for progressive growing.
    
    Args:
        max_scale_factor: Maximum upsampling factor (e.g., 8 for 32->256)
        base_epochs: Epochs to train at scale 2
        scale_multiplier: How much to increase epochs for each scale
        
    Returns:
        schedule: List of (scale_factor, epochs) tuples
    """
    schedule = []
    current_scale = 2
    current_epochs = base_epochs
    
    while current_scale <= max_scale_factor:
        schedule.append((current_scale, int(current_epochs)))
        current_scale *= 2
        current_epochs *= scale_multiplier
        
    return schedule


def update_model_alpha(model, alpha):
    """Update progressive alpha in model if it supports it."""
    if hasattr(model, 'module'):
        # Handle DistributedDataParallel
        if hasattr(model.module, 'progressive_alpha'):
            model.module.progressive_alpha = alpha
    elif hasattr(model, 'progressive_alpha'):
        model.progressive_alpha = alpha