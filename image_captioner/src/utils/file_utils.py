import os
import torch
import glob
from datetime import datetime
from typing import Dict, Any, Optional, Union

from src.models.decoder import Transformer_Decoder


def save_model(decoder: Transformer_Decoder, 
              project_features: torch.nn.Module, 
              metrics: Dict[str, Any], 
              output_path: str, 
              model_type: str = "checkpoint") -> str:
    """
    save model with comprehensive metadata

    Args:
        decoder: dcoder model
        project_features: features projection layer
        metrics: dictionary of metrics to save with the model
        output_path: base path for saving
        model_type: type of model being saved (checkpoint, best_val, best_clip)

    Returns:
        path for saved model file
    """
    import os
    import torch
    from datetime import datetime

    # create a timestamped filename so i can finally keep track properly
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if model_type == "checkpoint":
        filename = f"{output_path}_{model_type}_{timestamp}.pt"
    else:
        filename = f"{output_path}_{model_type}.pt"

    # prepare model save dictionary with metadata
    save_dict = {
        'decoder': decoder.state_dict(),
        'project_features': project_features.state_dict(),
        'metrics': metrics,
        'timestamp': timestamp,
        'type': model_type
    }

    # create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # save model
    print(f"Saving {model_type} model to {filename}...")
    torch.save(save_dict, filename)

    # if this is a best model, also save a copy to the standard path
    if model_type in ["best_val", "best_clip"]:
        standard_path = f"{output_path}.pt"
        print(f"Also saving to standard path: {standard_path}")
        torch.save(save_dict, standard_path)

    return filename


def load_model(model_path: str, 
              embed_size: int, 
              vocab_size: int, 
              hidden_size: int, 
              num_layers: int, 
              device: str) -> tuple:
    """
    load a saved model

    Args:
        model_path: path to the saved model
        embed_size: embedding dimension
        vocab_size: size of vocabulary
        hidden_size: hidden dimension
        num_layers: nr of layers
        device: device 

    Returns:
        tuple of (decoder, project_features, metadata)
    """
    import os
    import torch

    print(f"loading model from {model_path}...")

    try:
        # try loading the model with weights_only=False
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        except TypeError:
            # older versions of PyTorch don't have weights_only parameter
            checkpoint = torch.load(model_path, map_location=device)

        # initialize the models
        decoder = Transformer_Decoder(embed_size, vocab_size, hidden_size, num_layers).to(device)
        project_features = torch.nn.Linear(512, embed_size).to(device)  # assuming 512 for resnet18

        # load weights
        decoder.load_state_dict(checkpoint['decoder'])
        project_features.load_state_dict(checkpoint['project_features'])

        # extract metadata if available
        metadata = {}
        for key in ['metrics', 'timestamp', 'type']:
            if key in checkpoint:
                metadata[key] = checkpoint[key]

        print(f"model loaded successfully!")
        if 'type' in metadata:
            print(f"model type: {metadata['type']}")
        if 'timestamp' in metadata:
           
           
            print(f"saved on: {metadata['timestamp']}")

        return decoder, project_features, metadata

    except Exception as e:
        print(f"error loading model: {e}")
        print("initializing new model instead...")

        # initialize new models if there was none found
        decoder = Transformer_Decoder(embed_size, vocab_size, hidden_size, num_layers).to(device)
        
        # determine feature dimension based on the error message if possible
        feature_dim = 512 
        if "size mismatch" in str(e):
            # try to extract the expected dimension from the error message
            try:
                import re
                match = re.search(r'size (\d+)', str(e))
                if match:
                    feature_dim = int(match.group(1))
                    print(f"detected feature dimension from error: {feature_dim}")
            except:
                pass
                
        project_features = torch.nn.Linear(feature_dim, embed_size).to(device)

        return decoder, project_features, {}


def find_best_model(base_path: str, model_type: str = "best_clip") -> Optional[str]:
    """
    find the best saved model of a given type automatically

    Args:
        base_path: base path where models are saved
        model_type: type of model to find (best_val, best_clip)

    Returns:
        path to the best model, or None if no model found
    """
    import os
    import glob

    # look for exact match first
    exact_path = f"{base_path}_{model_type}.pt"
    if os.path.exists(exact_path):
        return exact_path

    # look for timestamped versions
    pattern = f"{base_path}_{model_type}_*.pt"
    matches = glob.glob(pattern)

    if matches:
        # sort by modification time (most recent first)
        matches.sort(key=os.path.getmtime, reverse=True)
        return matches[0]

    # default to standard path
    standard_path = f"{base_path}.pt"
    if os.path.exists(standard_path):
        return standard_path

    return None