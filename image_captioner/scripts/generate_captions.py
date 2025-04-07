import os
import sys
import argparse
import torch
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
import torchvision.models as models
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from src.config import MODELS_DIR, IMAGE_DIR, DEVICE
from src.models.decoder import Transformer_Decoder
from src.models.beam_search import beam_search_caption
from src.utils.file_utils import load_model


def parse_args():
    """
    command line arguments
    """
    parser = argparse.ArgumentParser(description="Generate captions for images")
    
    # paths
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to trained model")
    parser.add_argument("--image-dir", type=str, default=None,
                       help="Directory containing images to caption")
    parser.add_argument("--image-path", type=str, default=None,
                       help="Path to a single image to caption")
    parser.add_argument("--output-file", type=str, default="generated_captions.tsv",
                       help="File to save generated captions")
    
    # model parameters
    parser.add_argument("--embed-size", type=int, default=256,
                       help="Size of word embeddings")
    parser.add_argument("--hidden-size", type=int, default=768,
                       help="Size of hidden layers")
    parser.add_argument("--num-layers", type=int, default=6,
                       help="Number of transformer layers")
    parser.add_argument("--beam-width", type=int, default=5,
                       help="Beam width for beam search")
    
    # other parameters
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    return parser.parse_args()


def load_vocab(model_dir):
    """
    load vocabulary from model 
    
    Args:
        model_dir: directory with the model and vocab files
        
    Returns:
        tuple of (word2idx, idx2word)
    """
    
    # try to find vocabulary.json 
    vocab_json_path = os.path.join(os.path.dirname(model_dir), "vocabulary.json")
    if not os.path.exists(vocab_json_path):
        vocab_json_path = os.path.join(os.path.dirname(os.path.dirname(model_dir)), "vocabulary.json")
    
    if os.path.exists(vocab_json_path):
        # load vocabulary file
        print(f"loading vocabulary from {vocab_json_path}")
        import json
        with open(vocab_json_path, 'r') as f:
            vocab_dict = json.load(f)
        
        word2idx = vocab_dict["word2idx"]
        idx2word = vocab_dict["idx2word"]
        
        # convert idx2word keys to integers 
        if isinstance(next(iter(idx2word)), str):
            idx2word = {int(k): v for k, v in idx2word.items()}
        
        print(f"loaded vocabulary with {len(word2idx)} words")
    elif os.path.exists(vocab_path):
        # load PyTorch vocabulary file
        vocab_dict = torch.load(vocab_path)
        word2idx = vocab_dict["word2idx"]
        idx2word = vocab_dict["idx2word"]
        print(f"Loaded vocabulary with {len(word2idx)} words")
    else:
        # create default vocabulary
        print("vocabulary file not found. creating default vocabulary.")
        word2idx = {
            "<PAD>": 0,
            "< SOS >": 1,
            "<EOS>": 2,
            "<UNK>": 3
        }
        # added dummy entries to match model's expected size of 4904
        for i in range(4, 4904):
            word2idx[f"word_{i}"] = i
        
        idx2word = {v: k for k, v in word2idx.items()}
    
    return word2idx, idx2word


def extract_image_features(image_path, model, transform):
    """
    extract features from an image
    
    Args:
        image_path: path to the image
        model: feature extraction model
        transform: image transformation
        
    Returns:
        extracted features as a tensor
    """
    try:
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            features = model(image).squeeze(-1).squeeze(-1).cpu()
        return features
    except Exception as e:
        print(f"errror extracting features from {image_path}: {e}")
        return None


def main():
    """
    main function for generating captions
    """

    args = parse_args()
    
    # random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # make sure inputs are valid
    if args.image_dir is None and args.image_path is None:
        raise ValueError("either --image-dir or --image-path must be specified")
    
    word2idx, idx2word = load_vocab(args.model_path)
    
    # lload model
    decoder, project_features, _ = load_model(
        args.model_path,
        args.embed_size,
        len(word2idx),
        args.hidden_size,
        args.num_layers,
        DEVICE
    )
    
    # feature extraction
    feature_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    feature_model = torch.nn.Sequential(*list(feature_model.children())[:-1])
    feature_model.eval().to(DEVICE)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    results = []
    
    if args.image_path is not None:
        # process a single image
        features = extract_image_features(args.image_path, feature_model, transform)
        if features is not None:
            caption = beam_search_caption(
                features, decoder, project_features,
                word2idx, idx2word, DEVICE, beam_width=args.beam_width
            )
            image_name = os.path.basename(args.image_path)
            results.append((image_name, caption))
            print(f"\nImage: {image_name}")
            print(f"Caption: {caption}")
    
    if args.image_dir is not None:
        # process all images in directory
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            image_files.extend(list(Path(args.image_dir).glob(f"*{ext}")))
        
        print(f"Found {len(image_files)} images in {args.image_dir}")
        
        for image_file in tqdm(image_files, desc="Generating captions"):
            features = extract_image_features(image_file, feature_model, transform)
            if features is not None:
                caption = beam_search_caption(
                    features, decoder, project_features,
                    word2idx, idx2word, DEVICE, beam_width=args.beam_width
                )
                image_name = os.path.basename(image_file)
                results.append((image_name, caption))
    
    # results
    if results:
        output_path = args.output_file
        df = pd.DataFrame(results, columns=["image", "caption"])
        df.to_csv(output_path, sep='\t', index=False)
        print(f"\nGenerated {len(results)} captions")
        print(f"Saved captions to {output_path}")
    else:
        print("No captions were generated")


if __name__ == "__main__":
    main()