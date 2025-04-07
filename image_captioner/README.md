
A standalone application for training and using image captioning models. This application transforms images into natural language descriptions.

I reduced the amount of images provided in the folder to 10K images, instead of the full 140K images I used. This is to reduce the size of the files to be handed in/uploaded/downloaded, while still allowing this system to be tested. The results of training on this small subset of the larger dataset is not representative of the results achieved on the full 140K image-caption pairs. The provided dataset serves merely to showcase the functionality of the code.

## Features

- Train image captioning models on your own image datasets
- Generate captions for new images
- Multi-phase training with CLIP-based optimization
- Beam search caption generation for outputs
- Model checkpointing and comprehensive evaluation metrics
- Efficient feature caching to speed up repeat training

## Installation

### Prerequisites

- Python 3.7+
- PyTorch 1.9+
- CUDA-capable GPU recommended for training

### Setup

1. install dependencies directly:

```bash
pip install -r requirements.txt
```

## Project structure

```
image-captioning/
├── data/                  # data storage
│   ├── images/            # image files
│   ├── captions/          # caption files (TSV format)
│   └── models/            # trained models
├── src/                   # source code
│   ├── models/            # model definitions
│   ├── data/              # dataset and data preparation
│   ├── training/          # training utilities
│   ├── evaluation/        # evaluation metrics
│   └── utils/             # utility functions
└── scripts/               # command-line scripts
    ├── train_model.py     # training script
    ├── generate_captions.py # caption generation
    └── validate_images.py # image validation 
```

## Quick start

### Data Preparation

Make sure the images are placed in the `data/images/` directory and the captions in a TSV file at `data/captions/filtered_captions.tsv`. The TSV file (or any custom captions tsv file) should have two columns:

```
image_id    caption
image1.jpg  A person walking on the beach at sunset.
image2.jpg  A dog playing with a ball in the park.
```

### Training a model

To train a basic model:

```bash
python scripts/train_model.py --max-images 5000 --epochs 10
```

For multi-phase training with CLIP optimization:

```bash
python scripts/train_model.py --max-images 5000 --epochs 30 --multi-phase
```

### Generating captions

To generate captions for new images:

```bash
python scripts/generate_captions.py --model-path data/models/best_model.pt --image-dir path/to/images
```

Or for a single image:

```bash
python scripts/generate_captions.py --model-path data/models/best_model.pt --image-path path/to/image.jpg
```

### Validating Images

To check for and remove corrupted images:

```bash
python scripts/validate_images.py --image-dir data/images
```

## Advanced Usage

### Custom training configuration

Multi-phase:

```bash
python scripts/train_model.py  --captions-file data/captions/custom_captions.tsv --image-dir data/custom_images --output-dir data/models/custom_experiment --max-images 10000 --batch-size 32 --embed-size 256 --hidden-size 768 --num-layers 6 --learning-rate 0.0003 --clip-loss-weight 0.3 --multi-phase
```
Single-phase:

```bash
python scripts/train_model.py  --captions-file data/captions/custom_captions.tsv --image-dir data/custom_images --output-dir data/models/custom_experiment --max-images 10000 --batch-size 32 --embed-size 256 --hidden-size 768 --num-layers 6 --learning-rate 0.0003 --clip-loss-weight 0.3 
```

### Continuing training from a checkpoint

```bash
python scripts/train_model.py --model-path data/models/previous_model.pt --epochs 10 --learning-rate 0.0001
```

## Model architecture

The image captioning model consists of:

1. a ResNet18 image encoder (pre-extracted features)
2. a linear projection layer
3. a transformer decoder with self-attention
4. beam search decoding for inference

The model is trained using a combination of:
- cross-entropy loss for word prediction
- CLIP-based reward optimization using policy gradients

