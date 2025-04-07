import os
import torch


def load_references(tsv_path, image_dir, filter_ids=None):
    """
    Load reference captions from a .tsv file and map them to image paths.
    Optionally filter to only include specified image IDs.

    Args:
        tsv_path: Path to the .tsv file containing reference captions.
        image_dir: Directory where images are stored.
        filter_ids: Optional set of image IDs to include (for filtering)

    Returns:
        reference_captions: Dictionary {image_id: [list of reference captions]}
        image_paths: Dictionary {image_id: path to image file}
    """
    reference_captions = {}
    image_paths = {}

    with open(tsv_path, "r", encoding="utf-8") as file:
        for line in file:
            row = line.strip().split("\t")  # Split on tab

            if len(row) < 2:
                continue  # Skip malformed lines

            img_id, caption = row[0], row[1]  # First column is image ID, second is reference caption
            
            # Skip if not in our filter (if filter is provided)
            if filter_ids is not None and img_id not in filter_ids:
                continue
                
            image_path = os.path.join(image_dir, f"{img_id}.jpg")  # Assuming image filenames match img_id

            if not os.path.exists(image_path):
                continue  # Skip if the image is missing

            if img_id not in reference_captions:
                reference_captions[img_id] = []  # Initialize list if first time seeing this image_id

            reference_captions[img_id].append(caption)  # Append caption to the list
            image_paths[img_id] = image_path  # Store image path

    return reference_captions, image_paths

def evaluate_clip(model, dataloader, clip_calculator, tsv_path, image_dir, device):
    """
    Evaluate the model using CLIP score.

    Args:
        model: The image captioning model.
        dataloader: DataLoader providing (image, image_id).
        clip_calculator: An instance of CLIPCalculator.
        tsv_path: Path to the .tsv file containing reference captions.
        image_dir: Directory where images are stored.
        device: The device to run the model on.

    Returns:
        Average CLIP score across the dataset.
    """
    model.eval()  # Set model to evaluation mode
    generated_captions = {}

    # Load reference captions and image paths
    reference_captions, image_paths = load_references(tsv_path, image_dir)

    with torch.no_grad():
        for images, img_ids in dataloader:
            images = images.to(device)

            # Generate captions using the model
            outputs = model.generate(images)
            captions = [model.decode(output) for output in outputs]

            for img_id, caption in zip(img_ids, captions):
                img_id = str(img_id.item())  # Ensure it's a string
                if img_id in image_paths:  # Skip if no corresponding reference/image
                    generated_captions[img_id] = caption

    # Compute CLIP score with grammar penalty for generated captions
    generated_clip_score = clip_calculator.compute(reference_captions, generated_captions, image_paths)

    # Compute CLIP score for reference captions (without grammar penalty)
    reference_clip_score = clip_calculator.compute_reference_clip(reference_captions, image_paths)

    return {
        "generated_score": generated_clip_score,
        "reference_score": reference_clip_score
    }


"""
clip_calculator.compute() works as follows: 

It computes the CLIP score with grammar penalty.

Args:
    reference_captions: Dictionary {image_id: [list of reference captions]}
    generated_captions: Dictionary {image_id: generated caption}
    image_paths: Dictionary {image_id: path to image file}

Returns:
    Average adjusted CLIP score across all images.
"""


'''
Modify training loop with something like: 

tsv_path = "path/to/training.tsv"
image_dir = "path/to/image/folder"
clip_calculator = CLIPCalculator()  # Initialize CLIP metric

for epoch in range(num_epochs):
    train(...)  # Your training function

    if epoch % eval_interval == 0:  # Evaluate at intervals
        clip_scores = evaluate_clip(model, val_dataloader, clip_calculator, tsv_path, image_dir, device)
        print(f"Epoch {epoch}: Generated CLIP Score = {clip_scores['generated_score']:.4f}, Reference CLIP Score = {clip_scores['reference_score']:.4f}")
'''