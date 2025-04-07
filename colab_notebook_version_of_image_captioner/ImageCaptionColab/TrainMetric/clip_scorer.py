import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import math

class CLIPScorer:

    def __init__(self, model_name="openai/clip-vit-base-patch32"):

        """Initializes the CLIP model and processor."""

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = CLIPModel.from_pretrained(model_name).to(self.device)

        self.processor = CLIPProcessor.from_pretrained(model_name)



    def compute_score(self, image_path, caption):
        """Computes CLIPScore for a given image-caption pair (normalized 0-100)."""
        try:
            # Load image and ensure it's RGB (3 channels)
            image = Image.open(image_path).convert("RGB")
            
            # Process with CLIP processor
            inputs = self.processor(
                text=[caption],
                images=image, 
                return_tensors="pt", 
                padding=False,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                image_embeds = outputs.image_embeds  # (batch_size, embed_dim)
                text_embeds = outputs.text_embeds    # (batch_size, embed_dim)

                # Compute cosine similarity
                similarity_score = F.cosine_similarity(image_embeds, text_embeds).item()
                print(f"cosine sim: {similarity_score}")
                
                # Ensure value is in valid range for arccos
                similarity_score = max(min(similarity_score, 1.0), -1.0)
                
                # Convert to angular similarity
                angular_distance = math.acos(similarity_score) / math.pi
                angular_similarity = 1 - angular_distance

            return angular_similarity
        
        except ValueError as e:
            if "mean must have" in str(e):
                # This likely means the image processor expects a different number of channels
                print(f"Warning: Channel mismatch error for {image_path}. Trying alternative approach...")
                
                # Alternative approach: manually process the image with the expected parameters
                image = Image.open(image_path).convert("RGB")
                
                # Try using just the pixel values without normalization
                pixel_values = self.processor.feature_extractor(
                    images=image,
                    return_tensors="pt",
                    do_normalize=False,  # Skip normalization
                ).pixel_values.to(self.device)
                
                # Process text separately
                text_inputs = self.processor.tokenizer(
                    text=[caption],
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)
                
                # Manual forward pass
                with torch.no_grad():
                    image_features = self.model.get_image_features(pixel_values=pixel_values)
                    text_features = self.model.get_text_features(**text_inputs)
                    
                    # Normalize features
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    
                    # Compute similarity
                    similarity_score = torch.matmul(text_features, image_features.T).item()
                    # print(f"cosine sim (alternative): {similarity_score}")
                    
                    similarity_score = max(min(similarity_score, 1.0), -1.0)
                    
                    # Convert to angular similarity
                    angular_distance = math.acos(similarity_score) / math.pi
                    angular_similarity = 1 - angular_distance
                    
                return angular_similarity
            
            else:
                # If it's a different error, re-raise it
                raise



# Example usage
if __name__ == "__main__":
    scorer = CLIPScorer()
    image_path = "cat.png"
    caption = "kitty"
    clip_score = scorer.compute_score(image_path, caption)
    print(f"CLIP angular similarity Score: {clip_score:.2f}")
