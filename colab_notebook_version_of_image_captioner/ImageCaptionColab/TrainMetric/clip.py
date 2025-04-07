import numpy as np
from base import MetricCalculator
from clip_scorer import CLIPScorer
from grammar_checker import GrammarChecker  

class CLIPCalculator(MetricCalculator):
    """CLIP metric calculator with a grammar penalty."""

    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.clip_scorer = CLIPScorer(model_name=model_name)
        self.grammar_checker = GrammarChecker()  # Initialize grammar checker

    def getName(self):
        return "CLIP"

    def compute(self, reference_captions, generated_captions, image_paths):
        """
        Compute CLIP score with grammar penalty.

        Args:
            reference_captions: Dictionary {image_id: [list of reference captions]}
            generated_captions: Dictionary {image_id: generated caption}
            image_paths: Dictionary {image_id: path to image file}

        Returns:
            Average adjusted CLIP score across all images.
        """
        if not image_paths:
            return None

        adjusted_scores = []

        for img_id in generated_captions:
            if img_id not in image_paths:
                continue

            caption = generated_captions[img_id]
            image_path = image_paths[img_id]

            # Compute CLIP score
            clip_score = self.clip_scorer.compute_score(image_path, caption)

            # Compute grammar penalty
            penalty = self.grammar_checker.compute_penalty(caption)

            # Adjusted score
            adjusted_score = clip_score * penalty
            # print(f"adjusted score for image {img_id}: {adjusted_score}")
            adjusted_scores.append(adjusted_score)

        if not adjusted_scores:
            return None

        return np.mean(adjusted_scores)



    def compute_reference_clip(self, reference_captions, image_paths):
        """
        Compute CLIP score for reference captions without grammar penalty.

        Args:
            reference_captions: Dictionary {image_id: [list of reference captions]}
            image_paths: Dictionary {image_id: path to image file}

        Returns:
            Average CLIP score across all images.
        """
        if not image_paths:
            return None

        # Scores for each image (using the max score from all reference captions)
        image_scores = {}

        for img_id, captions in reference_captions.items():
            if img_id not in image_paths:
                continue

            image_path = image_paths[img_id]
            caption_scores = []

            for caption in captions:
                # Compute CLIP score for this caption (no grammar penalty)
                clip_score = self.clip_scorer.compute_score(image_path, caption)
                caption_scores.append(clip_score)

            if caption_scores:
                # Use the maximum score from all reference captions
                image_scores[img_id] = max(caption_scores)
                print(f"reference score for image {img_id}: {image_scores[img_id]}")

        if not image_scores:
            return None

        return np.mean(list(image_scores.values()))