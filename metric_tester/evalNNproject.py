
import nltk
import json
import numpy as np
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
from pythspicesim import Spice
from nltk.translate.meteor_score import meteor_score
from clip import CLIPScorer


# Ensure necessary NLTK resources are available
nltk.download('wordnet')
nltk.download('punkt')


class CaptionEvaluator:
    def __init__(self, reference_captions, generated_captions):
        """
        :param reference_captions: Dictionary {image_id: [list of human-written captions]}
        :param generated_captions: Dictionary {image_id: generated caption}
        """
        self.reference_captions = reference_captions
        self.generated_captions = generated_captions
        


    def compute_bleu(self):
        references = [self.reference_captions[img_id] for img_id in self.generated_captions]
        hypotheses = [self.generated_captions[img_id] for img_id in self.generated_captions]

        # Smoothing function to avoid zero BLEU scores for lower-order n-grams
        smoothie = SmoothingFunction().method4  # You can choose other methods like method1, method2, etc.

        bleu_score = corpus_bleu([[ref.split() for ref in refs] for refs in references],
                                 [hyp.split() for hyp in hypotheses], smoothing_function=smoothie)
        return bleu_score

    def compute_meteor(self):
        scores = []
        for img_id in self.generated_captions:
            # Tokenize reference captions (which are lists of captions) and the generated caption (which is a single string)
            reference_tokens = [word_tokenize(ref) for ref in self.reference_captions[img_id]]
            generated_tokens = word_tokenize(self.generated_captions[img_id])

            # Compute METEOR score for each reference caption vs. generated caption
            score = meteor_score(reference_tokens, generated_tokens)
            scores.append(score)

        return np.mean(scores)

    def compute_rouge(self):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        all_scores = {"rouge1": [], "rouge2": [], "rougeL": []}

        for img_id in self.generated_captions:
            reference_captions = self.reference_captions[img_id]
            generated_caption = self.generated_captions[img_id]

            image_scores = []
            for reference_caption in reference_captions:
                score = scorer.score(reference_caption, generated_caption)
                image_scores.append(score)

            # Compute average for this image
            avg_score = {key: np.mean([s[key].fmeasure for s in image_scores]) for key in image_scores[0].keys()}

            # Store in the overall list
            for key in all_scores.keys():
                all_scores[key].append(avg_score[key])

        # Compute final averages across all images
        final_avg_scores = {key: np.mean(all_scores[key]) for key in all_scores.keys()}

        return final_avg_scores

    def compute_spice(self):
        spice_scores = []
        spice_scorer = Spice(model="en_core_web_md")

        for img_id in self.generated_captions:
            predicted_caption = self.generated_captions[img_id]
            reference_captions = self.reference_captions[img_id]

            pred_tuples = spice_scorer.extract_tuples(predicted_caption)
            ref_tuples = [spice_scorer.extract_tuples(ref) for ref in reference_captions]

            spice_score = spice_scorer.compute_spice_score(pred_tuples, ref_tuples)
            spice_scores.append(spice_score["spice_score"])  # Use the overall SPICE score

        return np.mean(spice_scores)  # Return average SPICE score across all images

    def compute_CLIP(self, image_folder="./images/"):
    
        clip_scorer = CLIPScorer()
        scores = []
        
        for img_id in self.generated_captions:
            img_path = f"{image_folder}/{img_id}.jpg"
            caption = self.generated_captions[img_id]
            
            score = clip_scorer.compute_similarity(img_path, caption)
            scores.append(score)
        
        # Return average CLIP score
        return np.mean(scores) if scores else None
    



    def evaluate(self):
        results = {
            "BLEU": self.compute_bleu(),
            "METEOR": self.compute_meteor(),
            "ROUGE": self.compute_rouge(),
            "SPICE-like": self.compute_spice(),
            "CLIP": self.compute_CLIP()
        }
        return results





# Example Usage
if __name__ == "__main__":
    # Example captions (Replace with real data when available)
    reference_captions = {
        "img1": ["A cat sitting on a couch."],
        "img2": ["A person riding a bicycle on a road."],
        "img3": ["A boy riding a skateboard"],
        "img4": ["A cartoon of a dancing cat"],
    }
    generated_captions = {
        "img1": "A cat is on a sofa.",
        "img2": "A man is riding a bike on the road.",
        "img3": "A child playing on a skateboard",
        "img4": "A cartoon of a cat dancing",
    }

    

    evaluator = CaptionEvaluator(reference_captions, generated_captions)
    results = evaluator.evaluate()
    print(json.dumps(results, indent=4))
