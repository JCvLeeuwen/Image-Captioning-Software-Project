import spacy
import numpy as np

class Spice:
    def __init__(self, model="en_core_web_md"):
        """Initialize the SPICE scorer with a spaCy model."""
        self.nlp = spacy.load(model)

    def word_similarity(self, word1, word2):
        """Return similarity score between two words using word embeddings."""
        token1, token2 = self.nlp(word1), self.nlp(word2)

        if not token1.has_vector or not token2.has_vector:
            return 0.0  # If no vector, return minimum similarity

        return token1.similarity(token2)  # Return similarity score

    def match_with_embeddings(self, word, reference_set):
        """Find the best similarity score between a word and a set of reference words."""
        if not reference_set:
            return 0.0  # No reference words to compare with

        return max(self.word_similarity(word, ref) for ref in reference_set)

    def extract_tuples(self, sentence):
        """Extract objects and relationships from a sentence using dependency parsing."""
        doc = self.nlp(sentence)
        objects = set()
        relations = set()

        for token in doc:
            if token.dep_ in {"nsubj", "dobj", "pobj", "attr"}:  # Extract subjects & objects
                objects.add(token.lemma_)
            elif token.dep_ in {"prep", "acl", "nmod"}:  # Extract prepositional and nominal relations
                relations.add((token.head.lemma_, token.lemma_, token.head.head.lemma_))

        return {"objects": objects, "relations": relations}

    def compute_spice_score(self, predicted, references):
        """Compute SPICE-like score using word embeddings for semantic similarity."""
        pred_objects = predicted["objects"]
        pred_relations = predicted["relations"]

        ref_objects = set().union(*(r["objects"] for r in references))
        ref_relations = set().union(*(r["relations"] for r in references))

        # Object Matching with Similarity Scores
        object_similarities = [self.match_with_embeddings(obj, ref_objects) for obj in pred_objects]
        object_precision = sum(object_similarities) / max(len(pred_objects), 1)
        object_recall = sum(object_similarities) / max(len(ref_objects), 1)

        # Relation Matching with Similarity Scores
        relation_similarities = [
            (self.word_similarity(rel[0], r[0]) + self.word_similarity(rel[1], r[1]) + self.word_similarity(rel[2], r[2])) / 3
            for rel in pred_relations for r in ref_relations
        ]
        relation_precision = sum(relation_similarities) / max(len(pred_relations), 1)
        relation_recall = sum(relation_similarities) / max(len(ref_relations), 1)

        # Compute F1 scores
        object_f1 = 2 * (object_precision * object_recall) / max((object_precision + object_recall), 1e-8)
        relation_f1 = 2 * (relation_precision * relation_recall) / max((relation_precision + relation_recall), 1e-8)

        return {"object_f1": object_f1, "relation_f1": relation_f1, "spice_score": (object_f1 + relation_f1) / 2}

# Example Usage
if __name__ == "__main__":
    spice_scorer = Spice(model="en_core_web_md")

    pred_caption = "A dog sits on a bench."
    ref_captions = ["A dog is resting on a bench.", "A brown dog sits on a wooden bench.", "A labrador sits on a bench. "]

    pred_tuples = spice_scorer.extract_tuples(pred_caption)
    ref_tuples = [spice_scorer.extract_tuples(ref) for ref in ref_captions]

    score = spice_scorer.compute_spice_score(pred_tuples, ref_tuples)
    print(score)
