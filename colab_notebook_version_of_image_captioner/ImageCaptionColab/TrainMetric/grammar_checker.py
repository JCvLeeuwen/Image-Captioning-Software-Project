import language_tool_python
import numpy as np


class GrammarChecker:
    """
    Class to check grammar correctness and provide a penalty score.
    """

    def __init__(self, language="en"):
        """
        Initialize the grammar checker.

        Args:
            language: Language code (default: English).
        """
        self.tool = language_tool_python.LanguageTool(language)

    def compute_penalty(self, sentence):
        """
        Compute a grammar penalty score between 0 and 1.

        Args:
            sentence: The generated caption to evaluate.

        Returns:
            A penalty multiplier between 0 and 1 (1 = perfect grammar, 0 = very poor grammar).
        """
        matches = self.tool.check(sentence)
        num_errors = len(matches)

        # Simple heuristic: more errors = lower score
        penalty = np.exp(-0.1 * num_errors)  # Decay function, tweak as needed

        return penalty
