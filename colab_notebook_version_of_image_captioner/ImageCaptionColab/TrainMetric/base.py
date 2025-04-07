from abc import ABC, abstractmethod


class MetricCalculator(ABC):
    """Base abstract class for all caption evaluation metrics."""

    @abstractmethod
    def getName(self):
        """Returns the name of the metric."""
        pass

    @abstractmethod
    def compute(self, reference_captions, generated_captions):
        """
        Computes the metric between reference and generated captions.

        Args:
            reference_captions: Dictionary {image_id: [list of reference captions]}
            generated_captions: Dictionary {image_id: generated caption}

        Returns:
            Metric value or dictionary of metric values
        """
        pass