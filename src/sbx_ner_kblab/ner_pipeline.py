"""NerPipeline interface used by annotators."""

import abc

from sparv.api import Annotation, Output


class NerPipeline(abc.ABC):
    """The required interface of a NER pipeline."""

    @abc.abstractmethod
    def run(
        self,
        sentence: Annotation,
        word: Annotation,
        out_ne_type: Output,
        out_ne_score: Output,
    ) -> None:
        """Run the NER pipeline on the given sentence and record types and scores."""
