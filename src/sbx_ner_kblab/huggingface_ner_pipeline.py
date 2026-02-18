"""NER pipeline using HuggingFace pipeline."""

import logging
from collections.abc import Callable, Iterable

from huggingface_hub.utils import logging as hf_logging
from parallel_corpus import graph
from sparv.api import Annotation, Output, get_logger
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
from transformers.pipelines.token_classification import TokenClassificationPipeline

from sbx_ner_kblab import constants
from sbx_ner_kblab.ner_pipeline import NerPipeline

logger = get_logger(__name__)


SENT_SEP = "\n"
TOK_SEP = " "

SCORE_FORMATS = {
    1: "{:.1f}",
    2: "{:.2f}",
    3: "{:.3f}",
    4: "{:.4f}",
    5: "{:.5f}",
    6: "{:.6f}",
    7: "{:.7f}",
    8: "{:.8f}",
    9: "{:.9f}",
    10: "{:.10f}",
}


class HuggingFaceNerPipeline(NerPipeline):
    """A NER pipeline that uses HuggingFace pipeline."""

    def __init__(self, *, num_decimals: int = 5) -> None:
        """Create a HuggingFaceNerPipeline."""
        self.num_decimals = num_decimals
        _configure_third_party_loggers(show_progress=False)
        logger.warning("Load tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(
            constants.TOKENIZER_NAME, revision=constants.TOKENIZER_REVISION
        )
        logger.warning("Load model")
        self.model = AutoModelForTokenClassification.from_pretrained(
            constants.MODEL_NAME, revision=constants.MODEL_REVISION
        )
        logger.warning("Load model_pipeline")
        self.model_pipeline: TokenClassificationPipeline = pipeline(
            "token-classification", model=self.model, tokenizer=self.tokenizer
        )

    def run(
        self,
        sentence: Annotation,
        word: Annotation,
        out_ne_type: Output,
        out_ne_score: Output,
    ) -> None:
        """Run this pipeline."""
        sentences, _orphans = sentence.get_children(word)
        token_word = list(word.read())

        out_type_annotation = word.create_empty_attribute()
        out_score_annotation = word.create_empty_attribute()

        score_format = SCORE_FORMATS[self.num_decimals]
        for sent in sentences:
            sent_to_tag = TOK_SEP.join(token_word[token_index] for token_index in sent)

            for token_index, tag, score in _align_tags_and_tokens(
                _run_nlp_on_sentence(self.model_pipeline, sent_to_tag),
                token_word,
                score_format=score_format,
            ):
                out_type_annotation[token_index] = tag
                out_score_annotation[token_index] = score

        logger.info("writing annotations")
        out_ne_type.write(out_type_annotation)
        out_ne_score.write(out_score_annotation)


def _run_nlp_on_sentence(nlp: Callable, sentence: str) -> list[dict]:
    logger.info("run_nlp_on_sentence(len(sentence)='%d') called", len(sentence))
    return nlp(sentence)


TAGS = {"PER": "PRS"}


def _translate_tag(tag: str) -> str:
    return TAGS.get(tag) or tag


def _configure_third_party_loggers(*, show_progress: bool = False) -> None:
    from huggingface_hub.utils.tqdm import disable_progress_bars  # noqa: PLC0415

    if not show_progress:
        disable_progress_bars()
    disable_progress_bars()
    for logger_name in [
        "transformers",
        "huggingface_hub",
        "nlp",
        "torch",
        "tensorflow",
        "tensorboard",
        "torch.nn",
    ]:
        logging.getLogger(logger_name).setLevel(logging.ERROR)
    hf_logging.set_verbosity(hf_logging.ERROR)


def _align_tags_and_tokens(
    tokens: list[dict],
    token_word: list[str],
    *,
    score_format: str,
) -> Iterable[tuple[int, str, str]]:
    orig_sentence = " ".join(token_word)
    sentence = " ".join(t["word"] for t in tokens)
    graph_aligned = graph.init_with_source_and_target(orig_sentence, sentence)
    for edge in graph_aligned.edges.values():
        if len(edge.ids) > 1:
            s_ids, t_ids = _extract_ids(edge.ids)
            if len(s_ids) == 1:
                s_index = s_ids[0]
                yield (
                    s_index,
                    _extract_entity(tokens, t_ids),
                    score_format.format(_compute_score(tokens, t_ids)),
                )


def _extract_ids(ids: list[str]) -> tuple[list[int], list[int]]:
    t_ids = []
    s_ids = []
    for i in ids:
        if i.startswith("s"):
            s_id = int(i[1:])
            s_ids.append(s_id)
        else:
            t_id = int(i[1:])
            t_ids.append(t_id)
    return s_ids, t_ids


def _extract_entity(tokens: list[dict], t_ids: list[int]) -> str:
    entities = [_translate_tag(tokens[i]["entity"]) for i in t_ids]
    entities_set = set(entities)
    if len(entities_set) == 1:
        return entities[0]
    raise NotImplementedError(f"{entities=}")


def _compute_score(tokens: list[dict], t_ids: list[int]) -> float:
    score = sum(tokens[i]["score"] for i in t_ids)
    score /= len(t_ids)
    return score
