"""NER pipeline using HuggingFace pipeline."""

import logging
from collections.abc import Callable, Iterable

from huggingface_hub.utils import logging as hf_logging
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
                sent,  # sent_to_tag
                score_format=score_format,
            ):
                out_type_annotation[token_index] = tag
                out_score_annotation[token_index] = score

        logger.info("writing annotations")
        out_ne_type.write(out_type_annotation)
        out_ne_score.write(out_score_annotation)

    # def _run_model(
    #     model_pipeline,
    #     sentence: Annotation,
    #     word: Annotation,
    #     out_ne_type: Output,
    #     out_ne_score: Output,
    # ) -> None:
    # sentences, _orphans = sentence.get_children(word)
    # token_word = list(word.read())

    # out_type_annotation = word.create_empty_attribute()
    # out_score_annotation = word.create_empty_attribute()

    # for sent in sentences:
    #     sent_to_tag = TOK_SEP.join(token_word[token_index] for token_index in sent)

    #     for token_index, tag, score in _align_tags_and_tokens(
    #         _run_nlp_on_sentence(model_pipeline, sent_to_tag),
    #         token_word,
    #         sent,  # sent_to_tag
    #     ):
    #         out_type_annotation[token_index] = tag
    #         out_score_annotation[token_index] = score

    # logger.info("writing annotations")
    # out_ne_type.write(out_type_annotation)
    # out_ne_score.write(out_score_annotation)


def _run_nlp_on_sentence(nlp: Callable, sentence: str) -> list[dict]:
    logger.info("run_nlp_on_sentence(len(sentence)='%d') called", len(sentence))
    tokens = []
    # linking_token = ""
    found_link = False
    for token in nlp(sentence):
        # logger.debug("nlp: token = %s", token)
        if token["word"].startswith("##"):
            # logger.debug("found ## in %s", token["word"])
            if tokens and token["start"] == tokens[-1]["end"]:
                # tokens[-1]["word"] += token["word"][2:]
                tokens[-1]["end"] = token["end"]
            # else:
            #     logger.info(
            #         "found ## in '%s' (start=%d, end=%d) but no prev token, skipping ... ",
            #         token["word"],
            #         token["start"],
            #         token["end"],
            #     )
            # tokens.append(Token(**token))
        elif found_link:
            # logger.debug("found '%s' before word '%s'", linking_token, token["word"])
            found_link = False
            if token["start"] == tokens[-1]["end"]:
                # tokens[-1]["word"] += token["word"]
                tokens[-1]["end"] = token["end"]
                # amend_token_to_last(token, tokens)
            else:
                tokens.append(token)
        elif token["word"] in {",", ".", "-"}:
            # linking_token = token["word"]
            # logger.debug("found '%s'", linking_token)
            found_link = True
            if token["start"] == tokens[-1]["end"]:
                # tokens[-1]["word"] += token["word"]
                tokens[-1]["end"] = token["end"]
                # amend_token_to_last(token, tokens)
            else:
                tokens.append(token)
        else:
            tokens.append(token)
        # logger.debug("tokens[-1].word = %s", tokens[-1].word)
        # logger.debug("tokens = %s", tokens)
    return tokens


def _amend_token_to_last(token: dict, tokens: list[dict]) -> None:
    # logger.debug("adding '%s' to last word %s", token["word"], tokens[-1]["word"])
    tokens[-1]["word"] += token["word"]
    tokens[-1]["end"] = token["end"]


def _align_tags_and_tokens(
    tokens: list[dict],
    token_word: list[str],
    sent: list[int],  # , sentence: str
    *,
    score_format: str,
) -> Iterable[tuple[int, str, str]]:
    # ) -> Iterable[TaggedToken]:
    # logger.info("align_tags_and_tokens.tokens = %s", tokens)
    # logger.debug("interleave_tags_and_tokens.sentence = %s", sentence)
    curr_token = 0
    if curr_token >= len(tokens):
        return
    # end = len(sentence)
    curr_sent = 0
    curr_word_start = 0
    # curr_word_end = len(sentence)
    token_start = tokens[curr_token]["start"]
    for curr_sent in sent:
        curr_word_end = curr_word_start + len(token_word[curr_sent])

        # have we found curr_token?
        if curr_word_start <= token_start < curr_word_end:
            yield (
                curr_sent,
                _translate_tag(tokens[curr_token]["entity"]),
                score_format.format(tokens[curr_token]["score"]),
            )

            curr_token += 1
            if curr_token >= len(tokens):
                break
            token_start = tokens[curr_token]["start"]
        # else:
        #     yield ("", "")
        # yield TaggedToken.default()
        # update curr_word
        curr_word_start += len(token_word[curr_sent]) + 1
        # curr_sent += 1


def find_word_ending(sentence: str, token_end: int) -> int:
    """Find first whitespace from token_end or above."""
    while token_end < len(sentence) and sentence[token_end] != " ":
        token_end += 1
    return token_end


def _find_next_token(tokens: list, curr_token: int, curr: int) -> int:
    next_token = curr_token + 1
    while next_token < len(tokens) and tokens[next_token].start < curr:
        next_token += 1
    return next_token


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
