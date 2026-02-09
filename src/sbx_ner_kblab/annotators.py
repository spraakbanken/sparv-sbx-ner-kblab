"""Annotators for Sparv."""

import typing as t

from sparv.api import (
    Annotation,
    Config,
    Output,
    SparvErrorMessage,
    annotator,
    get_logger,
)

from sbx_ner_kblab.constants import PROJECT_NAME
from sbx_ner_kblab.ner_pipeline import NerPipeline

logger = get_logger(__name__)


SENT_SEP = "\n"
TOK_SEP = " "


def ner_pipeline_preloader(pipeline: str) -> NerPipeline:
    """Load the request pipeline."""
    # from sbx_named_entities_kb_ner.custom_ner_pipeline import CustomNerPipeline
    from sbx_ner_kblab.huggingface_ner_pipeline import (  # noqa: PLC0415
        HuggingFaceNerPipeline,
    )

    if pipeline == "huggingface_ner":
        return t.cast(NerPipeline, HuggingFaceNerPipeline())
    # if pipeline == "custom_ner":
    #     return t.cast(NerPipeline, CustomNerPipeline())
    raise SparvErrorMessage(f"Unknown pipeline '{pipeline}'")


@annotator(
    "Named entity tagging with KBLab/bert-base-lowermix-swedish-lowermix-reallysimple-ner",
    language=["swe"],
    preloader=ner_pipeline_preloader,
    preloader_params=["pipeline"],
    preloader_target="model_preloaded",
    config=[
        Config(f"{PROJECT_NAME}.pipeline", description="HuggingFace pipeline to use"),
    ],
)
def annotate_with_bert_base_swedish_lowermix_reallysimple_ner(
    out_ne_type: Output = Output(
        f"<token>:{PROJECT_NAME}.bert_base_swedish_lowermix_reallysimple_ner_ne_type",
        cls="named_entity",
        description="Named entity segment types from KBLab/bert-base-lowermix-swedish-lowermix-reallysimple-ner",  # noqa: E501
    ),
    out_ne_score: Output = Output(
        f"<token>:{PROJECT_NAME}.bert_base_swedish_lowermix_reallysimple_ner_ne_score",
        cls="named_entity",
        description="Named entity segment types from KBLab/bert-base-lowermix-swedish-lowermix-reallysimple-ner",  # noqa: E501
    ),
    word: Annotation = Annotation("<token:word>"),
    sentence: Annotation = Annotation("<sentence>"),
    pipeline: str = Config(f"{PROJECT_NAME}.pipeline", default="huggingface_ner"),
    model_preloaded: t.Any | None = None,
) -> None:
    """Annotate a sentence with Named Entities."""
    logger.info("huggingface_ner_pipeline")

    if model_preloaded is not None:
        ner_pipeline: NerPipeline = t.cast(NerPipeline, model_preloaded)
    else:
        logger.info(
            "loading ner pipeline(pipeline=%s)",
            pipeline,
        )
        ner_pipeline: NerPipeline = ner_pipeline_preloader(pipeline)

    ner_pipeline.run(sentence, word, out_ne_type, out_ne_score)


# @annotator("Named entity tagging with KB-BERT-NER", language=["swe"])
# def huggingface_ner_custom(
#     out_ne_type: Output = Output(
#         "<token>:huggingface_ner_custom.ne_type",
#         cls="named_entity",
#         description="Named entity segment types from KB-BERT-NER",
#     ),
#     out_ne_score: Output = Output(
#         "<token>:huggingface_ner_custom.ne_score",
#         cls="named_entity",
#         description="Named entity segment types from KB-BERT-NER",
#     ),
#     word: Annotation = Annotation("<token:word>"),
#     sentence: Annotation = Annotation("<sentence>"),
# ):
#     logger.info("huggingface_ner_custom")
#     logger.debug("word: %s", word)
