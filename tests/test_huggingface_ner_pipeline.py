import pytest
from syrupy.assertion import SnapshotAssertion

from sbx_ner_kblab.huggingface_ner_pipeline import (
    HuggingFaceNerPipeline,
    _align_tags_and_tokens,  # noqa: PLC2701
    _run_nlp_on_sentence,  # noqa: PLC2701
)

SENTENCES = {
    "ikea": "Ikea ( namnet är bildat av initialerna för Ingvar Kamprad Elmtaryd Agunnaryd ) är ett multinationellt möbelföretag som grundades 1943 av Ingvar Kamprad .",
    "verksamhetsåret": "Under verksamhetsåret 2012 omsatte Ikeakoncernen 241 miljarder kronor .",
    "kammerling": "– Jag tycker att Sveriges landslag ska vara nöjda , säger experten Anna-Karin Kammerling till Sveriges Radio .",  # noqa: RUF001
    "gruppen": "Gruppen föreslår ändringar i reglerna för 2,7 miljarder kronor .",
    "belarusen": "Poliser grep sedan två personer på flygplanet , belarusen Roman Protasevitj och hans flickvän Sofia Sapega .",
    "det-börjar": "Den börjar klockan 19.15 .",
    "encrochat": "Det programmet heter Encrochat .",
    "artisterna": "Det är till exempel artisterna Tusse , Dotter och Kikki Danielsson .",
    "street-race": "Filmerna handlar bland annat om olagliga tävlingar med bilar som kallas street-race .",
    "kvinnolobby": "Det säger Clara Berglund som är ledare för gruppen Sveriges kvinnolobby .",
    "internet": "Det går också att kolla klockan på internet på sajten 90510.se .",
}


@pytest.mark.parametrize(
    "sentence_name",
    [
        "belarusen",
        "det-börjar",
        "artisterna",
        "street-race",
        "kvinnolobby",
        "internet",
    ],
)
def test_run__align_tags_and_token_nlp_on_sentence(
    sentence_name: str,
    snapshot: SnapshotAssertion,
) -> None:
    sentence = SENTENCES[sentence_name]
    token_word = sentence.split(" ")
    nlp = HuggingFaceNerPipeline()
    actual_list = list(
        _align_tags_and_tokens(
            _run_nlp_on_sentence(nlp.model_pipeline, sentence),
            token_word,
            score_format="{:3f}",
        )
    )

    # tags = [(t[0], t[1]) for t in result]
    # assert tags == expected_tags
    assert actual_list == snapshot
    # matcher = syrupy_matchers.path_type({"score": (np.float32,)})
    # for actual in actual_list:
    #     assert actual == snapshot(matcher=matcher)
