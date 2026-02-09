import pytest
from syrupy.assertion import SnapshotAssertion

from sbx_ner_kblab.huggingface_ner_pipeline import (
    HuggingFaceNerPipeline,
    # interleave_tags_and_sentence,
    # interleave_tags_and_tokens,
    _run_nlp_on_sentence,  # noqa: PLC2701
    # run_nlp_on_tokens,
    # Token,
    find_word_ending,
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
    "sentence_name, token_end, expected",
    [("verksamhetsåret", 21, 21), ("verksamhetsåret", 39, 48), ("encrochat", 28, 30)],
)
def test_find_word_ending(sentence_name: str, token_end: int, expected: int) -> None:
    sentence = SENTENCES[sentence_name]

    assert find_word_ending(sentence, token_end) == expected
    assert sentence[find_word_ending(sentence, token_end)] == " "


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
def test_run_nlp_on_sentence(
    sentence_name: str,
    snapshot: SnapshotAssertion,
) -> None:
    sentence = SENTENCES[sentence_name]
    # token_word = sentence.split(" ")
    nlp = HuggingFaceNerPipeline()
    # sent = list(range(len(token_word)))
    actual = _run_nlp_on_sentence(nlp.model_pipeline, sentence)
    # result = interleave_tags_and_tokens(
    #     run_nlp_on_sentence(sentence),
    #     token_word,
    #     sent,  # sentence
    # )
    # tags = [(t[0], t[1]) for t in result]
    # assert tags == expected_tags
    assert actual == snapshot
