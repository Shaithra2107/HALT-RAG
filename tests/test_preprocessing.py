"""Tests for preprocessing additions: NFKC cleanup, sentence-boundary cuts,
and background-page filtering."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from halt_rag.data.passages import clean_paragraphs, merge_into_passages
from halt_rag.data.wiki2023 import is_background_page_ok


# ── clean_paragraphs: NFKC normalisation + whitespace ────────────────────────

def test_clean_paragraphs_removes_non_breaking_space():
    # \xa0 is the HTML non-breaking space
    para = "Hello\xa0world. " + "x" * 50
    result = clean_paragraphs([para])
    assert len(result) == 1
    assert "\xa0" not in result[0]
    assert "Hello world" in result[0]


def test_clean_paragraphs_collapses_repeated_spaces():
    para = "word1  word2   word3 " + "y" * 50
    result = clean_paragraphs([para])
    assert len(result) == 1
    assert "  " not in result[0]


def test_clean_paragraphs_nfkc_ligature():
    # ﬁ (U+FB01) is the fi ligature; NFKC normalises it to "fi"
    para = "ﬁrst edition published in " + "a" * 50
    result = clean_paragraphs([para])
    assert len(result) == 1
    assert result[0].startswith("first")


def test_clean_paragraphs_still_drops_short():
    assert clean_paragraphs(["short"]) == []


def test_clean_paragraphs_still_keeps_long():
    para = "a" * 60
    assert clean_paragraphs([para]) == [para]


def test_clean_paragraphs_strips_leading_trailing_whitespace():
    para = "  " + "word " * 20
    result = clean_paragraphs([para])
    assert result and not result[0].startswith(" ")


# ── merge_into_passages: sentence-boundary cuts ──────────────────────────────

def _words(n):
    return " ".join(f"w{i}" for i in range(n))


def test_merge_does_not_break_mid_sentence_when_unnecessary():
    # Two short sentences that together fit in 100 words — stay in one passage
    s1 = "The cat sat on the mat."
    s2 = "The dog ran away quickly."
    result = merge_into_passages([s1 + " " + s2])
    assert len(result) == 1
    assert "cat" in result[0] and "dog" in result[0]


def test_merge_cuts_at_sentence_boundary_not_mid_word():
    # 60 + 60 = 120 words: should split at the sentence boundary, not word 100
    s1 = _words(60) + "."
    s2 = _words(60) + "."
    result = merge_into_passages([s1 + " " + s2], max_words=100)
    # Must produce 2 passages (one per sentence), not one 120-word passage
    assert len(result) == 2
    # Each passage contains only the words from its own sentence
    assert "w0" in result[0] and "w0" in result[1]  # both start with w0 (different sentences)


def test_merge_breaks_mid_sentence_only_when_single_sentence_too_long():
    # 150-word sentence: must be cut, no sentence boundary to use
    long_sent = _words(150)         # no . ! ? so it's one sentence
    result = merge_into_passages([long_sent], max_words=100)
    # Expect 2 chunks: 100 and 50
    assert len(result) == 2
    assert len(result[0].split()) == 100
    assert len(result[1].split()) == 50


def test_merge_joins_short_sentences_until_limit():
    # 10 three-word sentences → 30 words total → one passage
    sentences = ["A B C." for _ in range(10)]
    result = merge_into_passages(sentences, max_words=100)
    assert len(result) == 1


def test_merge_multiple_paragraphs_packed_together():
    # Three 30-word paragraphs; each has one sentence
    para = _words(30) + "."
    result = merge_into_passages([para, para, para], max_words=100)
    # 3×30 = 90 ≤ 100 → all in one passage
    assert len(result) == 1


# ── is_background_page_ok ────────────────────────────────────────────────────

GOOD_PARAS = ["First paragraph " + "x" * 40,
              "Second paragraph " + "y" * 40,
              "Third paragraph " + "z" * 40]


def test_bg_ok_accepts_normal_page():
    assert is_background_page_ok("History of France", GOOD_PARAS) is True


def test_bg_rejects_disambiguation_in_title():
    assert is_background_page_ok("Mercury (disambiguation)", GOOD_PARAS) is False


def test_bg_rejects_may_refer_to_in_text():
    paras = ["Mercury may refer to several things."] + GOOD_PARAS
    assert is_background_page_ok("Mercury", paras) is False


def test_bg_rejects_list_of_page():
    assert is_background_page_ok("List of countries by area", GOOD_PARAS) is False


def test_bg_rejects_too_few_paragraphs():
    assert is_background_page_ok("History of France", GOOD_PARAS[:2]) is False


def test_bg_accepts_exactly_min_paras():
    assert is_background_page_ok("History of France", GOOD_PARAS, min_paras=3) is True


def test_bg_case_insensitive_disambiguation():
    assert is_background_page_ok("Mercury (Disambiguation)", GOOD_PARAS) is False


def test_bg_case_insensitive_list_of():
    assert is_background_page_ok("List Of Things", GOOD_PARAS) is False
