"""Unit tests for CJK-aware ASR metrics (CER, normalize_cjk_text).

CER is the standard metric for Japanese / Chinese / Korean ASR evaluation
because word boundaries are not orthographically marked. WER on the same
languages requires a tokenizer (MeCab for ja, jieba for zh) and is left as
future work — see CLAUDE.md.
"""

from __future__ import annotations

from whisperlivekit.metrics import compute_cer, normalize_cjk_text

# ---------------------------------------------------------------------------
# normalize_cjk_text
# ---------------------------------------------------------------------------


def test_normalize_strips_whitespace():
    assert normalize_cjk_text("今 日 は") == "今日は"
    assert normalize_cjk_text("今\t日\nは") == "今日は"


def test_normalize_strips_cjk_punctuation():
    assert normalize_cjk_text("今日は、良い天気。") == "今日は良い天気"
    assert normalize_cjk_text("你好，世界！") == "你好世界"


def test_normalize_strips_ascii_punctuation():
    assert normalize_cjk_text("hello, world!") == "helloworld"


def test_normalize_lowercases_ascii_only():
    # CJK characters have no case, ASCII does.
    assert normalize_cjk_text("Hello今日") == "hello今日"


def test_normalize_nfc_composes_decomposed_forms():
    # NFD ka + dakuten -> NFC ga (two codepoints -> one)
    decomposed = "が"  # か + ◌゙
    composed = "が"
    assert normalize_cjk_text(decomposed) == composed


# ---------------------------------------------------------------------------
# compute_cer — Japanese
# ---------------------------------------------------------------------------


def test_cer_perfect_match_ja():
    r = compute_cer("今日は良い天気", "今日は良い天気")
    assert r["cer"] == 0.0
    assert r["substitutions"] == 0
    assert r["insertions"] == 0
    assert r["deletions"] == 0
    assert r["ref_chars"] == 7
    assert r["hyp_chars"] == 7


def test_cer_single_substitution_ja():
    # 7 chars, 1 substitution → 1/7
    r = compute_cer("今日は良い天気", "今日は悪い天気")
    assert r["cer"] == 1 / 7
    assert r["substitutions"] == 1
    assert r["insertions"] == 0
    assert r["deletions"] == 0


def test_cer_insertion_ja():
    # ref 7 chars, hyp 8 chars (1 insertion) → 1/7
    r = compute_cer("今日は良い天気", "今日はとても良い天気")
    assert r["cer"] == 3 / 7
    assert r["insertions"] == 3
    assert r["substitutions"] == 0
    assert r["deletions"] == 0


def test_cer_deletion_ja():
    r = compute_cer("今日は良い天気", "今日は天気")
    assert r["cer"] == 2 / 7
    assert r["deletions"] == 2


def test_cer_ignores_whitespace_and_punctuation_ja():
    # Reference written with spaces and punctuation, hypothesis without — should match.
    r = compute_cer("今日は、良い天気。", "今日は良い天気")
    assert r["cer"] == 0.0


# ---------------------------------------------------------------------------
# compute_cer — Chinese
# ---------------------------------------------------------------------------


def test_cer_perfect_match_zh():
    r = compute_cer("今天天气真好", "今天天气真好")
    assert r["cer"] == 0.0
    assert r["ref_chars"] == 6


def test_cer_substitution_zh():
    # 真 → 很 : 1 substitution / 6 chars
    r = compute_cer("今天天气真好", "今天天气很好")
    assert r["cer"] == 1 / 6
    assert r["substitutions"] == 1


def test_cer_handles_chinese_punctuation():
    r = compute_cer("你好，世界！", "你好世界")
    assert r["cer"] == 0.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_cer_empty_reference_empty_hypothesis():
    r = compute_cer("", "")
    assert r["cer"] == 0.0
    assert r["ref_chars"] == 0
    assert r["hyp_chars"] == 0


def test_cer_empty_reference_nonempty_hypothesis():
    r = compute_cer("", "今日")
    # 2 insertions, no reference → CER reported as raw count
    assert r["cer"] == 2.0
    assert r["insertions"] == 2


def test_cer_nonempty_reference_empty_hypothesis():
    r = compute_cer("今日は", "")
    # 3 deletions / 3 ref chars
    assert r["cer"] == 1.0
    assert r["deletions"] == 3


def test_cer_can_exceed_one():
    # Hypothesis is much longer and entirely different.
    r = compute_cer("今日", "全然違う長い文章です")
    assert r["cer"] > 1.0


def test_cer_mixed_japanese_english():
    # Common in real transcripts: katakana foreign word + English brand.
    r = compute_cer("カメラはCanonです", "カメラはcanonです")
    # After normalisation Canon → canon, full match
    assert r["cer"] == 0.0


def test_cer_substitution_count_matches_distance():
    # Sanity: subs + ins + dels equal the edit distance.
    r = compute_cer("今日は良い天気です", "今天好天気だね")
    total_ops = r["substitutions"] + r["insertions"] + r["deletions"]
    assert total_ops == int(round(r["cer"] * r["ref_chars"]))
