from __future__ import annotations

from src.query_normalization import expand_query, normalize_text


def test_normalize_text_casing_punctuation_whitespace():
    assert normalize_text("  WhAt   IS   GMP??  ") == "what is gmp?"
    assert normalize_text("Data   integrity!!!   ") == "data integrity!"


def test_synonym_expansion_continuous_processing():
    q_norm = normalize_text("What is continuous processing in ICH Q13?")
    expanded = expand_query(q_norm)
    assert expanded[0] == q_norm
    assert any("continuous manufacturing" in q for q in expanded)


def test_acronym_expansion_gmp():
    q_norm = normalize_text("GMP validation requirements")
    expanded = expand_query(q_norm)
    assert expanded[0] == q_norm
    assert any("good manufacturing practice" in q for q in expanded)


def test_original_query_is_first():
    q_norm = normalize_text("  ICH Q13 continuous processing  ")
    expanded = expand_query(q_norm)
    assert expanded[0] == q_norm

