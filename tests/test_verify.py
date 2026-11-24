"""Quote verifier: verbatim, altered quote, whitespace and quote-mark variants, bad citations."""
from __future__ import annotations

from rag.generation.verify import quote_in_text, verify_claims

TEXT = ("Where a person other than a Collector, or than a guardian appointed by will or other instrument, "
        "has been appointed or declared by the Court to be guardian of the property of a ward, he shall not, "
        "without the previous permission of the Court,—\n(a) mortgage or charge, or transfer by sale, gift, "
        "exchange or otherwise, any part of the immoveable property of his ward, or")
JTEXT = "the Judgment dated 26.07.2024 (“Impugned Judgment”) vide which, the learned Judge’s order"


def test_verbatim_quote_passes():
    assert quote_in_text("he shall not, without the previous permission of the Court", TEXT)


def test_altered_quote_fails():
    assert not quote_in_text("he shall not, without the prior permission of the Court", TEXT)
    assert not quote_in_text("he may not, without the previous permission of the Court", TEXT)


def test_whitespace_variants_pass():
    assert quote_in_text("permission of the Court,— (a) mortgage   or charge", TEXT)
    assert quote_in_text("  he shall\nnot,\twithout the previous permission  ", TEXT)


def test_quote_mark_variants_pass_both_ways():
    assert quote_in_text('dated 26.07.2024 ("Impugned Judgment")', JTEXT)
    assert quote_in_text("the learned Judge's order", JTEXT)
    assert quote_in_text("the learned Judge’s order", "the learned Judge's order")


def test_only_whitespace_and_quotes_are_normalised():
    # a hyphen instead of the em dash, or different case, is NOT verbatim
    assert not quote_in_text("permission of the Court,- (a) mortgage", TEXT)
    assert not quote_in_text("HE SHALL NOT", TEXT)
    assert not quote_in_text("", TEXT)


def test_verify_claims_reasons_and_rate():
    ctx = {"GWA-1890:29#1": TEXT}
    corpus = {"GWA-1890:29#1": {}, "GWA-1890:31#1": {}}
    claims = [
        {"text": "ok", "chunk_id": "GWA-1890:29#1", "quote": "transfer by sale, gift, exchange or otherwise"},
        {"text": "altered", "chunk_id": "GWA-1890:29#1", "quote": "transfer by sale or gift"},
        {"text": "not shown", "chunk_id": "GWA-1890:31#1", "quote": "anything"},
        {"text": "invented", "chunk_id": "XYZ:1#1", "quote": "anything"},
        {"text": "no quote", "chunk_id": "GWA-1890:29#1", "quote": None},
    ]
    out, summary = verify_claims(claims, ctx, corpus, mode="c1")
    assert [c["reason"] for c in out] == ["ok", "quote_not_found", "chunk_not_in_context", "unknown_chunk",
                                         "missing_quote"]
    assert [c["verified"] for c in out] == [True, False, False, False, False]
    assert summary["n_claims"] == 5 and summary["n_verified"] == 1 and summary["verification_rate"] == 0.2
    assert len(summary["failures"]) == 4, "failures are kept, never dropped"


def test_c0_claims_without_citations_are_not_applicable():
    out, summary = verify_claims([{"text": "s.25 GWA", "chunk_id": None, "quote": None}], {}, None, mode="c0")
    assert out[0]["verified"] is None and out[0]["reason"] == "no_citation"
    assert summary["verification_rate"] is None
