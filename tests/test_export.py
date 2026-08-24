# -*- coding: utf-8 -*-
"""Tests for kilojoule.export's notebook-repair helpers used by both
export_html() and export_pdf() before handing a notebook to nbconvert."""

import base64

from kilojoule.export import (
    _decoded_len,
    _looks_like_valid_asset,
    resolve_cell_attachments,
    sanitize_notebook_outputs,
)

_PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20
_PNG_B64 = base64.b64encode(_PNG_BYTES).decode("ascii")
_BOGUS_B64 = base64.b64encode(b"not a real png").decode("ascii")


def _nb_with_output(output):
    return {"cells": [{"cell_type": "code", "outputs": [output]}]}


def test_sanitize_discards_stray_key_with_no_nested_counterpart():
    """This function's whole target is a mimetype duplicated between `data`
    and a stray top-level key -- a stray key with no matching entry already
    under `data` has nothing to disambiguate against, so it is discarded
    rather than promoted into `data` unchecked."""
    nb = _nb_with_output(
        {
            "output_type": "display_data",
            "data": {"text/plain": "x"},
            "metadata": {},
            "image/png": _PNG_B64,
        }
    )
    n_fixed = sanitize_notebook_outputs(nb)
    output = nb["cells"][0]["outputs"][0]
    assert n_fixed == 1
    assert "image/png" not in output  # stray key gone
    assert "image/png" not in output["data"]  # and not promoted either
    assert set(output) == {"output_type", "data", "metadata"}


def test_sanitize_prefers_genuine_asset_over_corrupted_placeholder():
    """When the nested `data` copy and the stray top-level copy disagree,
    the one that actually decodes to a real PNG wins, regardless of which
    slot it started in."""
    nb = _nb_with_output(
        {
            "output_type": "display_data",
            "data": {"image/png": "deadbeef"},  # corrupted placeholder
            "metadata": {},
            "image/png": _PNG_B64,  # genuine payload, misplaced
        }
    )
    sanitize_notebook_outputs(nb)
    assert nb["cells"][0]["outputs"][0]["data"]["image/png"] == _PNG_B64


def test_sanitize_keeps_nested_copy_when_it_is_already_the_genuine_one():
    nb = _nb_with_output(
        {
            "output_type": "display_data",
            "data": {"image/png": _PNG_B64},
            "metadata": {},
            "image/png": "deadbeef",  # the corrupted one this time
        }
    )
    sanitize_notebook_outputs(nb)
    assert nb["cells"][0]["outputs"][0]["data"]["image/png"] == _PNG_B64


def test_sanitize_ignores_valid_outputs():
    nb = _nb_with_output(
        {"output_type": "display_data", "data": {"text/plain": "x"}, "metadata": {}}
    )
    assert sanitize_notebook_outputs(nb) == 0


def test_sanitize_ignores_unknown_output_type():
    """An output_type not in the schema map (e.g. a future nbformat
    addition) is left alone rather than guessed at."""
    nb = _nb_with_output({"output_type": "some_future_type", "extra": 1})
    assert sanitize_notebook_outputs(nb) == 0
    assert nb["cells"][0]["outputs"][0]["extra"] == 1


def test_looks_like_valid_asset_checks_png_magic_bytes():
    assert _looks_like_valid_asset("image/png", _PNG_B64) is True
    assert _looks_like_valid_asset("image/png", _BOGUS_B64) is False


def test_looks_like_valid_asset_trusts_unverifiable_mimetypes():
    # Valid base64, but for a mimetype with no known magic-byte signature --
    # nothing to check it against, so it's assumed plausible.
    b64 = base64.b64encode(b"hello world").decode("ascii")
    assert _looks_like_valid_asset("text/plain", b64) is True


def test_looks_like_valid_asset_rejects_non_base64_string():
    assert _looks_like_valid_asset("image/png", "not base64 at all!!") is False


def test_decoded_len_prefers_decoded_byte_length():
    assert _decoded_len(_PNG_B64) == len(_PNG_BYTES)
    # non-base64 text falls back to raw string length
    assert _decoded_len("not base64 at all!!") == len("not base64 at all!!")


def test_resolve_cell_attachments_rewrites_img_src_to_data_uri():
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "source": '<img src="attachment:pic.png" style="max-width:100%">',
                "attachments": {"pic.png": {"image/png": _PNG_B64}},
            }
        ]
    }
    n_fixed = resolve_cell_attachments(nb)
    assert n_fixed == 1
    new_source = nb["cells"][0]["source"]
    assert f"data:image/png;base64,{_PNG_B64}" in new_source
    # trailing attributes on the tag are preserved
    assert 'style="max-width:100%"' in new_source


def test_resolve_cell_attachments_leaves_unmatched_reference_alone():
    original = '<img src="attachment:missing.png">'
    nb = {
        "cells": [
            {"cell_type": "markdown", "source": original, "attachments": {}}
        ]
    }
    assert resolve_cell_attachments(nb) == 0
    assert nb["cells"][0]["source"] == original


def test_resolve_cell_attachments_skips_cells_without_attachments():
    nb = {"cells": [{"cell_type": "code", "source": "x = 1"}]}
    assert resolve_cell_attachments(nb) == 0
