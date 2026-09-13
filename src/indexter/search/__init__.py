"""Hybrid search over an indexed repository: vector and keyword candidates
fused by reciprocal rank fusion, rolled up into class-aware entries with
snippets and graph context, and expanded with one hop of related nodes --
see design.md for the full ranking pipeline (M5).
"""

from __future__ import annotations
