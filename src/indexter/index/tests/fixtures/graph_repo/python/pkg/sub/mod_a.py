"""Relative import across a subpackage boundary."""

from ..util import helper


def use_helper():
    return helper()
