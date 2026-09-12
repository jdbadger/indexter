"""Top-level module exercising a package re-export, an aliased import, a
wildcard import, a shadowed builtin, and a same-name ambiguity across
modules."""

from pkg import Engine
from pkg.util import helper as h
from pkg.core import *


def len(items):
    return 0


def use_len(items):
    return len(items)


def ambiguous_call(x):
    return x.parse()


def build():
    e = Engine()
    return h() + use_len([1, 2]) + parse()
