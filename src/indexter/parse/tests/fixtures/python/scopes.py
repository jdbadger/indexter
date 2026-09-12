"""Nested-function and duplicate-name collision cases for node identity."""


def outer_a():
    def inner():
        return "a"

    return inner


def outer_b():
    def inner():
        return "b"

    return inner


def login():
    return "first"


def login():
    # Deliberate duplicate name at file scope -- exercises the ~N suffix rule.
    return "second"
