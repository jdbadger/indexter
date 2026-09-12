"""Sample module exercising imports, a documented class, decorators, and calls."""

import os
from collections import OrderedDict
from . import sibling
from ..pkg import thing

MAX_RETRIES = 3


class Base:
    """A base class with nothing interesting in it."""


class Handler(Base):
    """Handles requests, delegating validation to a helper."""

    def login(self, user):
        """Authenticate a user and record the attempt."""
        self.validate(user)
        os.path.join("var", "log")
        return thing(user)

    def validate(self, user):
        return user is not None


def deprecated(fn):
    """A fixture decorator -- not real, just wraps its argument."""
    return fn


@deprecated
def standalone():
    """A decorated module-level function."""
    return MAX_RETRIES
