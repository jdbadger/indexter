"""Core engine and its base class."""


class Base:
    def save(self):
        return True


class Engine(Base):
    def run(self):
        return self.save()


def parse():
    return "core"


def make_counter():
    def increment():
        return 1

    return increment()
