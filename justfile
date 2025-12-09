# List all the commands in this file
list:
    just -l

# Run all the tests against multiple Python versions
test:
    uv run --python 3.11 --group dev pytest
    uv run --python 3.12 --group dev pytest
    uv run --python 3.13 --group dev pytest