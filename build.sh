#
set -e

#
black src/etexood
black src/test

#
mypy

#
pytest
