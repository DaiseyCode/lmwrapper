pip install --upgrade -e '.[hf,anthropic,docs,dev]' || exit 1
python -m pytest "$@"