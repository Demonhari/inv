.PHONY: setup ingest ask test clean

setup:
\tpip install -r requirements.txt

ingest:
\tpython ingest.py

ask:
\tpython main.py "$(q)"

test:
\tpytest -q

clean:
\trm -rf .chroma __pycache__ .pytest_cache
