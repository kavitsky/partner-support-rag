PY ?= .venv/bin/python

.PHONY: all install fetch ingest index eval-retrieval eval check grade calibrate test

# One command from a fresh clone: download docs, ingest, compare retrieval, run answer evals.
all: install fetch ingest eval-retrieval eval

install:
	test -d .venv || python3.12 -m venv .venv
	$(PY) -m pip install -q -r requirements.txt

fetch:
	$(PY) main.py fetch

ingest:
	$(PY) main.py ingest

index:
	$(PY) main.py index

eval-retrieval:
	$(PY) main.py eval-retrieval

eval:
	$(PY) main.py eval

# What CI runs: production retrieval + answer evals, exit 1 below thresholds.
check:
	$(PY) main.py eval-retrieval --check
	$(PY) main.py eval

grade:
	$(PY) main.py grade

calibrate:
	$(PY) main.py calibrate

test:
	$(PY) -m unittest discover -s tests -t . -v
