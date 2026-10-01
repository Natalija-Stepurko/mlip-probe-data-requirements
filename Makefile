PYTHON ?= python
OUT ?= work/results

.PHONY: help install test page check-page reproduce
.DEFAULT_GOAL := help

help:
	@echo "make install      install the locked environment and this package (editable, with tests)"
	@echo "make test         run the test suite (rebuilds the committed results and the page)"
	@echo "make page         rebuild docs/index.html from results/"
	@echo "make check-page   run the built page in jsdom and fail on errors (needs node)"
	@echo "make reproduce DATASET=DIR [UMA=DIR] [OUT=DIR]   re-measure everything from the dataset release"

install:
	$(PYTHON) -m pip install -r requirements-lock.txt
	$(PYTHON) -m pip install -e ".[test]"

test:
	$(PYTHON) -m pytest

page:
	$(PYTHON) -m mlip_probe page --results results --template docs/template.html --output docs/index.html

check-page:
	[ -d node_modules/jsdom ] || npm install --no-save jsdom@29.1.1
	node scripts/check_page.js docs/index.html

reproduce:
	@test -n "$(DATASET)" || { echo "usage: make reproduce DATASET=DIR [UMA=DIR] [OUT=DIR]"; exit 2; }
	PYTHON="$(PYTHON)" scripts/reproduce.sh "$(DATASET)" "$(OUT)" $(if $(UMA),"$(UMA)")
