.PHONY: help clean dev docs package test

help:
	@echo "The following make targets are available:"
	@echo "	 devenv		create venv and install all deps for dev env (assumes python3 cmd exists)"
	@echo "	 dev 		install all deps for dev env (assumes venv is present)"
	@echo "  docs		create pydocs for all relveant modules (assumes venv is present)"
	@echo "	 package	package for pypi"
	@echo "	 test		run all tests with coverage (assumes venv is present)"

devenv:
	pip3 install -r requirements.txt
	pre-commit install

dev:
	pip3 install -r requirements.txt

docs:
	rm -rf docs/api
	rm -rf docs/build
	sphinx-apidoc --no-toc -f -t=docs/_templates -o docs/api tune/
	sphinx-apidoc --no-toc -f -t=docs/_templates -o docs/api tune_hyperopt/
	sphinx-apidoc --no-toc -f -t=docs/_templates -o docs/api tune_optuna/
	sphinx-apidoc --no-toc -f -t=docs/_templates -o docs/api tune_sklearn/
	sphinx-apidoc --no-toc -f -t=docs/_templates -o docs/api tune_tensorflow/
	sphinx-apidoc --no-toc -f -t=docs/_templates -o docs/api tune_notebook/
	sphinx-apidoc --no-toc -f -t=docs/_templates -o docs/api tune_test/
	sphinx-build -b html docs/ docs/build/

lint:
	pre-commit run --all-files

package:
	rm -rf dist/*
	python3 setup.py sdist
	python3 setup.py bdist_wheel

lab:
	mkdir -p tmp
	pip install .
	jupyter lab --port=8888 --ip=0.0.0.0 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password='' --NotebookApp.allow_origin='*'

test:
	python3 -bb -m pytest tests

testcore:
	python3 -bb -m pytest tests/tune tests/tune_notebook tests/tune_hyperopt tests/tune_optuna
