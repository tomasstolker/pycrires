.PHONY: help pypi pypi-test docs coverage test clean

help:
	@echo "pypi - submit to PyPI server"
	@echo "pypi-test - submit to TestPyPI server"
	@echo "docs - generate Sphinx documentation"
	@echo "coverage - check code coverage"
	@echo "test - run unit tests"
	@echo "clean - remove artifacts"

pypi:
	python -m build
	twine upload dist/*

pypi-test:
	python -m build
	twine upload --repository testpypi dist/*

docs:
	rm -rf docs/api
	sphinx-apidoc -o docs pycrires
	cd docs/
	$(MAKE) -C docs clean
	$(MAKE) -C docs html

coverage:
	coverage run --source=pycrires -m pytest
	coverage report -m

test:
	pytest -s --cov=pycrires/ --cov-report=xml

clean:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -rf {} +
	rm -f .coverage*
	rm -f coverage.xml
	rm -rf build/
	rm -rf dist/
	rm -rf pycrires.egg-info/
	rm -rf htmlcov/
	rm -rf .tox/
	rm -rf .pytest_cache/
	rm -f docs/files.json
	rm -f docs/header.csv
	rm -f docs/header.xlsx
	rm -f docs/skycalc_temp.fits
	rm -rf docs/_build/
	rm -rf docs/api/
	rm -rf docs/.ipynb_checkpoints/
	rm -rf docs/calib/
	rm -rf docs/config/
	rm -rf docs/product/
	rm -rf docs/tmp/
	rm -rf docs/raw/
	rm -rf tests/calib/
	rm -rf tests/config/
	rm -rf tests/product/
	rm -rf tests/tmp/
	rm -rf tests/raw/
	rm -rf tests/files.json
	rm -rf tests/header.csv
	rm -rf tests/header.xlsx
