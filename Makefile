conda-create:
	conda env create -f environment.yaml

deps:
	poetry install