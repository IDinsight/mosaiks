# Load project config
include ./project_config.cfg
export

SHELL = /bin/bash

# directories:
# 	mkdir -p logs data config src/nodes src/pipelines playground docs reference tests
# 	touch config/parameters.yml

setup-env:
	conda create --name $(PROJECT_CONDA_ENV) python==3.9 -y
	$(CONDA_ACTIVATE) $(PROJECT_CONDA_ENV); pip install --upgrade pip
	$(CONDA_ACTIVATE) $(PROJECT_CONDA_ENV); pip install -r requirements.txt --ignore-installed
	$(CONDA_ACTIVATE) $(PROJECT_CONDA_ENV); pre-commit install

