# Load project config
include ./project_config.cfg
export

SHELL = /bin/bash

# Note that the extra activate is needed to ensure that the activate floats env to the front of PATH
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

directories:
	mkdir -p logs data config src/nodes src/pipelines playground docs reference tests
	touch config/parameters.yml

setup-env:
	conda create --name $(PROJECT_CONDA_ENV) python==3.9 -y
	$(CONDA_ACTIVATE) $(PROJECT_CONDA_ENV); pip install --upgrade pip
	$(CONDA_ACTIVATE) $(PROJECT_CONDA_ENV); pip install -r requirements.txt --ignore-installed
	$(CONDA_ACTIVATE) $(PROJECT_CONDA_ENV); pre-commit install

preprocess-shrug-rural-keys:
	$(CONDA_ACTIVATE) $(PROJECT_CONDA_ENV); python src/00_preprocess_shrug_rural_keys.py

preprocess-create-mosaiks-points:
	$(CONDA_ACTIVATE) $(PROJECT_CONDA_ENV); python src/01_preprocess_create_mosaiks_points.py