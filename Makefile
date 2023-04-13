# Load project config
include ./project_config.cfg
export

SHELL = /bin/bash

# Note that the extra activate is needed to ensure that the activate floats env to the front of PATH
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

directories:
	mkdir -p logs data config src/nodes src/pipelines playground docs reference tests
	touch config/parameters.yml

data-directories:
	mkdir -p logs data/00_raw/SHRUG data/01_preprocessed/mosaiks_request_points \
		data/02_modelinput data/03_intermediate data/04_modeloutput

setup-env:
	conda create --name $(PROJECT_CONDA_ENV) python==3.10 -y
	$(CONDA_ACTIVATE) $(PROJECT_CONDA_ENV); pip install --upgrade pip
	$(CONDA_ACTIVATE) $(PROJECT_CONDA_ENV); pip install -r requirements.txt --ignore-installed
	$(CONDA_ACTIVATE) $(PROJECT_CONDA_ENV); pre-commit install