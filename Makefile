.PHONY: setup_env remove_env data features train predict run clean test
PROJECT_NAME=work-at-gojek

ifeq (,$(shell which pyenv))
	HAS_PYENV=False
	CONDA_ROOT=$(shell conda info --root)
else
	HAS_PYENV=True
	CONDA_VERSION=$(shell echo $(shell pyenv version | awk '{print $$1;}') | awk -F "/" '{print $$1}')
endif

setup_env:
ifeq (True,$(HAS_PYENV))
	@echo ">>> Detected pyenv, setting pyenv version to ${CONDA_VERSION}"
	pyenv local ${CONDA_VERSION}
	conda env create --name $(PROJECT_NAME) -f environment.yaml --force
	pyenv local ${CONDA_VERSION}/envs/${PROJECT_NAME}
else
	@echo ">>> Creating conda environment."
	conda env create --name $(PROJECT_NAME) -f environment.yaml --force
	@echo ">>> Activating new conda environment"
	source $(CONDA_ROOT)/bin/activate $(PROJECT_NAME)
endif

remove_env:
ifeq (True,$(HAS_PYENV))
	@echo ">>> Detected pyenv, removing pyenv version."
	pyenv local ${CONDA_VERSION} && rm -rf ~/.pyenv/versions/${CONDA_VERSION}/envs/$(PROJECT_NAME)
else
	@echo ">>> Removing conda environment"
	conda remove -n $(PROJECT_NAME) --all
endif

data:
	@echo "Creating dataset from booking_log and participant_log.."
	python -m src.data.make_dataset  # Use python from activated environment

features:
	@echo "Running feature engineering on dataset.."
	python -m src.features.build_features  # Use python from activated environment

train:
	@echo "Training classification model for allocation task.."
	python -m src.models.train_model  # Use python from activated environment

predict:
	@echo "Performing model inference to identify best drivers.."
	python -m src.models.predict_model  # Use python from activated environment

test:
	@echo "Running all unit tests.."
	nosetests --nologcapture  # Use nosetests from activated environment

run: clean data features train predict test

clean:
	@find . -name "*.pyc" -exec rm {} \;
	@rm -f data/processed/* models/* submission/*;
