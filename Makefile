.PHONY: setup data train catalog test api web notebooks all clean

VENV := backend/.venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest
JUPYTER := $(VENV)/bin/jupyter

setup:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r backend/requirements.txt
	$(PIP) install -e backend
	cd frontend && npm install

data:
	$(PYTHON) -m sneakerml.data.download
	$(PYTHON) -m sneakerml.data.simulate
	$(PYTHON) -m sneakerml.data.clean

train:
	$(PYTHON) -m sneakerml.train

catalog:
	$(PYTHON) -m sneakerml.catalog

test:
	$(PYTEST)

api:
	$(PYTHON) -m flask --app sneakerml.api.app run --port 5000

web:
	cd frontend && npm run dev

notebooks:
	$(JUPYTER) nbconvert --execute --to notebook --inplace backend/notebooks/*.ipynb

all: setup data train catalog test notebooks
	cd frontend && npm run build

clean:
	rm -rf $(VENV) backend/build backend/dist backend/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
