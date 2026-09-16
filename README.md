# Sneaker Resale Price Estimator

A machine learning project to estimate resale prices for sneakers based on historical data.

## Setup

```bash
make setup
```

## Usage

Run the full pipeline:
```bash
make all
```

Run individual targets:
```bash
make data       # Download, simulate, and clean data
make train      # Train the model
make catalog    # Generate model catalog
make test       # Run tests
make api        # Start the API server
make web        # Start the web frontend
```

## Project Structure

- `backend/` - Python backend with ML models and API
- `frontend/` - Web frontend (TBD)
- `Makefile` - Build and automation targets
