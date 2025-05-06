# Smartwatch Health Monitoring System

This project demonstrates a machine learning monitoring system for smartwatch health data. It showcases how to implement drift detection, performance monitoring, and visualization for ML models deployed in a health application.

## Overview

The system simulates smartwatch devices sending health metrics to a central server, which uses a machine learning model to predict stress levels. It includes:

1. **Data Generation**: Scripts to create realistic health data for multiple users
2. **ML Model**: A regression model predicting stress levels from health metrics
3. **Monitoring APIs**: FastAPI endpoints for predictions and monitoring
4. **Visualization**: Evidently AI dashboards for model performance and data drift
5. **Client Simulation**: Scripts to simulate smartwatch clients sending data

## Health Metrics Explanation

The system tracks 5 key health metrics from simulated smartwatches:

- **Heart Rate Variability (feature1)**: Measures the variation in time between heartbeats. Higher values typically indicate better cardiovascular health and stress resilience.
- **Activity Level (feature2)**: Normalized representation of daily steps and physical activity. Higher values indicate more active individuals.
- **Sleep Quality (feature3)**: Represents overall sleep quality based on duration and interruptions. Higher values indicate better sleep.
- **Resting Heart Rate (feature4)**: Normalized heart rate during periods of rest. Lower values typically indicate better cardiovascular fitness.
- **Skin Temperature Variation (feature5)**: Normalized skin temperature relative to baseline. Can indicate stress, illness, or recovery state.

The model predicts a **Stress Level** score from 0-10, where higher values indicate greater stress.

## Setup Instructions

### Option 1: Local Development

#### 1. Initialize the Environment

```bash
cd ~/smartwatch_health_monitoring
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### 2. Generate Simulated Data

Generate reference and current data for 5 virtual users over a week period:

```bash
cd scripts
python generate_smartwatch_data.py
```

This creates:
- `data/reference_data.csv` - Training data for the model
- `data/current_data.csv` - Current data for predictions
- Individual user data files in the data directory

#### 3. Train the Model

Train a regression model to predict stress levels:

```bash
python train_smartwatch_model.py
```

The model will be saved to `data/model.pkl`.

#### 4. Initialize the Database

```bash
cd ..
python -m app.db.init_db
```

#### 5. Start the API Server

```bash
uvicorn main:app --reload
```

The API will be available at http://localhost:8000.

#### 6. Simulate Smartwatch Clients

In a new terminal window:

```bash
cd ~/smartwatch_health_monitoring
source venv/bin/activate
cd scripts
python simulate_smartwatch_clients.py --interval 5
```

This will simulate smartwatches sending data every 5 seconds.

### Option 2: Docker Deployment

For easier setup and deployment, you can use Docker Compose to run all components:

#### 1. Build and Start the Services

```bash
# Navigate to project directory
cd ~/smartwatch_health_monitoring

# Start all services
docker-compose up -d

# Generate data and train model (first time only)
docker-compose exec app bash -c "cd scripts && python generate_smartwatch_data.py && python train_smartwatch_model.py"

# Initialize the database
docker-compose exec app python -m app.db.init_db
```

This will start:
- FastAPI application on http://localhost:8000
- PostgreSQL database
- Prefect server on http://localhost:4200
- Client simulator service

#### 2. Stop the Services

```bash
docker-compose down
```

#### 3. View Logs

```bash
docker-compose logs -f app
```

#### 4. Troubleshooting

Sometimes we may see the monitoring pipeline alone did not work as expected.
Try running it as
```bash
docker exec smartwatch-monitoring-app python -m app.workflows.monitoring_pipeline
```

If there are changes carried out in app/workflow/tasks/monitoring_pipeline.py, we need to rebuild the docker image

```bash
docker-compose down --rmi all
docker-compose build --no-cache
docker-compose exec app bash -c "cd scripts && python generate_smartwatch_data.py && python train_smartwatch_model.py"
docker-compose exec app python -m app.db.init_db

docker exec smartwatch-monitoring-app python -c "from app.workflows.tasks.monitoring_tasks import get_reference_data, get_current_data, generate_model_performance_metrics; import pandas as pd; ref_data = get_reference_data(); cur_data = get_current_data(); common_cols = list(set(ref_data.columns).intersection(set(cur_data.columns))); ref_data = ref_data[common_cols]; cur_data = cur_data[common_cols]; generate_model_performance_metrics(ref_data, cur_data)"
```


## Exploring the Monitoring Features

1. **API Documentation**: Visit http://localhost:8000/docs to see and test the API endpoints

2. **Monitoring Dashboards**:
   - Model Performance: http://localhost:8000/api/v1/monitoring/monitor-model
   - Data Drift: http://localhost:8000/api/v1/monitoring/monitor-target

3. **Metrics History**: http://localhost:8000/api/v1/monitoring/metrics

4. **Prefect Dashboard**: Visit http://localhost:4200 to view workflow runs (Docker setup only)

## Understanding the Drift Detection

The system uses Evidently AI to detect and visualize two types of drift:

1. **Data Drift**: Detects when the distribution of input features changes compared to the reference data. For example, if users' heart rate patterns change significantly over time.

2. **Target Drift**: Detects when the relationship between features and the target variable changes. For example, if high heart rate variability no longer correlates with lower stress levels.

The visualizations help identify which features are drifting and how severely.

## Simulating Different Scenarios

To simulate specific scenarios, you can modify the user profiles in `scripts/generate_smartwatch_data.py`:

1. **Seasonal Changes**: Modify the base values of activity and temperature to reflect seasonal patterns
2. **Health Events**: Create specific periods with unusual patterns in heart rate and sleep
3. **User Behavior Changes**: Adjust the weekend/weekday modifiers to reflect changing work patterns

## Project Structure

- `app/` - FastAPI application
  - `api/` - API routes and endpoints
  - `core/` - Core configurations
  - `db/` - Database models and session management
  - `schemas/` - Pydantic schemas for validation
  - `services/` - Business logic
  - `workflows/` - Monitoring pipelines
- `data/` - Reference and current data
- `scripts/` - Utility scripts for simulation
- `docs/` - Additional documentation
