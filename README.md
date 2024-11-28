# ARIMA Model for Time Series Forecasting: Air Quality Data

## Overview

This project implements an ARIMA (AutoRegressive Integrated Moving Average) model for forecasting air quality (PM2.5) data from Nairobi. The data is retrieved from a MongoDB instance, cleaned, and resampled for time series analysis. The analysis includes generating plots for autocorrelation (ACF) and partial autocorrelation (PACF), splitting data into training and test sets, and evaluating model performance with mean absolute error (MAE) and walk-forward validation.

## Tasks

### 1. Data Wrangling:
- Connect to MongoDB using `pymongo` and retrieve air quality data.
- Clean the data by filtering, removing outliers, and resampling the data with a configurable resampling interval.

### 2. Data Exploration:
- Generate and visualize ACF and PACF plots to assess the autocorrelation and partial autocorrelation in the data.

### 3. Data Splitting:
- Split the data into a training set (`y_train` from October 2018) and a test set (`y_test` from November 1st, 2018).

### 4. ARIMA Model:
- Train an ARIMA model with different hyperparameter combinations (p, q).
- Calculate the baseline mean absolute error (MAE) as a performance metric.
- Use the best hyperparameters to create a model with walk-forward validation and evaluate its performance on the test set.

### 5. Results Communication:
- Plot the predicted and actual values on the test set using `plotly` for visualization.

## Tools and Libraries Used:
- Python 3.x
- Pandas
- Matplotlib, Seaborn, Plotly
- Statsmodels (for ARIMA)
- scikit-learn (for metrics)
- PyMongo (for MongoDB interaction)

## Setup Instructions

To run the code in this repository, you need to have the following dependencies installed:

1. Install Python dependencies:

```bash
pip install pandas matplotlib seaborn plotly statsmodels scikit-learn pymongo
