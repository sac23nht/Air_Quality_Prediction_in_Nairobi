# Air Quality Forecasting Using Linear Regression and ARIMA Models

## Overview
This project demonstrates the use of both Linear Regression and ARIMA (AutoRegressive Integrated Moving Average) models for forecasting air quality data, specifically PM2.5 (Particulate Matter) levels in Nairobi. The project includes data extraction, cleaning, time series analysis, model training, and evaluation using multiple forecasting techniques.

### Key Features:
- Data extraction from a MongoDB database containing air quality data for Nairobi.
- Data cleaning, including handling missing values and removing outliers.
- Exploration of time series data using visualizations like autocorrelation (ACF) and partial autocorrelation (PACF).
- Linear Regression model to predict future PM2.5 levels based on historical data.
- ARIMA model for time series forecasting and evaluation.
- Evaluation of models using metrics like Mean Absolute Error (MAE) and Walk-forward validation.
  
## Table of Contents
1. [Project Setup](#project-setup)
2. [Data Wrangling](#data-wrangling)
3. [Modeling](#modeling)
   - [Linear Regression](#linear-regression)
   - [ARIMA](#arima)
4. [Model Evaluation](#model-evaluation)
5. [Results](#results)
6. [Tools and Libraries](#tools-and-libraries)
7. [License](#license)

## Project Setup

To run this project, you'll need to have Python 3.x installed along with the required libraries. Follow the instructions below to set up the environment.

### Prerequisites

- Python 3.x
- MongoDB (if you're connecting to a local MongoDB instance for the data)

### Install Dependencies

1. Clone this repository:

```bash
git clone <repository-url>
----------------------------------------------------------------------------------------------------------------------------------

Install the required Python libraries:
bash
Copy code
pip install -r requirements.txt
Alternatively, you can manually install the dependencies with pip:

bash
Copy code
pip install pandas matplotlib seaborn plotly statsmodels scikit-learn pymongo
Data Wrangling
The data is extracted from a MongoDB database. The following steps were performed on the raw data:

MongoDB Connection:
 A connection was made to a local MongoDB instance using the pymongo library.
Data Extraction: The data was retrieved from the "air-quality" database, specifically the "nairobi" collection.
Data Cleaning: The dataset was cleaned by:
Filtering the data to include only PM2.5 measurements.
Removing outliers where PM2.5 readings exceeded 500.
Resampling the data on an hourly basis and using forward filling for missing values.
Feature Engineering: A lag feature was created to incorporate previous PM2.5 readings as a predictor for future values.
Modeling
Linear Regression
Data Splitting: The cleaned data was split into training and testing sets.
Model Training: A Linear Regression model was trained on the training set using the previous hour's PM2.5 reading (P2.L1) as the predictor.
Evaluation: The model’s performance was evaluated using the Mean Absolute Error (MAE) on both the training and test sets.
python
Copy code
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Instantiate and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate MAE
mae = mean_absolute_error(y_test, y_pred)
