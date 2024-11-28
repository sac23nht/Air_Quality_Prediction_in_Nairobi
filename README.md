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

##Install the required Python libraries:
bash
pip install -r requirements.txt

##Alternatively, you can manually install the dependencies with pip:

bash
##Copy code
pip install pandas matplotlib seaborn plotly statsmodels scikit-learn pymongo
##Data Wrangling
The data is extracted from a MongoDB database. The following steps were performed on the raw data:

