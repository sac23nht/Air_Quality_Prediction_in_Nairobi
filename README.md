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


Data Wrangling
This project uses air quality data from Nairobi, which is stored in a MongoDB database. The data is retrieved using pymongo and then cleaned and processed for time series forecasting.

Steps:
Connect to MongoDB: Establish a connection using the pymongo library.
Data Cleaning: Remove outliers and resample the data to hourly intervals.
Feature Engineering: Create lag features (previous hour's readings) to be used in the forecasting models.
Timezone Handling: Localize the timestamps to the "Africa/Nairobi" timezone.


Here is the complete README.md file that includes the data wrangling, modeling, and evaluation steps. It is formatted in Markdown for use in GitHub.

markdown
Copy code
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
Install the required libraries:
bash
Copy code
pip install -r requirements.txt
You can also manually install the necessary dependencies:

bash
Copy code
pip install pandas matplotlib seaborn plotly statsmodels scikit-learn pymongo
Data Wrangling
This project uses air quality data from Nairobi, which is stored in a MongoDB database. The data is retrieved using pymongo and then cleaned and processed for time series forecasting.

Steps:
Connect to MongoDB: Establish a connection using the pymongo library.
Data Cleaning: Remove outliers and resample the data to hourly intervals.
Feature Engineering: Create lag features (previous hour's readings) to be used in the forecasting models.
Timezone Handling: Localize the timestamps to the "Africa/Nairobi" timezone.
The following function wrangle() processes the raw data for analysis:
def wrangle(collection):
    results = collection.find({
        "sensor": "pm2_5",  # Filter for PM2.5 data
        "value": {"$gt": 0},  # Remove non-positive readings
    })

    data = pd.DataFrame(list(results))
    
    # Convert 'timestamp' field to datetime format
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    # Set the timestamp as the index and localize to Africa/Nairobi timezone
    data.set_index('timestamp', inplace=True)
    data.index = data.index.tz_localize('UTC').tz_convert('Africa/Nairobi')
    
    # Remove outliers (values that are extremely high or low)
    data = data[data['value'] < data['value'].quantile(0.95)]
    
    # Resample data to hourly intervals (e.g., using mean)
    data = data.resample('H').mean()
    
    # Create lag features
    data['lag_1'] = data['value'].shift(1)
    
    return data


