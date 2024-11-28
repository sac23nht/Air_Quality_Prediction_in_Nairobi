ARIMA Model for Time Series Forecasting: Air Quality Data
Overview
This project implements an ARIMA (AutoRegressive Integrated Moving Average) model for forecasting PM2.5 air quality data from Nairobi. The data is retrieved from a MongoDB instance, cleaned, and resampled to suit time series analysis. The project includes the following tasks:

Data Wrangling: Retrieve and prepare the data for analysis.
Data Exploration: Generate and visualize ACF (Autocorrelation Function) and PACF (Partial Autocorrelation Function) plots.
Data Splitting: Split the data into training and test sets.
ARIMA Model: Build and tune the ARIMA model, evaluate performance using MAE, and apply walk-forward validation.
Results Communication: Visualize the predictions and actual values.
The primary goal is to predict the future air quality based on historical data using ARIMA and assess model performance.

Tasks Breakdown
1. Data Wrangling
Connect to MongoDB: Use pymongo to retrieve the air quality data from a MongoDB instance.
Clean Data: Filter and remove any outliers, and handle missing data appropriately.
Resample Data: Resample the data into hourly intervals or any specified frequency to make the time series more manageable and consistent.
2. Data Exploration
Autocorrelation & Partial Autocorrelation: Generate and visualize ACF and PACF plots to determine the correlation between observations at different time lags. This helps in identifying the optimal parameters for the ARIMA model (p, q).

ACF (Autocorrelation Function): Measures the correlation between the current observation and its past values.
PACF (Partial Autocorrelation Function): Helps identify the order of the autoregressive (AR) part by showing the correlation after removing the effect of intervening lags.
3. Data Splitting
Train-Test Split: Split the data into:
Training Set (y_train): Data from October 2018 onwards.
Test Set (y_test): Data starting from November 1st, 2018.
4. ARIMA Model
Model Training: Use statsmodels to fit the ARIMA model to the training data by experimenting with different combinations of hyperparameters (p, d, q). These parameters correspond to:
p: Number of lag observations included in the model (AR part).
d: Degree of differencing (I part) to make the series stationary.
q: Size of the moving average window (MA part).
Performance Evaluation: Calculate the baseline Mean Absolute Error (MAE) on the test set. Then use walk-forward validation to further assess model performance by iteratively forecasting and updating the model.
5. Results Communication
Visualization: Plot both predicted and actual PM2.5 values on the test set using Plotly for interactive and informative plots, which helps to assess the forecasting accuracy visually.
Tools and Libraries Used
Python 3.x
Pandas: For data manipulation and analysis.
Matplotlib, Seaborn, Plotly: For plotting and data visualization.
Statsmodels: For implementing ARIMA models and statistical tests.
scikit-learn: For evaluating the performance (e.g., Mean Absolute Error).
PyMongo: For MongoDB interaction and retrieving the air quality data.
Setup Instructions
To run the code in this repository, you need to have the following dependencies installed:

1. Install Python dependencies:
bash
Copy code
pip install pandas matplotlib seaborn plotly statsmodels scikit-learn pymongo
2. MongoDB Setup:
Ensure that you have access to a MongoDB instance and that the air quality data is stored in the database air-quality and collection nairobi.

Example MongoDB setup:

Database: air-quality
Collection: nairobi
You can fetch PM2.5 data by querying the appropriate fields for timestamp and air quality measurements.
3. Running the Code:
Clone the repository to your local machine and navigate to the project directory.
Open the notebook or Python script and run the code in order. Ensure your MongoDB server is running and accessible.
Example Code to Connect to MongoDB and Fetch Data:
python
Copy code
from pymongo import MongoClient
import pandas as pd
import pytz

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['air-quality']
collection = db['nairobi']

# Wrangle data: Fetch and clean the data
def wrangle_data(collection):
    results = collection.find({"metadata.site": 29, "metadata.measurement": "P2"}, 
                               projection={"P2": 1, "timestamp": 1, "_id": 0})
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Set index to timestamp and localize time zone
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df.index = df.index.tz_localize("UTC").tz_convert("Africa/Nairobi")
    
    # Filter out outliers and resample data
    df = df[df['P2'] < 500]
    df = df['P2'].resample('H').mean().fillna(method='ffill')
    
    return df

# Fetch and clean data
df = wrangle_data(collection)
print(df.head())
4. Exploratory Data Analysis (ACF & PACF Plots):
python
Copy code
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

# Plot ACF and PACF
plt.figure(figsize=(12, 6))
plt.subplot(121)
plot_acf(df, ax=plt.gca(), lags=50)
plt.subplot(122)
plot_pacf(df, ax=plt.gca(), lags=50)
plt.show()
5. Modeling ARIMA:
python
Copy code
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error

# Train ARIMA model
model = ARIMA(df, order=(1,1,1))  # (p,d,q) values can be adjusted based on ACF/PACF
model_fit = model.fit()

# Forecast
forecast = model_fit.forecast(steps=30)

# Evaluate forecast
y_test = df[-30:]  # Last 30 values as test data
mae = mean_absolute_error(y_test, forecast)
print(f'Mean Absolute Error (MAE): {mae}')
6. Walk-Forward Validation:
python
Copy code
# Implement walk-forward validation (rolling forecast)
history = df[:len(df)-30]
test = df[-30:]

predictions = list()
for i in range(len(test)):
    model = ARIMA(history, order=(1,1,1))  # (p,d,q)
    model_fit = model.fit()
    output = model_fit.forecast(steps=1)
    predictions.append(output[0])
    history = history.append(test.iloc[i:i+1])

# Plot predictions vs actual values
import plotly.graph_objs as go
trace_actual = go.Scatter(x=test.index, y=test, mode='lines', name='Actual')
trace_pred = go.Scatter(x=test.index, y=predictions, mode='lines', name='Predicted')
layout = go.Layout(title='ARIMA Model Predictions vs Actual',
                   xaxis={'title': 'Date'},
                   yaxis={'title': 'PM2.5'})
fig = go.Figure(data=[trace_actual, trace_pred], layout=layout)
fig.show()
Conclusion
This project applies the ARIMA model to predict PM2.5 air quality levels in Nairobi, Kenya. By cleaning, analyzing, and splitting the data, then applying ARIMA with walk-forward validation, we are able to create a robust forecasting model. The results are visualized using Plotly for easy interpretation.

Contributions and Future Work
Experiment with different ARIMA hyperparameters for model optimization.
Explore other models like SARIMA or LSTM for more complex time series forecasting.
