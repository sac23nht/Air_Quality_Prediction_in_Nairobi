# Air Quality Forecasting with ARIMA Model

## Overview

This project focuses on analyzing and forecasting air quality data for Nairobi using the ARIMA (AutoRegressive Integrated Moving Average) model. The data is sourced from a MongoDB database containing air quality measurements, and the goal is to forecast particulate matter (P2) levels for the city.

The process includes data wrangling, model training and tuning, and performance evaluation using Mean Absolute Error (MAE) as the evaluation metric. A walk-forward validation approach is also employed to test the model’s real-world prediction accuracy.

## Dataset

The data used for this analysis is retrieved from the `air-quality` database stored in MongoDB. Specifically, the dataset comes from the `nairobi` collection, which contains air quality measurements for various locations and times.

The key features of the dataset include:

- **`timestamp`**: The timestamp of when the measurement was taken.
- **`metadata.site`**: The site identifier where the measurement was recorded (site 29 is used for this project).
- **`metadata.measurement`**: The type of air quality measurement. In this case, the focus is on `P2`, which measures particulate matter.
- **`P2`**: The actual particulate matter (PM2.5) level in micrograms per cubic meter (µg/m³), which represents air pollution levels.

The data is resampled hourly and outliers (values higher than 500 µg/m³) are removed to ensure the quality of the analysis.

## Steps and Process

### 1. **Data Extraction and Wrangling**
   - The data is extracted from MongoDB by querying the `nairobi` collection.
   - The timestamp is converted to the local time zone of Nairobi (Africa/Nairobi).
   - The data is filtered to include only measurements for site 29 and the particulate matter (`P2`) type.
   - Outliers (P2 values higher than 500) are removed.
   - The data is resampled into hourly intervals and missing values are forward-filled.

### 2. **Exploratory Data Analysis (EDA)**
   - The **Autocorrelation Function (ACF)** and **Partial Autocorrelation Function (PACF)** plots are generated to explore the temporal dependencies in the data. These plots help identify the potential parameters for the ARIMA model.

### 3. **Train-Test Split**
   - The dataset is split into a **training set** (from October 1, 2018, to October 31, 2018) and a **test set** (on November 1, 2018).
   - This split allows for training the model on past data and testing it on unseen future data.

### 4. **Baseline Model Evaluation**
   - A baseline model is created by predicting the mean value of the training set for all test samples.
   - The Mean Absolute Error (MAE) of this baseline model is calculated to serve as a benchmark for evaluating more complex models.

### 5. **ARIMA Model Training**
   - The ARIMA model is trained using different combinations of `p` (autoregressive order) and `q` (moving average order) parameters, with the model being evaluated based on MAE.
   - A grid search over different `p` and `q` values is performed to find the optimal model.
   - A heatmap is created to visualize the MAE values across different combinations of `p` and `q`.

### 6. **Model Diagnostics**
   - The residuals of the best ARIMA model (based on MAE evaluation) are checked using the `plot_diagnostics()` function. This step ensures that the model’s assumptions (like homoscedasticity and normality of errors) are satisfied.

### 7. **Walk-Forward Validation**
   - Walk-forward validation is implemented on the test set. In this process, the model is retrained at each step using the history of the training and past predictions, and one step ahead is forecasted.
   - This simulates how the model would perform in real-time by updating the model with each new data point.
   - The MAE of the walk-forward validation is calculated and compared with the baseline.

### 8. **Prediction and Visualization**
   - After forecasting the particulate matter for the test set, the predictions are compared with the actual values.
   - A plot is generated to visualize the predicted and actual `P2` values, providing insights into how well the ARIMA model performs.

## ARIMA Model Explanation

ARIMA stands for **AutoRegressive Integrated Moving Average**. It is a time series forecasting method that combines three components:
- **AR (AutoRegressive)**: This component uses the relationship between an observation and several lagged observations (previous time steps).
- **I (Integrated)**: This component involves differencing the series to make it stationary (i.e., the mean and variance are constant over time).
- **MA (Moving Average)**: This component models the relationship between an observation and the residual errors from a moving average model applied to lagged observations.

The ARIMA model is specified by three parameters: **p**, **d**, and **q**:
- **p**: The number of lag observations included in the model (autoregressive order).
- **d**: The number of times the raw observations are differenced to make the series stationary.
- **q**: The size of the moving average window.

For this project, the best ARIMA model is determined through grid search and diagnostics. The model's accuracy is evaluated using MAE and compared against a baseline model to assess its predictive performance.

## Results

- **Baseline MAE**: The baseline MAE was calculated using the mean of the training data, providing a benchmark for model evaluation.
- **ARIMA Model Evaluation**: After training multiple ARIMA models with different `(p, q)` combinations, the model `(8, 0, 1)` emerged as the best based on the lowest MAE.
- **Walk-Forward Validation MAE**: The walk-forward validation results show how well the model generalizes to new data, with the test MAE being calculated for the predicted values.

## Conclusion

The ARIMA model provides a solid approach for forecasting particulate matter levels in Nairobi. Through careful preprocessing, model training, and evaluation, we can accurately predict air quality and make informed decisions. The walk-forward validation step is particularly useful for assessing the real-world applicability of the model.

## Future Work

- Experiment with different ARIMA variations such as SARIMA (Seasonal ARIMA) to account for seasonality in the data.
- Implement additional time series models like Prophet or LSTM (Long Short-Term Memory) networks for comparison.
- Incorporate other air quality measurements to improve the forecasting accuracy.

