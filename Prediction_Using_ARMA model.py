# Import necessary libraries
import inspect
import time
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns

from pymongo import MongoClient
from sklearn.metrics import mean_absolute_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Connect to MongoDB and retrieve the "nairobi" collection from the "air-quality" database
client = MongoClient(host="localhost", port=27017)
db = client["air-quality"]
nairobi = db["nairobi"]

# Wrangle data from MongoDB: filter for specific measurements and resample the data
def wrangle(collection, resample_rule="1H"):
    """
    This function reads data from the MongoDB collection, filters it by site and measurement type,
    removes outliers, and resamples the data by the specified time rule.
    """
    results = collection.find(
        {"metadata.site": 29, "metadata.measurement": "P2"},
        projection={"P2": 1, "timestamp": 1, "_id": 0},
    )
    # Read results into DataFrame and set timestamp as index
    df = pd.DataFrame(list(results)).set_index("timestamp")

    # Convert the timestamp to Nairobi local time
    df.index = df.index.tz_localize("UTC").tz_convert("Africa/Nairobi")

    # Remove outliers: assume P2 values higher than 500 are outliers
    df = df[df["P2"] < 500]

    # Resample and forward-fill missing data
    y = df["P2"].resample(resample_rule).mean().fillna(method="ffill")
    return y

# Wrangle data from the "nairobi" collection
y = wrangle(nairobi)
print(y.head())

# Plot the ACF for the data
fig, ax = plt.subplots(figsize=(15, 6))
plot_acf(y, ax=ax)
ax.set_xlabel("Lag [hours]")
ax.set_ylabel("Correlation Coefficient")
plt.show()

# Plot the PACF for the data
fig, ax = plt.subplots(figsize=(15, 6))
plot_pacf(y, ax=ax)
ax.set_xlabel("Lag [hours]")
ax.set_ylabel("Correlation Coefficient")
plt.show()

# Split the data into training and test sets
y_train = y.loc["2018-10-01":"2018-10-31"]
y_test = y.loc["2018-11-01":"2018-11-01"]

# Verify the lengths of the training and test sets
print(f"Training set length: {len(y_train)}")
print(f"Test set length: {len(y_test)}")

# Calculate the baseline Mean Absolute Error (MAE)
y_train_mean = y_train.mean()
y_pred_baseline = [y_train_mean] * len(y_train)
mae_baseline = mean_absolute_error(y_train, y_pred_baseline)
print(f"Baseline Mean P2 Reading: {round(y_train_mean, 2)}")
print(f"Baseline MAE: {round(mae_baseline, 2)}")

# Define ranges for p and q values
p_params = range(0, 25, 8)
q_params = range(0, 3, 1)

# Train ARIMA models with every combination of p and q values
mae_grid = dict()

# Outer loop for p values
for p in p_params:
    mae_grid[p] = list()  # Initialize list to store MAE for each q value
    # Inner loop for q values
    for q in q_params:
        order = (p, 0, q)
        start_time = time.time()
        # Train the ARIMA model
        model = ARIMA(y_train, order=order).fit()
        elapsed_time = round(time.time() - start_time, 2)
        print(f"Trained ARIMA{order} in {elapsed_time} seconds.")
        # Generate in-sample (training) predictions
        y_pred = model.predict()
        # Calculate the MAE for this model
        mae = mean_absolute_error(y_train, y_pred)
        mae_grid[p].append(mae)

# Convert the mae_grid dictionary to a DataFrame
mae_df = pd.DataFrame(mae_grid)
mae_df = mae_df.round(4)
print(mae_df)

# Create a heatmap of the MAE values
sns.heatmap(mae_df, cmap="Blues")
plt.xlabel("p values")
plt.ylabel("q values")
plt.show()

# Check the residuals of the best model (Assume ARIMA(8, 0, 1) is the best model)
best_model = ARIMA(y_train, order=(8, 0, 1)).fit()
fig, ax = plt.subplots(figsize=(15, 12))
best_model.plot_diagnostics(fig=fig)
plt.show()

# Perform walk-forward validation on the test set
y_pred_wfv = pd.Series()
history = y_train.copy()

# Iterate through the test set, making predictions and updating the training history
for i in range(len(y_test)):
    model = ARIMA(history, order=(8, 0, 1)).fit()
    next_pred = model.forecast()
    y_pred_wfv = y_pred_wfv.append(next_pred)
    history = history.append(y_test[next_pred.index])

# Calculate the MAE for walk-forward validation
test_mae = mean_absolute_error(y_test, y_pred_wfv)
print(f"Test MAE (walk-forward validation): {round(test_mae, 2)}")

# Create a DataFrame for the predictions and plot the results
df_predictions = pd.DataFrame(
    {"y_test": y_test, "y_pred_wfv": y_pred_wfv}, index=y_test.index
)
fig = px.line(df_predictions)
fig.update_layout(yaxis_title="P2")
fig.show()
