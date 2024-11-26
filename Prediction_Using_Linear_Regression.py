# Import necessary libraries
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import pytz
from pymongo import MongoClient
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Connect to MongoDB database
client = MongoClient(host="localhost", port=27017)  # Connect to MongoDB running locally
db = client["air-quality"]  # Access the 'air-quality' database
nairobi = db["nairobi"]  # Access the 'nairobi' collection in the database


# Function to wrangle and process data from MongoDB collection
def wrangle(collection):
    """
    This function loads data from the MongoDB collection, processes it,
    removes outliers, resamples it to hourly intervals, and creates a lag feature.
    """
    # Query MongoDB for specific measurements and data points
    results = collection.find(
        {"metadata.site": 29, "metadata.measurement": "P2"},  # Query parameters
        projection={"P2": 1, "timestamp": 1, "_id": 0},  # Select only 'P2' and 'timestamp' fields
    )

    # Convert the results into a pandas DataFrame
    df = pd.DataFrame(results).set_index("timestamp")

    # Localize the timezone to 'Africa/Nairobi' from 'UTC'
    df.index = df.index.tz_localize("UTC").tz_convert("Africa/Nairobi")

    # Remove outliers (filter out values above 500 for 'P2')
    df = df[df["P2"] < 500]

    # Resample the data to a 1-hour window and forward fill missing values
    df = df["P2"].resample("1H").mean().fillna(method="ffill").to_frame()

    # Create a lag feature for 'P2' (previous hour's 'P2' value)
    df["P2.L1"] = df["P2"].shift(1)
    df['P2.L1'] = df['P2.L1'].fillna(method='ffill')  # Forward fill any missing lag values

    # Drop any remaining rows with missing values
    df.dropna(inplace=True)

    return df  # Return the cleaned DataFrame


# Wrangle the data from the MongoDB collection
df = wrangle(nairobi)

# Visualize the distribution of PM2.5 (P2) readings
fig, ax = plt.subplots(figsize=(15, 6))
df["P2"].plot(kind="box", vert=False, title="Distribution of PM2.5 Readings", ax=ax)

# Time series plot of PM2.5 over time
fig, ax = plt.subplots(figsize=(15, 6))
df["P2"].plot(xlabel="Time", ylabel="PM2.5", title="PM2.5 Time Series", ax=ax)

# Plot the weekly rolling average of PM2.5 values
fig, ax = plt.subplots(figsize=(15, 6))
df["P2"].rolling(168).mean().plot(ax=ax, ylabel="PM2.5", title="Weekly Rolling Average")

# Auto-correlation: Scatter plot of lag vs original 'P2'
df["P2.L1"] = df["P2"].shift(1)
df.dropna(inplace=True)

# Calculate correlation between 'P2' and 'P2.L1'
corr = df.corr()
print(corr)  # Print the correlation matrix

# Scatter plot of 'P2' vs lagged 'P2.L1' values
fig, ax = plt.subplots(figsize=(6, 6))
plt.scatter(x=df["P2.L1"], y=df["P2"])
plt.plot([0, 120], [0, 120], linestyle="--", color="orange")  # Line for ideal correlation
plt.xlabel("P2.L1")  # X-axis label
plt.ylabel("P2")  # Y-axis label
plt.title("Auto Correlation")  # Plot title

# Prepare data for linear regression model
target = "P2"  # Define target variable ('P2')
features = ["P2.L1"]  # Define feature variable ('P2.L1')

y = df[target]  # Target variable (PM2.5 values)
X = df[features]  # Feature variable (lagged PM2.5 values)

# Split the data into training and testing sets (80% training, 20% testing)
cutoff = int(len(X) * 0.8)  # 80% training data
X_train, y_train = X.iloc[:cutoff], y.iloc[:cutoff]
X_test, y_test = X.iloc[cutoff:], y.iloc[cutoff:]

# Calculate and print the baseline Mean Absolute Error (MAE) for the training data
y_mean = y_train.mean()  # Baseline prediction: mean of 'y_train'
y_pred_baseline = [y_mean] * len(y_train)  # Baseline prediction for all training data
mae_baseline = mean_absolute_error(y_train, y_pred_baseline)  # Calculate MAE for baseline prediction

print("Mean P2 Reading:", round(y_train.mean(), 2))  # Print mean of target variable
print("Baseline MAE:", round(mae_baseline, 2))  # Print baseline MAE

# Train a Linear Regression model
model = LinearRegression()  # Initialize the model
model.fit(X_train, y_train)  # Fit the model to the training data

# Evaluate the model's performance
training_mae = mean_absolute_error(y_train, model.predict(X_train))  # MAE on training set
test_mae = mean_absolute_error(y_test, model.predict(X_test))  # MAE on test set
print("Training MAE:", round(training_mae, 2))  # Print training MAE
print("Test MAE:", round(test_mae, 2))  # Print test MAE

# Print the model's equation (intercept and coefficient)
intercept = model.intercept_.round(2)
coefficient = model.coef_.round(2)
print(f"P2 = {intercept} + ({coefficient} * P2.L1)")  # Linear equation

# Prepare predictions for the test set
if isinstance(y_test, pd.DataFrame):
    y_test_series = y_test.squeeze()  # Convert to Series if y_test is a DataFrame
else:
    y_test_series = y_test  # If already a Series, use it directly

y_pred = model.predict(X_test)  # Get predictions from the model

# Ensure the predictions are a 1D array
if y_pred.ndim > 1:
    y_pred = y_pred.ravel()  # Flatten to 1D if necessary

# Create a DataFrame to compare actual vs predicted values
df_pred_test = pd.DataFrame(
    {"y_test": y_test_series, "y_pred": y_pred}, index=y_test_series.index
)
df_pred_test.head()  # Display first few rows of predictions

# Print shapes of the test set and predictions
import numpy as np

print(f"Shape of y_test: {np.shape(y_test)}")
print(f"Shape of predictions: {np.shape(y_pred)}")

# Visualize the actual vs predicted values using Plotly
fig = px.line(df_pred_test, labels={"values": "P2"})  # Create a line plot
fig.show()  # Show the plot
