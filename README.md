# Ex.No: 6               HOLT WINTERS METHOD
### Date: 30-09-2025



### AIM:  
To holy winters method of Indian Ocean

### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:
```py
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pandas as pd

# ----------------------------
# Load and prepare the dataset
# ----------------------------
file_path = 'Sunspots.csv'      # your uploaded file
data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')

# Rename value column
data.rename(columns={'Monthly Mean Total Sunspot Number': 'Value'}, inplace=True)

# Ensure numeric and drop NaNs
data['Value'] = pd.to_numeric(data['Value'], errors='coerce')
data = data.dropna(subset=['Value'])

# Resample to monthly frequency (if not already monthly)
monthly_data = data['Value'].resample('MS').mean()

# ----------------------------
# Split into Train & Test
# ----------------------------
train_data = monthly_data[:int(0.9 * len(monthly_data))]
test_data = monthly_data[int(0.9 * len(monthly_data)):]

# ----------------------------
# Fit Holt-Winters model (Additive trend)
# ----------------------------
fitted_model = ExponentialSmoothing(train_data, trend='add', seasonal=None).fit()

# Forecast for test period
test_predictions = fitted_model.forecast(len(test_data))

# ----------------------------
# Graph 1: Test Data vs Predictions
# ----------------------------
plt.figure(figsize=(12,6))
test_data.plot(label='Actual Test Data', marker='o')
test_predictions.plot(label='Predicted Test Data', marker='x')
plt.title('Sunspots: Test Data vs Predictions')
plt.xlabel('Date')
plt.ylabel('Sunspot Number')
plt.legend()
plt.show()

# ----------------------------
# Evaluate performance
# ----------------------------
mae = mean_absolute_error(test_data, test_predictions)
mse = mean_squared_error(test_data, test_predictions)
print(f"Mean Absolute Error = {mae:.2f}")
print(f"Mean Squared Error = {mse:.2f}")

# ----------------------------
# Fit on Full Data & Forecast 12 Months Ahead
# ----------------------------
final_model = ExponentialSmoothing(monthly_data, trend='add', seasonal=None).fit()
forecast_predictions = final_model.forecast(steps=12)

# ----------------------------
# Graph 2: Final 12-Month Forecast
# ----------------------------
plt.figure(figsize=(12,6))
monthly_data.plot(label='Original Data', legend=True)
forecast_predictions.plot(label='12-Month Forecast', marker='o', color='green')
plt.title('Sunspots: Final 12-Month Forecast')
plt.xlabel('Date')
plt.ylabel('Sunspot Number')
plt.legend()
plt.show()
```

### OUTPUT:
#### GIVEN DATA:
```
Columns in dataset: Index(['Unnamed: 0', 'Date', 'Monthly Mean Total Sunspot Number'], dtype='object')
   Unnamed: 0        Date  Monthly Mean Total Sunspot Number
0           0  1749-01-31                               96.7
1           1  1749-02-28                              104.3
2           2  1749-03-31                              116.7
3           3  1749-04-30                               92.8
4           4  1749-05-31                              141.7
Date
1749-01-01     96.7
1749-02-01    104.3
1749-03-01    116.7
1749-04-01     92.8
1749-05-01    141.7
Freq: MS, Name: Monthly Mean Total Sunspot Number, dtype: float64
```
#### SCALED DATA SPOT:
<img width="561" height="453" alt="image" src="https://github.com/user-attachments/assets/f9faf5f9-6f19-4c05-af73-e5b75dc58c77" />

#### DECOMPOSED PLOT:
<img width="630" height="470" alt="image" src="https://github.com/user-attachments/assets/8de43301-5e44-460a-a168-b887120ea800" />

#### TEST_PREDICTION:

<img width="838" height="545" alt="image" src="https://github.com/user-attachments/assets/0a0b2d78-cbad-4838-86b4-59d8f62b4b08" />

```
Test RMSE: 0.16751780015256632
Scaled Data Std Dev: 0.17049039808654087
Scaled Data Mean: 1.2053711071952424
```

#### FINAL_PREDICTION
<img width="846" height="545" alt="image" src="https://github.com/user-attachments/assets/d65cba85-9180-4913-855d-1d5635e3fd6b" />

### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
