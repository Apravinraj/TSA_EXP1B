### Developed by: Pravin Raj A
### Register Number: 212222240079
### Date :

# Ex.No: 1B            CONVERSION OF NON STATIONARY TO STATIONARY DATA
# Date: 

### AIM:
To perform regular differncing,seasonal adjustment and log transformatio on Bit-coin price prediction data.

### ALGORITHM:

1. Import the required packages like pandas and numpy
2. Read the data using the pandas
3. Perform the data preprocessing if needed and apply regular differncing,seasonal adjustment,log transformation.
4. Plot the data according to need, before and after regular differncing,seasonal adjustment,log transformation.
5. Display the overall results.

### PROGRAM:

```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Load the CSV file
file_path = '/content/BTC-USD(1).csv'  # Replace with your file path
df = pd.read_csv(file_path)

# Convert the Date column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Sort the data by Date just in case
df = df.sort_values('Date')

# Log Transformation
df['Close_log'] = np.log(df['Close'])

# Differencing
df['Close_diff'] = df['Close'].diff()

# Seasonal Differencing (Assume a weekly seasonality for demonstration, you can change it)
df['Close_seasonal_diff'] = df['Close'].diff(7)

# Detrending by Regression
# Fit a linear model
X = np.arange(len(df))
Y = df['Close']
model = np.polyfit(X, Y, 1)
trend = np.polyval(model, X)
df['Close_detrended'] = df['Close'] - trend

# Plotting the transformations
plt.figure(figsize=(14, 10))

plt.subplot(4, 1, 1)
plt.plot(df['Date'], df['Close_log'], label='Log Transformed', color='green')
plt.title('Log Transformation')
plt.grid(True)

plt.subplot(4, 1, 2)
plt.plot(df['Date'], df['Close_diff'], label='Differenced', color='red')
plt.title('Differencing')
plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(df['Date'], df['Close_seasonal_diff'], label='Seasonal Differencing (Lag 7)', color='purple')
plt.title('Seasonal Differencing')
plt.grid(True)

plt.subplot(4, 1, 4)
plt.plot(df['Date'], df['Close_detrended'], label='Detrended by Regression', color='blue')
plt.title('Detrended by Regression')
plt.grid(True)

plt.tight_layout()
plt.show()

```

### OUTPUT:

![Screenshot 2024-08-24 220337](https://github.com/user-attachments/assets/6a0a8d81-ce9e-4214-ac37-4ecddce932a0)



### RESULT:
Thus we have created the python code for the conversion of non stationary to stationary data on Bit-coin price prediction data.

