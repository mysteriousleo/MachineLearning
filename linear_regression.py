import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Sample data
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 3, 4, 5, 6])

# Perform linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

# Define a function for the regression line
def regression_line(x):
    return slope * x + intercept

# Calculate the predicted values for the regression line
predicted_values = regression_line(x)

# Create a scatter plot of the data points
plt.scatter(x, y, label='Data')

# Plot the regression line
plt.plot(x, predicted_values, color='red', label='Regression Line')

# Add labels and a legend
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

# Display the plot
plt.show()

# Print regression statistics
print(f"Slope: {slope}")
print(f"Intercept: {intercept}")
print(f"R-squared: {r_value**2}")
print(f"P-value: {p_value}")
print(f"Standard Error: {std_err}")
