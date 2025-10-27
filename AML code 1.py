import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Actual and Predicted Sales
actual_sales = np.array([200, 220, 250, 270, 300, 320, 340, 360, 380, 400])
predicted_sales = np.array([210, 215, 245, 280, 295, 310, 345, 355, 390, 405])

# Statistical Measures
mse = mean_squared_error(actual_sales, predicted_sales)
mae = mean_absolute_error(actual_sales, predicted_sales)
rmse = np.sqrt(mse)
r2 = r2_score(actual_sales, predicted_sales)

print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-Squared (R²):", r2)
