import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
     

# Dataset
area = np.array([800, 1000, 1200, 1500, 1800, 2000]).reshape(-1,1)
price = np.array([150, 180, 200, 230, 260, 290])  # in lakhs

     

# Train Model
model = LinearRegression()
model.fit(area, price)

predicted_price = model.predict(area)
     

# Plot
plt.scatter(area, price, color="blue", label="Actual Data")
plt.plot(area, predicted_price, color="red", label="Regression Line")
plt.xlabel("Area (sq ft)")
plt.ylabel("Price (Lakhs)")
plt.legend()
plt.show()

