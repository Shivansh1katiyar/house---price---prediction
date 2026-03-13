# House Price Prediction Project

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# -----------------------------
# Step 1: Create Dataset
# -----------------------------

data = {
    "area": [1000,1200,1500,1800,2000,2300,2500,2700,3000,3200],
    "bedrooms": [2,2,3,3,3,4,4,4,5,5],
    "bathrooms": [1,2,2,2,3,3,3,4,4,4],
    "age": [10,8,7,6,5,4,3,3,2,1],
    "price": [200000,220000,300000,340000,360000,420000,460000,480000,540000,600000]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)

# -----------------------------
# Step 2: Define Features & Target
# -----------------------------

X = df[["area","bedrooms","bathrooms","age"]]
y = df["price"]

# -----------------------------
# Step 3: Train-Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Step 4: Train Model
# -----------------------------

model = LinearRegression()

model.fit(X_train, y_train)

# -----------------------------
# Step 5: Prediction
# -----------------------------

y_pred = model.predict(X_test)

# -----------------------------
# Step 6: Evaluation
# -----------------------------

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance")
print("Mean Absolute Error:", mae)
print("R2 Score:", r2)

# -----------------------------
# Step 7: Visualization
# -----------------------------

plt.scatter(y_test, y_pred, color="blue")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.show()

# -----------------------------
# Step 8: Predict New House
# -----------------------------

# area, bedrooms, bathrooms, age
new_house = [[2400, 4, 3, 5]]

predicted_price = model.predict(new_house)

print("\nPredicted Price for New House:", predicted_price[0])
