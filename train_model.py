import pandas as pd, pickle
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv("car data.csv")
df["Car_Age"] = 2025 - df["Year"]                       # feature engineering
df = df.drop(["Car_Name", "Year"], axis=1)
df["Fuel_Type"]    = df["Fuel_Type"].map({"Petrol": 0, "Diesel": 1, "CNG": 2})
df["Seller_Type"]  = df["Seller_Type"].map({"Dealer": 0, "Individual": 1})
df["Transmission"] = df["Transmission"].map({"Manual": 0, "Automatic": 1})

X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("R2   :", round(r2_score(y_test, pred), 3))                       # 0.958
print("RMSE :", round(mean_squared_error(y_test, pred) ** 0.5, 2))      # 0.98
cv = cross_val_score(model, X, y, cv=KFold(5, shuffle=True, random_state=42), scoring="r2")
print("CV R2:", cv.round(3), "mean", round(cv.mean(), 3))               # mean 0.918

with open("car_price_model.pkl", "wb") as f:
    pickle.dump(model, f)
