# TASK : Car Price Prediction with Machine Learning 
# ● Collect car-related features like brand goodwill, horsepower, mileage, etc. 
# ● Train a regression model to predict car prices based on these features. 
# ● Handle data preprocessing, feature engineering, and model evaluation 
# ● Use Python libraries like Pandas, Scikit-learn and Matplotlib for the workflow. 
# ● Understand real-world applications of machine learning in price prediction.


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import numpy as np
import pandas as pd

df = pd.read_csv("car data.csv")

current_year = 2024
df["Car_Age"] = current_year - df["Year"]
df.drop("Year", axis=1, inplace = True)

categorical_columns = ["Car_Name", "Fuel_Type", "Selling_type", "Transmission"]

df_label = pd.get_dummies(df, columns=categorical_columns, dtype=int)

X = df_label.drop('Selling_Price', axis = 1)
y = df_label['Selling_Price']

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2 , random_state=42)

model = LinearRegression()

model.fit(X_train,y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R² Score:", r2)

