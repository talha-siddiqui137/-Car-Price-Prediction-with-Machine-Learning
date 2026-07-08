from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd

df = pd.read_csv("car data.csv")

# df_label = df.copy()
# label = LabelEncoder()

# df_label["Fuel_Type"] = label.fit_transform(df_label["Fuel_Type"])
# df_label["Selling_type"] = label.fit_transform(df_label["Selling_type"])
# df_label["Transmission"] = label.fit_transform(df_label["Transmission"])


# df_label = pd.get_dummies(df_label, columns=['Car_Name'], dtype=int)

# print(df_label)

categorical_columns = ["Car_Name", "Fuel_Type", "Selling_type", "Transmission"]

df_label = pd.get_dummies(df, columns=categorical_columns, dtype=int)

print(df_label)