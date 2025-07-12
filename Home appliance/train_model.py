import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# Load dataset
df = pd.read_csv('data/appliances_bulk.csv')

# Fill missing capacity with median
df['capacity_l'].fillna(df['capacity_l'].median(), inplace=True)
df['star_rating'].fillna(3, inplace=True)

X = df[['appliance_type', 'brand', 'capacity_l', 'star_rating', 'features']]
y = df['price']

# Categorical + numerical pipeline
categorical = ['appliance_type', 'brand', 'features']
numerical = ['capacity_l', 'star_rating']

preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical)
], remainder='passthrough')

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Save model
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/model.pkl')
print("✅ Model trained and saved to model/model.pkl")
