import joblib
import pandas as pd

model = joblib.load("model/model.pkl")

def predict_prices(appliances):
    df = pd.DataFrame(appliances)
    df['capacity_l'].fillna(df['capacity_l'].median(), inplace=True)
    df['star_rating'].fillna(3, inplace=True)
    predictions = model.predict(df)
    return predictions.tolist()
