import joblib
from sklearn.feature_extraction.text import CountVectorizer

def predict_email(text):
    # Load model and vectorizer
    model = joblib.load('spam_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')

    # Transform input
    vector = vectorizer.transform([text])
    
    # Predict
    prediction = model.predict(vector)[0]

    # Map to label
    label = 'HAM' if prediction == 0 else 'SPAM'
    return label

# Test prediction
print("Prediction:", predict_email("This is your last chance to claim your reward."))
