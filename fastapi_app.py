# fastapi_app.py

from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import pandas as pd

app = FastAPI()

# Load and preprocess the dataset
data = pd.read_csv('/home/mulibonface187/coursebot/chatbot_dataset.csv')
data['Question'] = data['Question'].apply(lambda x: ' '.join(x.lower().split()))

# Vectorize the dataset and train the model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['Question'])
y = data['Answer']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

class UserInput(BaseModel):
    question: str

@app.post("/predict/")
def predict(user_input: UserInput):
    # Preprocess user input for case-insensitive comparison
    user_input_lower = ' '.join(user_input.question.lower().split())
    # Vectorize the user input and predict the response
    X_user = vectorizer.transform([user_input_lower])
    prediction = model.predict(X_user)
    response = prediction[0] if prediction else "Sorry, I don't understand."
    return {"response": response}
