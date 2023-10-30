from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load your fine-tuned BERT model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("your_finetuned_bert_model")
tokenizer = AutoTokenizer.from_pretrained("your_finetuned_bert_model")

def predict_visa_chances(transcript):
    inputs = tokenizer(transcript, return_tensors="pt")
    outputs = model(**inputs)
    
    # You might need additional post-processing here depending on your model's output format

    predicted_answer = "High"  # Replace with your model's output
    return predicted_answer



def IdealValuesCalc():
    # Load the dataset from a CSV file
    df = pd.read_csv("DataSet.csv")

    # Split the data into features and labels
    X = df[["Communication Skills", "Confidence", "Knowledge about Destination Country", "Clarity of Purpose", "Financial Preparedness"]]
    y = df["Visa Received"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Get feature importance scores
    feature_importance = clf.feature_importances_

    # Create ideal values based on feature importance
    # Normalize the feature importance scores to sum up to 1
    ideal_values = feature_importance / feature_importance.sum()

    # Print ideal values
    return ideal_values

# Define criterion names and their weights based on ideal values
criterion_weights = {
    "Communication Skills": 0.2,
    "Confidence": 0.15,
    "Knowledge about Destination Country": 0.1,
    "Clarity of Purpose": 0.25,
    "Financial Preparedness": 0.3
}

criterion_names = ["Communication Skills", "Confidence", "Knowledge about Destination Country", "Clarity of Purpose", "Financial Preparedness"]

app = Flask(__name__)

ideal_values = IdealValuesCalc()

def calculate_improvement_areas(user_scores, ideal_values, criterion_weights):
    diff = [user - ideal for user, ideal in zip(user_scores, ideal_values)]
    sorted_diff = sorted(enumerate(diff), key=lambda x: x[1])
    strong_area = criterion_names[sorted_diff[0][0]]  # Determine the strong area
    sorted_diff = sorted_diff[1:]  # Remove the strong area from the list
    improvement_areas = [criterion_names[criteria[0]] for criteria in sorted_diff]
    return improvement_areas, strong_area

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    # Get user input from the form
    communication = int(request.form['communication'])
    confidence = int(request.form['confidence'])
    knowledge = int(request.form['knowledge'])
    clarity = int(request.form['clarity'])
    financial = int(request.form['financial'])

    # Create a list of user scores
    user_scores = [communication, confidence, knowledge, clarity, financial]

    # Ensure user scores are within the range of 0 to 5
    user_scores = [min(5, max(0, score)) for score in user_scores]

    # Calculate feedback based on the user input
    user_transcript = "Combine the user's input here..."
    feedback = predict_feedback(user_transcript)

    return render_template('result.html', feedback=feedback)


if __name__ == '__main__':
    app.run(debug=True)
