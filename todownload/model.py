import json
import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import csv
import pickle

# Load tokenizer and model for intent recognition
tokenizer = AutoTokenizer.from_pretrained("v5")
model = AutoModelForSequenceClassification.from_pretrained("v5", from_tf=False)

# Load label classes from label_classes.txt
with open("v5/label_classes.txt", "r") as f:
    label_classes = [line.strip() for line in f]

# Load best_model and scaler from the pavana folder
with open("pavana/best_model.pkl", "rb") as model_file:
    best_model = pickle.load(model_file)
print("XGBoost model loaded:", best_model)  # Debug output

with open("pavana/scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)
print("Scaler loaded:", scaler)  # Debug output

# Helper class to manage structured responses
class ResponseManager:
    def __init__(self, response_file):
        with open(response_file, "r") as f:
            self.responses = json.load(f)
    
    def get_response(self, intent):
        response_data = self.responses.get(intent)
        if response_data:
            main_message = response_data.get("main", "I'm not sure how to help with that.")
            details = response_data.get("details", [])
            response_text = main_message
            if details:
                response_text += "\n" + "\n".join(f"- {detail}" for detail in details)
            return response_text
        return "I'm not sure how to help with that."

# Initialize ResponseManager
response_manager = ResponseManager("responses.json")

# Global state tracking variables
user_logged_in = False
expecting_feedback = False
last_user_query = ""
prompting_username = False
prompting_password = False
entered_username = ""
feedback_phase = False  # New state to indicate feedback phase

# Login function
def login(username, password):
    global entered_username
    try:
        with open("user_db.csv", mode="r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row["username"] == username and row["password"] == password:
                    entered_username = username
                    print("Login successful!")
                    return True
            print("Login failed. Incorrect username or password.")
            return False
    except FileNotFoundError:
        print("User database not found!")
        return False

# Retrieve user features for loan prediction
def get_user_features(username):
    try:
        # Load user data and clean up column names
        user_data = pd.read_csv("user_db.csv")
        user_data.columns = user_data.columns.str.strip()

        # Locate the user row
        user_row = user_data[user_data["username"] == username].copy()
        if not user_row.empty:
            # Align features to model training columns, filling missing ones with 0
            user_features = user_row.reindex(columns=scaler.feature_names_in_, fill_value=0)

            # Scale features and return
            scaled_features = scaler.transform(user_features)
            print("User features successfully scaled:", scaled_features)  # Debug output
            return scaled_features
        else:
            print("User data not found in the database.")
            return None
    except Exception as e:
        print(f"Error retrieving user data: {e}")
        return None

# Predict loan eligibility using best_model
def predict_loan_eligibility(username):
    print("Entering predict_loan_eligibility")  # Debug output
    features = get_user_features(username)
    if features is not None:
        # Predict probabilities using best_model
        probabilities = best_model.predict_proba(features)
        print("Probabilities from best_model:", probabilities)  # Debug output
        
        # Extract the probability for the "approved" class (assuming class 1 = "approved")
        approval_probability = probabilities[0][1]
        print("Approval probability:", approval_probability)  # Debug output
        return f"The approval probability for this loan application is {approval_probability:.2f}%"
    
    return "Unable to retrieve user data for prediction."

# Intent formatting and response handling
def format_intent_probabilities(sorted_intents):
    formatted = "\nIntent probabilities:\n"
    for intent_name, prob in sorted_intents:
        formatted += f"{intent_name}: {prob:.2f}%\n"
    return formatted

# Define the list of valid intents
valid_intents = [
    "IDOOS",
    "OOS",
    "account_info",
    "apply_for_loan",
    "cancel_loan",
    "check_fees",
    "check_loan_payments",
    "customer_service",
    "find_atm",
    "find_branch",
    "human_agent",
    "reset_password",
    "transfer_info"
]

def get_response(user_input):
    global expecting_feedback, last_user_query, user_logged_in
    global prompting_username, prompting_password, entered_username
    global feedback_phase  # Use the new state

    print("Received user input:", user_input)  # Debug output

    # If in feedback phase, process the user's input
    if feedback_phase:
        if user_input.strip().lower() == "no":
            # Show all possible intents when the user types "No"
            available_intents = "\n".join(valid_intents)  # Get all valid intent names
            feedback_phase = True  # Reset feedback phase
            return (
                "Thank you for your feedback! Here are the possible intents:\n"
                f"{available_intents}\n"
                "Please type the correct intent name from the list."
            )
        else:
            # Check if the input is a valid intent
            correct_intent = user_input.strip()  # User specifies the correct intent
            if correct_intent in valid_intents:
                record_feedback(last_user_query, correct_intent)  # Record the feedback
                feedback_phase = False  # Reset the feedback phase
                return "Thank you! Your feedback has been recorded for further training."
            else:
                # If the input is not valid, treat it as a normal command
                print("Input not in valid intents, processing as normal inquiry.")  # Debug output
                feedback_phase = False  # Reset feedback phase for normal processing

    # If feedback phase is not active, proceed with normal command processing
    last_user_query = user_input
    inputs = tokenizer(user_input, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probabilities = F.softmax(logits, dim=1)[0]
    predicted_label = torch.argmax(logits, dim=1).item()
    confidence_score = probabilities[predicted_label].item() * 100
    intent = label_classes[predicted_label]  # Map predicted label to intent using label_classes

    print("Detected intent:", intent)  # Debug output
    print("Confidence score:", confidence_score)  # Debug output

    if intent == "apply_for_loan" and not user_logged_in:
        prompting_username = True
        return "To apply for a loan, please enter your username."

    if intent == "apply_for_loan" and user_logged_in:
        print("Predicting loan eligibility for user:", entered_username)
        eligibility_result = predict_loan_eligibility(entered_username)
        print("Eligibility result:", eligibility_result)  # Debug output
        return eligibility_result

    response_text = f"Intent detected: {intent} (Confidence: {confidence_score:.2f}%)\n"
    if confidence_score < 90.0:
        response_text += (
            "I'm sorry, I don't quite understand. Could you please rephrase it or "
            "refer to the list of things I can assist you with:\n"
            "- Apply for a loan\n"
            "- Check loan payments\n"
            "- Cancel a transfer\n"
            "- Find an ATM\n"
            "- and more in the loan and transfer domains."
        )
        expecting_feedback = True  # Only ask for feedback if the confidence is low
        feedback_phase = True  # Enter feedback phase
        response_text += "\nDid this answer your question? (Please type 'No' to provide additional feedback or simply ignore this message.)"
    else:
        response_text += response_manager.get_response(intent)

    print("Final response:", response_text)  # Debug output

    expecting_feedback = True  # Set to True to prompt for feedback after providing an answer
    feedback_phase = True  # Enter feedback phase to expect feedback

    return response_text




def record_feedback(user_query, correct_intent):
    feedback_df = pd.DataFrame([[user_query, correct_intent]], columns=["User Query", "Correct Intent"])
    try:
        if os.path.exists("totrain.xlsx"):
            existing_df = pd.read_excel("totrain.xlsx", engine="openpyxl")
            feedback_df = pd.concat([existing_df, feedback_df], ignore_index=True)
        with pd.ExcelWriter("totrain.xlsx", engine="openpyxl") as writer:
            feedback_df.to_excel(writer, index=False)
        print("Feedback saved successfully")  # Debug output
        return "Thank you! Your feedback has been recorded for further training."
    except Exception as e:
        print(f"Failed to save feedback: {e}")
        return f"Failed to save feedback: {e}"

# Example usage
if __name__ == "__main__":
    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            break
        response = get_response(user_input)
        print("Bot:", response)
