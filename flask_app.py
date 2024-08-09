from flask import Flask, render_template, request, jsonify
import requests

# Initialize Flask
flask_app = Flask(__name__)

# Route for the main page
@flask_app.route('/')
def index():
    return render_template('index.html')

# Route to handle user input and get the response from FastAPI
@flask_app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.json.get('question')
    if user_input:
        # Send request to FastAPI backend
        response = requests.post("http://127.0.0.1:8000/predict/", json={"question": user_input})
        response_data = response.json()
        bot_response = response_data.get("response", "Sorry, I don't understand.")
        return jsonify({"response": bot_response})
    return jsonify({"response": "No input provided."})

# Run the Flask app
if __name__ == '__main__':
    flask_app.run(debug=True, port=5000)
