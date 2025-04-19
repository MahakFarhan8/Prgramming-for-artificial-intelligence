from flask import Flask, render_template, request, jsonify
from main import get_chatbot_response, analyze_sentiment  # import functions

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json["message"]
    
    if user_input.lower() == "sentiment":
        return jsonify({"response": "Tell me a sentence and I'll analyze the mood!"})
    
    elif user_input.startswith("analyze:"):
        sentiment_text = user_input.replace("analyze:", "").strip()
        sentiment_result = analyze_sentiment(sentiment_text)
        return jsonify({"response": f"Sentiment Analysis Result {sentiment_result}"})
    
    else:
        response = get_chatbot_response(user_input)
        return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
