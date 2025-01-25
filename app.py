from flask import Flask, request, jsonify
from transformers import pipeline
from datetime import datetime
import numpy as np

app = Flask(__name__)

summarizer = pipeline("summarization")
ner_extractor = pipeline("ner", grouped_entities=True)
sentiment_analyzer = pipeline("sentiment-analysis")

history = []

def make_json_serializable(data):
    """
    Recursively converts numpy.float32 (and other incompatible types) to JSON-serializable Python types.
    """
    if isinstance(data, dict):
        return {k: make_json_serializable(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [make_json_serializable(v) for v in data]
    elif isinstance(data, np.float32):
        return float(data)
    elif isinstance(data, np.int32):
        return int(data)
    return data

@app.route('/process', methods=["POST"])
def process_text():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "invalid input, 'text' field is required"}), 400

        text = data["text"]

        if not isinstance(text, str) or len(text.strip()) == 0:
            return jsonify({"error": "text must be a non empty string"}), 400

        summary = summarizer(text, max_length=50, min_length=25, do_sample=False)[0]["summary_text"]
        entities = ner_extractor(text)
        sentiment = sentiment_analyzer(text)[0]
        
        result = {
            "summary": summary,
            "entities": entities,
            "sentiment": sentiment,
            "timestamp": datetime.now().isoformat()
        }
        
        serializable_reponse = make_json_serializable(result)
        history.append({"summary": summary, "sentiment": sentiment})
        return jsonify(serializable_reponse), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/history', methods=['GET'])
def get_history():
    return jsonify(history), 200

@app.route('/clear_history', methods=['DELETE'])
def clear_history():
    global history
    history = []
    return jsonify({"message": "history cleared successfully"}), 200

@app.route('/process_bulk', methods=["POST"])
def process_bulk():
    try:
        data = request.get_json()
        if not data or "texts" not in data:
            return jsonify({"error": "invalid input, 'texts' is required"}), 400

        texts = data["texts"]

        if not isinstance(texts, list) or not all(isinstance(text, str) and len(text.strip()) > 0 for text in texts):
            return jsonify({"error": "'text' must be a list of non empty strings"}), 400

        results = []
        for text in texts:
            summary = summarizer(text, max_length=50, min_length=25, do_sample=False)[0]["summary_text"]
            entities = ner_extractor(text)
            sentiment = sentiment_analyzer(text)[0]

            result = {
                "text": text,
                "summary": summary,
                "entities": entities,
                "sentiment": sentiment,
                "timestamp": datetime.now().isoformat()
            }
            
            serializable_result = make_json_serializable(result)
            history.append({"summary": summary, "sentiment": sentiment})
            results.append(serializable_result)
        return jsonify(results), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Returns statistics about the processed texts."""
    num_entries = len(history)
    sentiment_counts = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
    
    for entry in history:
        sentiment = entry.get("sentiment", {}).get("label", "NEUTRAL")
        sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1

    stats = {
        "total_processed": num_entries,
        "sentiment_breakdown": sentiment_counts
    }
    
    return jsonify(stats), 200

if __name__ == "__main__":
    app.run(debug=True)