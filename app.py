# Install transformers from source - only needed for versions <= v4.34
# pip install git+https://github.com/huggingface/transformers.git
# pip install accelerate

import torch
from transformers import pipeline
from flask import Flask, request, jsonify
import re
import json
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize the pipeline globally
pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")

def clean_llm_response(response):
    def standardize_keys(data):
        """Recursively standardize keys in JSON data."""
        if isinstance(data, dict):
            return {('latitude' if k == 'lat' else 'longitude' if k == 'lon' else k): standardize_keys(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [standardize_keys(item) for item in data]
        else:
            return data

    try:
        # Attempt to find JSON content between triple backticks first
        json_matches = re.findall(r'```(?:\w*\n)?(\{.*?\}|\[.*?\])```', response, re.DOTALL)
        for json_str in json_matches:
            try:
                data = json.loads(json_str)
                return standardize_keys(data)
            except json.JSONDecodeError:
                continue  # Try the next match if JSON parsing fails

        # If no JSON found between backticks, try to extract JSON-like structures
        json_like_matches = re.findall(r'(\{.*?\}|\[.*?\])', response, re.DOTALL)
        for json_str in json_like_matches:
            try:
                data = json.loads(json_str)
                return standardize_keys(data)
            except json.JSONDecodeError:
                continue  # Try the next match if JSON parsing fails

        # Attempt to extract latitude and longitude from plain text with "Lat: <value> Lng: <value>"
        lat_lng_match = re.search(r'Lat:([-\d.]+)\s+Lng:([-\d.]+)', response)
        if lat_lng_match:
            latitude = float(lat_lng_match.group(1))
            longitude = float(lat_lng_match.group(2))
            return {"latitude": latitude, "longitude": longitude}

        # Attempt to extract latitude and longitude from comma-separated values
        lat_lng_comma_match = re.search(r'([-\d.]+),\s*([-\d.]+)', response)
        if lat_lng_comma_match:
            latitude = float(lat_lng_comma_match.group(1))
            longitude = float(lat_lng_comma_match.group(2))
            return {"latitude": latitude, "longitude": longitude}

        # Attempt to extract latitude and longitude with directional suffixes
        lat_lng_directional_match = re.search(r'([-\d.]+)([NS])\s+([-\d.]+)([EW])', response)
        if lat_lng_directional_match:
            latitude = float(lat_lng_directional_match.group(1))
            if lat_lng_directional_match.group(2) == 'S':
                latitude = -latitude
            longitude = float(lat_lng_directional_match.group(3))
            if lat_lng_directional_match.group(4) == 'W':
                longitude = -longitude
            return {"latitude": latitude, "longitude": longitude}

        # Return the actual response if no valid JSON or coordinates are found
        return response
    except Exception as e:
        # Return the actual response if any other exception occurs
        return response

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        
        # Log the incoming request JSON
        logging.info(f"Received request JSON: {data}")
        
        if not data or 'question' not in data:
            return jsonify({'error': 'Question is required'}), 400
        if not data or 'context' not in data:
            return jsonify({'error': 'Context is required'}), 400

        # Format messages for the chat
        messages = [
            {
                "role": "user",
                "content": data['context'],
            },
            {"role": "user", "content": data['question']},
        ]
        
        # Generate response
        prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        
        # Clean and transform the response
        raw_response = outputs[0]["generated_text"]
        cleaned_json = clean_llm_response(raw_response)
        
        return jsonify(cleaned_json)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
