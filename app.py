# app.py
from flask import Flask, request, jsonify
import os
from PIL import Image
import io
from flask_cors import CORS 

from transformers import ViTImageProcessor, ViTForImageClassification
import torch

app = Flask(__name__)
CORS(app)
processor = ViTImageProcessor.from_pretrained("C:/Users/User/Desktop/ai-image-detector")
model = ViTForImageClassification.from_pretrained("C:/Users/User/Desktop/ai-image-detector")


def analyze_image(image_data):
    try:
        img = Image.open(io.BytesIO(image_data))

        if img.mode != 'RGB':
            img = img.convert('RGB')

        inputs = processor(images=img, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            predictions = outputs.logits.softmax(dim=-1)
        
        scores = predictions[0].tolist()
        results = [
            {"label": "REAL", "score": scores[0]},
            {"label": "FAKE", "score": scores[1]}
        ]
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return {
            "is_ai_generated": results[0]["label"] == "FAKE",
            "prediction": results[0]["label"],
            "confidence": f"{results[0]['score']*100:.2f}%",
            "detailed_scores": [
                f"{r['label']}: {r['score']*100:.2f}%" 
                for r in results
            ]
        }
    
    except Exception as e:
        return {"error": str(e)}

@app.route('/get-image', methods=['POST'])
def get_image():
    try:
        image_data = None
        
        if 'image' in request.files:
            image = request.files['image']
            image_data = image.read()
        
        elif 'url' in request.json:
            import requests
            url = request.json['url']
            response = requests.get(url)
            image_data = response.content
        
        else:
            return jsonify({"error": "No image provided. Please upload an image file or provide a URL."}), 400
        
        # Analyze the image
        results = analyze_image(image_data)
        
        if "error" in results:
            return jsonify({"error": results["error"]}), 400
        print("result", results)
        return jsonify(results)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))