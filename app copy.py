# app.py
from flask import Flask, request, jsonify
import os
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from flask_cors import CORS 

app = Flask(__name__)
CORS(app)

# This is a placeholder for a real AI detection model
# In a real implementation, you would use a pre-trained model specifically for AI image detection
def load_ai_detection_model():
    # Base ResNet model
    base_model = ResNet50(weights='imagenet', include_top=False)
    
    # Add custom layers for AI detection
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(2, activation='softmax')(x)  # 2 classes: AI-generated or real
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # In a real implementation, you would load weights from a trained model
    # model.load_weights('path_to_weights.h5')
    
    return model

# Load the model at startup
model = load_ai_detection_model()

def analyze_image(image_data):
    """
    Analyze image to determine if it's AI-generated
    
    Args:
        image_data: bytes of the image data
    
    Returns:
        dict: Analysis results
    """
    try:
        # Open the image
        img = Image.open(io.BytesIO(image_data))
        
        # Resize and preprocess for the model
        img = img.resize((224, 224))
        img_array = np.array(img)
        
        # Handle grayscale images
        if len(img_array.shape) == 2:
            img_array = np.stack((img_array,) * 3, axis=-1)
        
        # Handle RGBA images
        if img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]
            
        # Preprocess the image
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Make prediction
        # In a real implementation, this would use an actual AI detection model
        # For this example, we'll use a placeholder that randomly classifies images
        preds = model.predict(img_array)
        print("PREDS  ", preds)
        print("PREDS  ", preds[0][0])
        randomNum = np.random.random()
        print("randomNum  ", randomNum)
        # Placeholder: random prediction with 70% chance of being correct
        # In a real implementation, remove this and uncomment the line above
        is_ai_generated = preds[0][0] > 0.5
        confidence = np.random.uniform(0.7, 0.95)
        
        # Extract features that suggest AI generation
        ai_indicators = []
        if is_ai_generated:
            possible_indicators = [
                "Unusual texture patterns",
                "Inconsistent lighting",
                "Symmetry anomalies",
                "Unrealistic details",
                "Artifacts in complex areas"
            ]
            # Select 1-3 random indicators
            num_indicators = np.random.randint(1, 4)
            ai_indicators = np.random.choice(possible_indicators, num_indicators, replace=False).tolist()
            
        return {
            "is_ai_generated": bool(is_ai_generated),
            "confidence": float(preds[0][1]),
            "ai_indicators": ai_indicators
        }
    
    except Exception as e:
        return {"error": str(e)}

@app.route('/get-image', methods=['POST'])
def get_image():
    """
    Endpoint to receive and analyze images
    
    Expects:
        - image: file upload (image file)
        - or url: string (URL to image)
    
    Returns:
        - JSON with analysis results
    """
    try:
        image_data = None
        
        # Check if an image file was uploaded
        if 'image' in request.files:
            image = request.files['image']
            image_data = image.read()
        
        # Check if a URL was provided
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
            
        return jsonify(results)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))