from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import os
from PIL import Image
import io
import base64
import logging
from werkzeug.utils import secure_filename
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Class labels
class_names = ['Northen Leaf Blight', 'Common Rust', 'Gray Leaf Spot', 'Healthy']

def load_model():
    """Load the trained model with proper error handling"""
    model_path = r'D:\python project\corn_leaf_disease_detection v2 v2\model\corn_densenet_model.h5'
    
    try:
        logger.info(f"Loading model from: {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        model = tf.keras.models.load_model(model_path)
        logger.info("Model loaded successfully!")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

# Load model at startup
try:
    model = load_model()
except Exception as e:
    logger.error(f"Failed to initialize application: {str(e)}")
    exit(1)

def allowed_file(filename):
    """Check if the file has an allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image):
    """Preprocess image for model prediction"""
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Resize and normalize
        image = image.resize((224, 224))
        img_array = tf.keras.utils.img_to_array(image)
        img_array = tf.expand_dims(img_array, axis=0) / 255.0
        return img_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

def predict_image(image):
    """Make prediction on preprocessed image"""
    try:
        predictions = model.predict(image)
        predicted_class = class_names[np.argmax(predictions)]
        confidence = float(np.max(predictions))
        
        # Get all predictions with percentages
        all_predictions = [
            {
                'class': class_names[i],
                'confidence': float(predictions[0][i]) * 100
            } for i in range(len(class_names))
        ]
        
        # Sort by confidence (descending)
        all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        return predicted_class, confidence, all_predictions
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html', class_names=class_names)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image prediction requests"""
    try:
        # Check if file was uploaded
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Validate file
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload a JPG or PNG image.'}), 400
        
        # Secure filename and save temporarily
        filename = secure_filename(file.filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)
        
        try:
            # Process image
            image = Image.open(temp_path)
            img_array = preprocess_image(image)
            
            # Make prediction
            predicted_class, confidence, all_predictions = predict_image(img_array)
            
            # Log prediction details to terminal
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n[{timestamp}] Prediction Results for: {filename}")
            print("="*50)
            for pred in all_predictions:
                print(f"{pred['class']}: {pred['confidence']:.2f}%")
            print("="*50)
            print(f"Final Prediction: {predicted_class} ({confidence*100:.2f}% confidence)\n")
            
            # Convert image to base64 for display
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            return jsonify({
                'success': True,
                'prediction': predicted_class,
                'confidence': round(confidence * 100, 2),
                'all_predictions': all_predictions,
                'image': img_str,
                'filename': filename
            })
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return jsonify({'error': 'Error processing image'}), 500
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

if __name__ == '__main__':
    # Create upload directory if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    print("\nCorn Disease Classification Server Ready!")
    print("Waiting for image uploads...\n")
    app.run(debug=True, host='0.0.0.0', port=5000)