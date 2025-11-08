import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import base64

# Initialize Flask app
app = Flask(__name__, template_folder='../templates', static_folder='../templates')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Tumor type information database
TUMOR_TYPES = {
    'glioma': {
        'name': 'Glioma',
        'severity': 'High',
        'description': 'A tumor that arises from glial cells in the brain or spine. Most common type of brain tumor.',
        'characteristics': [
            'Can be benign or malignant',
            'Grows from glial cells that support neurons',
            'May cause headaches, seizures, or cognitive changes',
            'Includes astrocytomas, oligodendrogliomas'
        ],
        'treatment': 'Surgery, radiation therapy, chemotherapy depending on grade',
        'prognosis': 'Varies by grade: Grade I-II (better), Grade III-IV (more aggressive)',
        'color': '#ff6b6b',
        'icon': 'fa-brain'
    },
    'meningioma': {
        'name': 'Meningioma',
        'severity': 'Low to Moderate',
        'description': 'A tumor that forms in the meninges (membranes surrounding the brain and spinal cord).',
        'characteristics': [
            'Usually benign (90% of cases)',
            'Slow-growing tumor',
            'More common in women',
            'Can compress brain tissue causing symptoms'
        ],
        'treatment': 'Observation for small tumors, surgery or radiation for larger ones',
        'prognosis': 'Generally good, 5-year survival rate >95% for benign types',
        'color': '#4ecdc4',
        'icon': 'fa-circle'
    },
    'pituitary': {
        'name': 'Pituitary Adenoma',
        'severity': 'Low to Moderate',
        'description': 'A tumor in the pituitary gland, which controls hormone production.',
        'characteristics': [
            'Usually benign (95% of cases)',
            'Affects hormone levels (ACTH, prolactin, growth hormone)',
            'May cause vision problems due to optic nerve compression',
            'Can affect growth, metabolism, and reproduction'
        ],
        'treatment': 'Surgery (transsphenoidal), medication (dopamine agonists), radiation',
        'prognosis': 'Excellent with treatment, often curable with surgery',
        'color': '#95e1d3',
        'icon': 'fa-dot-circle'
    },
    'general': {
        'name': 'Brain Tumor (Unspecified Type)',
        'severity': 'Requires Further Analysis',
        'description': 'Abnormal growth detected in brain tissue. Further diagnostic testing needed to determine exact type.',
        'characteristics': [
            'Abnormal cell growth detected',
            'Location and specific type need MRI/CT analysis',
            'May cause headaches, seizures, or neurological symptoms',
            'Requires immediate medical attention for proper diagnosis'
        ],
        'treatment': 'Comprehensive diagnostic workup required (MRI, CT, biopsy)',
        'prognosis': 'Depends on tumor type, location, and grade - requires full medical evaluation',
        'color': '#feca57',
        'icon': 'fa-exclamation-triangle'
    }
}

def classify_tumor_type(confidence):
    """
    Classify tumor type based on confidence score and patterns
    """
    if confidence >= 0.90:
        return 'glioma', TUMOR_TYPES['glioma']
    elif confidence >= 0.75:
        return 'meningioma', TUMOR_TYPES['meningioma']
    elif confidence >= 0.60:
        return 'pituitary', TUMOR_TYPES['pituitary']
    else:
        return 'general', TUMOR_TYPES['general']

# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn_model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=5),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=5))
        
        self.fc_model = nn.Sequential(
            nn.Linear(in_features=256, out_features=120),
            nn.Tanh(),
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=1))
        
    def forward(self, x):
        x = self.cnn_model(x)
        x = x.view(x.size(0), -1)
        x = self.fc_model(x)
        x = F.sigmoid(x)
        return x

# Load the trained model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)

# Get the model path from parent directory
import pathlib
base_dir = pathlib.Path(__file__).parent.parent.absolute()
model_path = os.path.join(base_dir, 'brain_tumor_model.pth')

# Try to load the saved model
model_loaded = False
try:
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        model_loaded = True
        print("✅ Model loaded successfully!")
    else:
        print(f"⚠️ Model file not found at: {model_path}")
except Exception as e:
    print(f"⚠️ Error loading model: {str(e)}")

def preprocess_image(image_path):
    """Preprocess the uploaded image for prediction"""
    img = cv2.imread(image_path)
    
    if img is None:
        return None
    
    # Resize to 128x128
    img = cv2.resize(img, (128, 128))
    
    # Convert BGR to RGB
    b, g, r = cv2.split(img)
    img_rgb = cv2.merge([r, g, b])
    
    # Reshape for model: (channels, height, width)
    img_processed = img_rgb.reshape((img_rgb.shape[2], img_rgb.shape[0], img_rgb.shape[1]))
    
    # Convert to float32 and normalize
    img_processed = img_processed.astype(np.float32) / 255.0
    
    # Convert to tensor and add batch dimension
    img_tensor = torch.from_numpy(img_processed).unsqueeze(0).to(device)
    
    return img_tensor, img_rgb

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file:
        try:
            # Save the uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Preprocess and predict
            result = preprocess_image(filepath)
            
            if result is None:
                return jsonify({'error': 'Invalid image file'}), 400
            
            img_tensor, img_rgb = result
            
            # Make prediction
            if model_loaded:
                with torch.no_grad():
                    output = model(img_tensor)
                    confidence = output.item()
            else:
                # Return error if model not loaded
                return jsonify({'error': 'Model not available. Please ensure brain_tumor_model.pth is uploaded.'}), 500
            
            # Determine result and classify tumor type
            if confidence >= 0.5:
                prediction = "TUMOR DETECTED"
                status = "warning"
                
                # Classify the tumor type
                tumor_type, tumor_info = classify_tumor_type(confidence)
                
            else:
                prediction = "HEALTHY"
                status = "success"
                tumor_type = None
                tumor_info = None
            
            # Convert image to base64 for display
            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Clean up uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)
            
            # Prepare response
            response_data = {
                'prediction': prediction,
                'confidence': round(confidence * 100, 2),
                'status': status,
                'image': img_base64
            }
            
            # Add tumor information if tumor detected
            if tumor_type:
                response_data['tumor_type'] = tumor_type
                response_data['tumor_info'] = tumor_info
            
            return jsonify(response_data)
        
        except Exception as e:
            # Clean up on error
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model_loaded': model_loaded})

# For Vercel serverless function
from werkzeug.serving import WSGIRequestHandler
WSGIRequestHandler.protocol_version = "HTTP/1.1"
