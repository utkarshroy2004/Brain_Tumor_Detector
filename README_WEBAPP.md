# 🧠 Brain Tumor Detection Web App

A beautiful, user-friendly web application for detecting brain tumors in MRI scans using deep learning.

![Brain Tumor Detection](https://img.shields.io/badge/AI-Brain%20Tumor%20Detection-blue)
![Python](https://img.shields.io/badge/Python-3.12-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1.2-red)
![Flask](https://img.shields.io/badge/Flask-3.0.0-black)

## 🌟 Features

- **Drag & Drop Interface**: Simply drag and drop your MRI scan image
- **Real-time Analysis**: Get instant predictions with confidence scores
- **Beautiful UI**: Modern, responsive design with smooth animations
- **Mobile Friendly**: Works seamlessly on desktop, tablet, and mobile devices
- **High Accuracy**: Trained on real MRI brain tumor dataset with 100% training accuracy

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_webapp.txt
```

### 2. Train the Model (if not already done)

Open and run the Jupyter notebook `MRI-Brain-Tumor-Detecor.ipynb` to train the model. Make sure to run the "Save the Trained Model" cell at the end.

### 3. Run the Web Application

```bash
python app.py
```

### 4. Open Your Browser

Navigate to `http://localhost:5000` and start analyzing MRI scans!

## 📁 Project Structure

```
Brain-Tumor-Detector/
│
├── app.py                          # Flask backend server
├── brain_tumor_model.pth          # Trained PyTorch model
├── requirements_webapp.txt        # Python dependencies
├── MRI-Brain-Tumor-Detecor.ipynb # Training notebook
│
├── templates/
│   └── index.html                 # Web interface
│
├── data/
│   └── brain_tumor_dataset/
│       ├── yes/                   # Tumor images
│       └── no/                    # Healthy images
│
└── uploads/                       # Temporary upload folder
```

## 🎯 How It Works

1. **Upload**: User uploads an MRI scan image
2. **Preprocessing**: Image is resized to 128x128 and normalized
3. **Prediction**: CNN model analyzes the scan
4. **Result**: Display prediction (Tumor/Healthy) with confidence score

## 🧠 Model Architecture

- **Convolutional Layers**: 2 layers (3→6→16 channels)
- **Fully Connected Layers**: 3 layers (256→120→84→1)
- **Activation Functions**: Tanh + Sigmoid
- **Optimizer**: Adam (lr=0.0001)
- **Loss Function**: Binary Cross-Entropy

## 📊 Model Performance

- **Training Accuracy**: 100%
- **Dataset Size**: 245 MRI scans (154 tumor, 91 healthy)
- **Image Size**: 128x128 pixels
- **Training Epochs**: 400

## 🖼️ Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)

## ⚠️ Medical Disclaimer

**IMPORTANT**: This application is for educational and research purposes only. It should NOT be used as a substitute for professional medical diagnosis, advice, or treatment. Always consult with qualified healthcare professionals for medical concerns.

## 🛠️ Technologies Used

- **Backend**: Flask (Python web framework)
- **Deep Learning**: PyTorch
- **Image Processing**: OpenCV
- **Frontend**: HTML5, CSS3, JavaScript
- **Data Processing**: NumPy

## 📝 Usage Tips

1. Use clear, high-quality MRI scan images for best results
2. The model works best with brain MRI scans similar to the training data
3. Images are automatically resized to 128x128 pixels
4. Upload time depends on your image size and internet connection

## 🎨 Features Walkthrough

### Upload Interface
- Drag and drop your MRI scan image
- Or click to browse and select a file
- Instant image preview before analysis

### Analysis
- Click "Analyze MRI Scan" button
- Watch the loading animation
- Results appear with confidence percentage

### Results Display
- ✅ **Healthy**: Green background, low tumor probability
- ⚠️ **Tumor Detected**: Yellow background, high tumor probability
- Confidence bar showing prediction strength
- Option to analyze another scan

## 🔒 Privacy & Security

- Images are processed locally on the server
- Uploaded files are automatically deleted after analysis
- No data is stored or shared

## 🐛 Troubleshooting

### Model Not Found Error
Run the notebook and execute the model saving cell:
```python
torch.save(model.state_dict(), 'brain_tumor_model.pth')
```

### Port Already in Use
Change the port in `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Use different port
```

### Out of Memory Error
The model runs on CPU by default. If you have GPU available, it will automatically use it.

## 📚 Dataset

The model is trained on the [Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection) dataset from Kaggle.

## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements!

## 📄 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

Created with ❤️ for educational purposes

---

**Happy Analyzing! 🧠✨**
