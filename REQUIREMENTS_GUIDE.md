# 🚀 Installation & Requirements Guide

This project has multiple `requirements.txt` files for different purposes:

## 📋 Requirements Files

### 1. `requirements_webapp.txt` (Recommended for Local Development)
```bash
pip install -r requirements_webapp.txt
```

**Contents:**
- Flask 3.0.0 - Web framework
- PyTorch 2.1.2 - Deep learning
- OpenCV 4.10.0 - Image processing
- NumPy 1.26.3 - Numerical computing
- Werkzeug 3.0.1 - WSGI utilities

**Use this for:** Running the web app locally on your machine

---

### 2. `requirements_vercel.txt` (For Vercel Deployment)
```bash
pip install -r requirements_vercel.txt
```

**Contents:**
- Minimal production dependencies only
- No platform-specific packages (pywin32, etc.)
- No development tools (Jupyter, etc.)
- Compatible with Python 3.12 on Vercel

**Use this for:** Deploying on Vercel serverless platform

---

### 3. `requirements_dev.txt` (For Full Development)
```bash
pip install -r requirements_dev.txt
```

**Contents:**
- All production dependencies
- Jupyter and Jupyter Lab
- Data analysis tools (pandas, matplotlib, seaborn)
- scikit-learn for ML utilities

**Use this for:** Full development environment with Jupyter notebooks

---

### 4. `requirements.txt` (Old - Not Recommended)
⚠️ **Do NOT use this file** - It contains:
- Platform-specific packages (`pywin32`)
- Incompatible versions for Python 3.12
- Unnecessary development dependencies
- Causes Vercel deployment failures

---

## 🎯 Which File to Use?

### Local Development (Windows/Mac/Linux)
```bash
pip install -r requirements_webapp.txt
python app.py
```

### Jupyter Notebook Development
```bash
pip install -r requirements_dev.txt
jupyter notebook MRI-Brain-Tumor-Detecor.ipynb
```

### Vercel Deployment
✅ Already configured in `vercel.json`
```bash
git push origin main  # Vercel automatically uses requirements_vercel.txt
```

### Railway/Render Deployment
```bash
pip install -r requirements_webapp.txt
```

---

## 🔧 Installation Steps

### Step 1: Clone Repository
```bash
git clone https://github.com/utkarshroy2004/Brain_Tumor_Detector.git
cd Brain_Tumor_Detector
```

### Step 2: Create Virtual Environment (Recommended)

**Windows:**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

**Option A: Minimal (Just Web App)**
```bash
pip install -r requirements_webapp.txt
```

**Option B: Full Development (With Jupyter)**
```bash
pip install -r requirements_dev.txt
```

### Step 4: Verify Installation
```bash
python -c "import torch; import flask; import cv2; print('✅ All packages installed!')"
```

### Step 5: Run Application
```bash
python app.py
```

Visit: `http://localhost:5000`

---

## ⚡ Troubleshooting Installation

### Issue: "No module named torch"

**Solution:**
```bash
pip install --upgrade pip
pip install -r requirements_webapp.txt
```

### Issue: "pywin32 not found" on Linux/Mac

**Solution:**
- The file doesn't contain pywin32
- If you have old `requirements.txt`, delete it
- Use `requirements_webapp.txt` instead

### Issue: "Incompatible Python version"

**Solution:**
```bash
python --version  # Check your Python version
# Must be Python 3.8 or higher
python3.10 -m venv venv  # Use specific Python version
```

### Issue: OpenCV fails to install

**Solution:**
```bash
# Install system dependencies first
# Ubuntu/Debian:
sudo apt-get install python3-opencv

# macOS:
brew install opencv

# Then install via pip:
pip install opencv-python==4.10.0.84
```

### Issue: PyTorch installation takes too long

**Solution:**
```bash
# PyTorch is large (~600MB). This is normal.
# Make sure you have good internet connection
# Takes 5-10 minutes depending on speed
```

---

## 📦 Package Versions

All packages are pinned to specific versions for compatibility:

| Package | Version | Purpose |
|---------|---------|---------|
| Flask | 3.0.0 | Web framework |
| PyTorch | 2.1.2 | Deep learning |
| torchvision | 0.16.2 | Image utilities |
| OpenCV | 4.10.0.84 | Image processing |
| NumPy | 1.26.3 | Numerical computing |
| Werkzeug | 3.0.1 | WSGI utilities |

---

## 🌐 Deployment Requirements

### Vercel
- Python 3.12 compatible
- Uses `requirements_vercel.txt`
- Maximum 30 second execution
- Automatically configured in `vercel.json`

### Railway
- Python 3.8+
- Use `requirements_webapp.txt`
- No restrictions on execution time
- Automatic dependency installation

### Heroku (Not available - free tier discontinued)
- Use `requirements_webapp.txt`
- Would need Procfile configuration

### Local Server
- Any Python 3.8+
- Use `requirements_webapp.txt`
- Full GPU support if available

---

## 🔍 Verify Deployment

After deployment, test the health endpoint:

```bash
curl https://YOUR_APP_URL/health
```

Expected response:
```json
{"status": "ok", "model_loaded": true}
```

---

## 📚 Additional Resources

- [Flask Documentation](https://flask.palletsprojects.com/)
- [PyTorch Installation](https://pytorch.org/get-started/locally/)
- [OpenCV Docs](https://docs.opencv.org/)
- [Vercel Python Support](https://vercel.com/docs/concepts/functions/serverless-functions/python)
- [Railway Documentation](https://docs.railway.app/)

---

## ✅ Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] Correct requirements file installed
- [ ] All imports working (`import torch`, `import flask`)
- [ ] `app.py` runs without errors
- [ ] Model file present (`brain_tumor_model.pth`)
- [ ] Browser opens to `http://localhost:5000`

---

**Ready to deploy? See [VERCEL_DEPLOYMENT.md](VERCEL_DEPLOYMENT.md) or [RAILWAY_DEPLOYMENT.md](RAILWAY_DEPLOYMENT.md)**
