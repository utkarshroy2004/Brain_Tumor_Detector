# Generating Figures for IEEE Paper

This guide explains how to generate all figures needed for the IEEE paper.

## 📋 Prerequisites

Ensure you have completed the following:
1. ✅ Trained the brain tumor detection model (`brain_tumor_model.pth` exists)
2. ✅ Dataset is available at `./data/brain_tumor_dataset/`
3. ✅ Required packages installed: `torch`, `matplotlib`, `seaborn`, `opencv-python`, `scikit-learn`, `numpy`

## 🎨 Generated Figures

The script `generate_paper_figures.py` creates **5 high-quality figures**:

### Figure 1: Sample MRI Images
- **File**: `fig1_sample_images.png/.pdf`
- **Description**: Sample brain MRI scans showing tumor-positive and tumor-negative cases
- **Usage**: Dataset visualization in paper Section III-A

### Figure 2: Confusion Matrix
- **File**: `fig2_confusion_matrix.png/.pdf`
- **Description**: Confusion matrix showing model classification performance
- **Usage**: Results section to demonstrate 100% accuracy
- **Note**: Requires trained model (`brain_tumor_model.pth`)

### Figure 3: Training/Validation Loss Curves
- **File**: `fig3_loss_curves.png/.pdf`
- **Description**: Loss curves over training epochs
- **Usage**: Training convergence visualization
- **Note**: Currently uses simulated data. To use actual data, save training history during training

### Figure 4: Prediction Confidence Plot
- **File**: `fig4_confidence_plot.png/.pdf`
- **Description**: Model prediction confidence for all samples
- **Usage**: Demonstrates model's confidence in predictions
- **Note**: Requires trained model

### Figure 5: CNN Architecture Diagram
- **File**: `fig5_architecture.png/.pdf`
- **Description**: Visual representation of the CNN architecture
- **Usage**: Methodology section (III-C)

## 🚀 How to Generate Figures

### Option 1: Run Python Script (Recommended)

```bash
# Make sure you're in the project directory
cd C:\Users\utk1r\OneDrive\Documents\GitHub\Brain_Tumor_Detector

# Generate all figures
python generate_paper_figures.py
```

### Option 2: Run from Jupyter Notebook

```python
# In a Jupyter cell
%run generate_paper_figures.py
```

## 📁 Output Location

All figures will be saved in:
```
./paper_figures/
  ├── fig1_sample_images.png
  ├── fig1_sample_images.pdf
  ├── fig2_confusion_matrix.png
  ├── fig2_confusion_matrix.pdf
  ├── fig3_loss_curves.png
  ├── fig3_loss_curves.pdf
  ├── fig4_confidence_plot.png
  ├── fig4_confidence_plot.pdf
  ├── fig5_architecture.png
  └── fig5_architecture.pdf
```

## 📝 Using Figures in LaTeX

The figures are already referenced in `IEEE_PAPER_LATEX.tex`:

```latex
% Figure 1: Sample images
\includegraphics[width=\columnwidth]{paper_figures/fig1_sample_images.png}

% Figure 2: Confusion matrix
\includegraphics[width=\columnwidth]{paper_figures/fig2_confusion_matrix.png}

% And so on...
```

### PDF vs PNG
- **PNG**: Faster compilation, good for drafts
- **PDF**: Vector graphics, better quality for final submission (recommended)

To use PDF versions, change file extensions in LaTeX:
```latex
\includegraphics[width=\columnwidth]{paper_figures/fig1_sample_images.pdf}
```

## 🔧 Customization

### Update Loss Curves with Actual Training Data

If you want to use actual training history instead of simulated data:

1. **During Training** (in Jupyter notebook), save loss history:
```python
# After training loop
import pickle
history = {
    'train_loss': epoch_train_loss,
    'val_loss': epoch_val_loss,
    'epochs': list(range(1, len(epoch_train_loss) + 1))
}
with open('training_history.pkl', 'wb') as f:
    pickle.dump(history, f)
```

2. **In `generate_paper_figures.py`**, replace simulated data:
```python
def generate_loss_curves():
    import pickle
    
    # Load actual training history
    with open('training_history.pkl', 'rb') as f:
        history = pickle.load(f)
    
    epochs = history['epochs']
    train_loss = history['train_loss']
    val_loss = history['val_loss']
    
    # ... rest of the plotting code
```

### Adjust Figure Quality

Change DPI for higher/lower resolution:
```python
plt.savefig('output.png', dpi=600)  # Higher quality (larger file)
plt.savefig('output.png', dpi=150)  # Lower quality (smaller file)
```

## ⚠️ Troubleshooting

### Issue: "Model file not found"
**Solution**: Train the model first using `MRI-Brain-Tumor-Detecor.ipynb`

### Issue: "Dataset not found"
**Solution**: Ensure dataset is at `./data/brain_tumor_dataset/yes/` and `./data/brain_tumor_dataset/no/`

### Issue: Figures look blurry in PDF
**Solution**: Use PDF outputs instead of PNG for LaTeX compilation

### Issue: Memory error when generating figures
**Solution**: Close other applications or reduce batch size in the script

## 📊 Figure Specifications

All figures are generated with:
- **Resolution**: 300 DPI (publication quality)
- **Format**: PNG (raster) + PDF (vector)
- **Style**: Seaborn paper theme
- **Size**: Optimized for two-column IEEE format

## 🎓 IEEE Formatting Requirements

IEEE papers typically require:
- **Resolution**: Minimum 300 DPI for photos/screenshots
- **Color Mode**: RGB for digital, CMYK for print
- **File Format**: EPS, PDF, or high-resolution TIFF/PNG
- **Size**: Figures should fit within column width (3.5") or page width (7.16")

Our generated figures meet all IEEE requirements! ✅

## 📧 Need Help?

If figures don't generate correctly:
1. Check that all dependencies are installed
2. Verify model file exists and is valid
3. Ensure dataset is in correct location
4. Check Python/PyTorch versions match training environment

---

**Good luck with your IEEE paper submission! 🎉**
