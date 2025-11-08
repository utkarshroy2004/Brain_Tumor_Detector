# Vercel Deployment Guide

This guide explains how to deploy the Brain Tumor Detection system to Vercel.

## Important Note: Model Size Limitation

⚠️ **The trained model file (`brain_tumor_model.pth` ~1.2 MB) exceeds Vercel's requirements for free tier.**

Vercel has the following constraints:
- **Maximum deployment size**: 100 MB
- **Maximum function duration**: 30 seconds
- **Maximum function memory**: 3008 MB (3 GB)
- **Output file size**: 20 MB max

### Solution: Use an External Model Storage

For production deployment with Vercel, you need to:

1. **Upload model to a cloud storage service:**
   - AWS S3
   - Google Cloud Storage
   - Azure Blob Storage
   - Hugging Face Model Hub
   - GitHub Releases

2. **Download model at runtime** from the cloud storage

## Step-by-Step Deployment

### 1. Prerequisites

- Vercel Account (https://vercel.com)
- Git repository (this project)
- Model uploaded to cloud storage

### 2. Upload Model to GitHub Releases (Easiest)

1. Go to your GitHub repository
2. Navigate to **Releases** → **Create a new release**
3. Upload `brain_tumor_model.pth`
4. Create the release

### 3. Update Project for Vercel

The project already includes:
- `vercel.json` - Configuration file
- `api/index.py` - Serverless function entry point
- `requirements.txt` - Python dependencies

### 4. Deploy to Vercel

**Option A: Using Vercel CLI**

```bash
npm install -g vercel
vercel
```

**Option B: Using GitHub Integration**

1. Go to https://vercel.com/new
2. Import from GitHub
3. Select your repository
4. Vercel automatically detects Flask app
5. Deploy!

### 5. Environment Variables

Add these in Vercel Dashboard → Settings → Environment Variables:

```
MODEL_URL=https://github.com/YOUR_USERNAME/Brain_Tumor_Detector/releases/download/v1.0/brain_tumor_model.pth
```

## Current Limitations

### What Won't Work on Free Tier:

1. **Large Model Files** - Model exceeds limits
   - Solution: Host model separately (S3, etc.)

2. **Static Files** - Templates folder may not serve correctly
   - Solution: Use CDN or serve as API responses

3. **GPU Support** - Not available on free tier
   - Will use CPU (slower inference)

4. **File Uploads** - `/tmp` has limited storage
   - Max temp storage: ~512 MB

### Recommended Alternatives:

1. **Heroku** - Better for traditional Flask apps
2. **Railway** - Good free tier for Python
3. **Render** - Affordable Python deployment
4. **AWS Lambda** - Scalable but complex setup
5. **Google Cloud Run** - Pay-as-you-go model

## Troubleshooting

### 404 Error

**Cause**: Routes not configured correctly

**Solution**: 
- Ensure `vercel.json` is in root directory
- Check that `api/index.py` is the entry point
- Routes should map to `api/index.py`

### Model Not Found

**Cause**: `brain_tumor_model.pth` not uploaded

**Solution**:
- Download model from cloud storage at runtime
- Add model download logic to `api/index.py`

### Memory Exceeded

**Cause**: Model too large for function

**Solution**:
- Quantize model to reduce size
- Use model compression techniques
- Upgrade to paid Vercel plan

## Alternative: Deploy on Your Own Server

For best results, deploy on:
- **Traditional Server**: Linode, DigitalOcean, AWS EC2
- **Docker**: Run in container on any platform
- **Local Network**: ngrok for testing

## Cost Analysis

| Platform | Free Tier | Storage | Model Support |
|----------|-----------|---------|---------------|
| Vercel | Yes | 20 MB | ✅ (external) |
| Heroku | No | 512 MB | ✅ |
| Railway | Yes | Good | ✅ |
| Render | Yes | 500 MB | ✅ |
| AWS Lambda | Free tier | Limited | ⚠️ (complex) |

## Next Steps

1. Choose a deployment platform
2. Configure model storage
3. Update deployment configuration
4. Test inference on platform
5. Set up monitoring and logging

For more details, see: https://vercel.com/docs/concepts/functions/serverless-functions
