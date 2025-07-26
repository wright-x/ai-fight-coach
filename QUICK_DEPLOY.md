# 🚀 Quick Deployment Guide for ai-boxing.com

## **Easiest Option: Railway (Recommended for beginners)**

### **Step 1: Prepare Your Code**
1. Make sure all files are in the `ai-fight-coach` folder
2. Update `email_config.py` with your Gmail credentials
3. Commit all changes to Git

### **Step 2: Deploy to Railway**
1. Go to [Railway.app](https://railway.app)
2. Sign up with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select your repository
5. Railway will automatically detect it's a Python app

### **Step 3: Configure Environment Variables**
In Railway dashboard, go to your project → Variables tab and add:

```
GOOGLE_API_KEY=AIzaSyDsJRnbA3GZckLE83mK2yA2bIYMmungtQA
ELEVENLABS_API_KEY=sk_cce495b4c5d2cf5661ad1645be482965997e6f0fe258588d
```

### **Step 4: Connect Custom Domain**
1. In Railway dashboard → Settings → Domains
2. Add `ai-boxing.com`
3. Railway will give you DNS records to add to your domain registrar

### **Step 5: Update DNS**
In your domain registrar (where you bought ai-boxing.com), add these DNS records:

```
Type: CNAME
Name: @
Value: [Railway-provided URL]

Type: CNAME  
Name: www
Value: [Railway-provided URL]
```

### **Step 6: Test**
1. Wait 5-10 minutes for DNS propagation
2. Visit `https://ai-boxing.com`
3. Test registration and video upload

---

## **Alternative: Render (Also Easy)**

### **Step 1: Deploy to Render**
1. Go to [Render.com](https://render.com)
2. Sign up with GitHub
3. Click "New" → "Web Service"
4. Connect your repository
5. Render will auto-detect Python

### **Step 2: Configure**
- **Name**: `ai-boxing-coach`
- **Environment**: `Python 3`
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`

### **Step 3: Add Environment Variables**
Same as Railway above.

### **Step 4: Connect Domain**
1. Go to Settings → Custom Domains
2. Add `ai-boxing.com`
3. Update DNS records as instructed

---

## **Cost Comparison**

| Platform | Cost | Ease | Features |
|----------|------|------|----------|
| Railway | $5-20/month | ⭐⭐⭐⭐⭐ | Auto-scaling, easy setup |
| Render | $7-25/month | ⭐⭐⭐⭐ | Good free tier, easy setup |
| DigitalOcean | $6/month | ⭐⭐⭐ | Full control, more setup |

---

## **Quick Checklist**

- [ ] Update email credentials in `email_config.py`
- [ ] Test locally first
- [ ] Deploy to chosen platform
- [ ] Add environment variables
- [ ] Connect custom domain
- [ ] Test all features
- [ ] Monitor logs for errors

---

## **Troubleshooting**

### **Common Issues:**
1. **Build fails**: Check `requirements.txt` is complete
2. **Email not working**: Verify Gmail app password
3. **Domain not working**: Wait for DNS propagation (up to 48 hours)
4. **Video upload fails**: Check file size limits

### **Get Help:**
- Check platform logs in dashboard
- Verify all environment variables are set
- Test with a simple video first

---

**Railway is recommended for the easiest deployment experience!** 🚀 