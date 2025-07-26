# AI Fight Coach Deployment Guide
## Deploying to ai-boxing.com

This guide will help you deploy your AI Fight Coach application to your domain `ai-boxing.com`.

## 🎯 **Recommended: DigitalOcean VPS Deployment**

### **Step 1: Set Up DigitalOcean Droplet**

1. **Create Account**: Sign up at [DigitalOcean](https://digitalocean.com)
2. **Create Droplet**:
   - Choose Ubuntu 22.04 LTS
   - Select Basic plan ($6/month)
   - Choose datacenter close to your users
   - Add your SSH key or create password
   - Name it: `ai-boxing-server`

### **Step 2: Connect to Your Server**

```bash
# SSH into your server
ssh root@YOUR_SERVER_IP

# Update system
sudo apt update && sudo apt upgrade -y

# Install required packages
sudo apt install -y python3 python3-pip python3-venv nginx git curl
```

### **Step 3: Set Up Python Environment**

```bash
# Create project directory
mkdir -p /var/www/ai-boxing
cd /var/www/ai-boxing

# Clone your repository (if using Git)
git clone YOUR_REPOSITORY_URL .

# Or upload files via SFTP/SCP
# scp -r ai-fight-coach/* root@YOUR_SERVER_IP:/var/www/ai-boxing/

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### **Step 4: Create Requirements File**

Create `requirements.txt` in your project root:

```txt
fastapi==0.109.2
uvicorn[standard]==0.27.1
python-multipart==0.0.6
opencv-python==4.8.1.78
opencv-contrib-python==4.11.0.86
mediapipe==0.10.8
moviepy==1.0.3
google-generativeai==0.3.2
elevenlabs==0.2.27
numpy==1.24.3
Pillow==10.0.1
python-dotenv==1.0.0
```

### **Step 5: Configure Environment Variables**

Create `.env` file:

```bash
# Create .env file
nano .env
```

Add your API keys:

```env
GOOGLE_API_KEY=AIzaSyDsJRnbA3GZckLE83mK2yA2bIYMmungtQA
ELEVENLABS_API_KEY=sk_cce495b4c5d2cf5661ad1645be482965997e6f0fe258588d
```

### **Step 6: Update Email Configuration**

Edit `email_config.py`:

```python
SMTP_CONFIG = {
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "email": "your-actual-email@gmail.com",  # Your Gmail
    "password": "your-16-char-app-password"  # Gmail app password
}
```

### **Step 7: Create Systemd Service**

Create service file:

```bash
sudo nano /etc/systemd/system/ai-boxing.service
```

Add content:

```ini
[Unit]
Description=AI Boxing Coach FastAPI Application
After=network.target

[Service]
Type=exec
User=www-data
Group=www-data
WorkingDirectory=/var/www/ai-boxing
Environment=PATH=/var/www/ai-boxing/venv/bin
ExecStart=/var/www/ai-boxing/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### **Step 8: Configure Nginx**

Create Nginx configuration:

```bash
sudo nano /etc/nginx/sites-available/ai-boxing
```

Add content:

```nginx
server {
    listen 80;
    server_name ai-boxing.com www.ai-boxing.com;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header Referrer-Policy "no-referrer-when-downgrade" always;
    add_header Content-Security-Policy "default-src 'self' http: https: data: blob: 'unsafe-inline'" always;

    # File upload size limit
    client_max_body_size 100M;

    # Static files
    location /static/ {
        alias /var/www/ai-boxing/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # Main application
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }

    # Health check
    location /health {
        access_log off;
        return 200 "healthy\n";
        add_header Content-Type text/plain;
    }
}
```

Enable the site:

```bash
sudo ln -s /etc/nginx/sites-available/ai-boxing /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### **Step 9: Set Up SSL with Let's Encrypt**

```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d ai-boxing.com -d www.ai-boxing.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

### **Step 10: Start Services**

```bash
# Set permissions
sudo chown -R www-data:www-data /var/www/ai-boxing
sudo chmod -R 755 /var/www/ai-boxing

# Start application
sudo systemctl enable ai-boxing
sudo systemctl start ai-boxing

# Check status
sudo systemctl status ai-boxing
```

### **Step 11: Configure Domain DNS**

In your domain registrar's DNS settings:

```
Type: A
Name: @
Value: YOUR_SERVER_IP

Type: A
Name: www
Value: YOUR_SERVER_IP
```

---

## 🔧 **Alternative: Railway Deployment (Easier)**

### **Step 1: Prepare for Railway**

1. **Create `railway.json`**:

```json
{
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "uvicorn main:app --host 0.0.0.0 --port $PORT",
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
```

2. **Create `Procfile`**:

```
web: uvicorn main:app --host 0.0.0.0 --port $PORT
```

### **Step 2: Deploy to Railway**

1. Go to [Railway](https://railway.app)
2. Connect your GitHub repository
3. Add environment variables:
   - `GOOGLE_API_KEY`
   - `ELEVENLABS_API_KEY`
4. Deploy

### **Step 3: Connect Custom Domain**

1. In Railway dashboard, go to your project
2. Click "Settings" → "Domains"
3. Add `ai-boxing.com`
4. Update DNS records as instructed

---

## 🔧 **Alternative: Render Deployment**

### **Step 1: Prepare for Render**

Create `render.yaml`:

```yaml
services:
  - type: web
    name: ai-boxing-coach
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: GOOGLE_API_KEY
        value: AIzaSyDsJRnbA3GZckLE83mK2yA2bIYMmungtQA
      - key: ELEVENLABS_API_KEY
        value: sk_cce495b4c5d2cf5661ad1645be482965997e6f0fe258588d
```

### **Step 2: Deploy**

1. Go to [Render](https://render.com)
2. Connect your repository
3. Create new Web Service
4. Configure environment variables
5. Deploy

---

## 🛠️ **Production Optimizations**

### **1. Update main.py for Production**

```python
# Add to main.py
import os

# Production settings
if os.getenv("ENVIRONMENT") == "production":
    # Disable debug mode
    app.debug = False
    
    # Add CORS middleware
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["https://ai-boxing.com"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
```

### **2. File Storage**

For production, consider using cloud storage:

```python
# Add to requirements.txt
boto3==1.34.0  # For AWS S3
```

### **3. Database**

For user management, consider PostgreSQL:

```python
# Add to requirements.txt
sqlalchemy==2.0.23
psycopg2-binary==2.9.9
```

### **4. Monitoring**

Add health check endpoint:

```python
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}
```

---

## 🔍 **Testing Your Deployment**

1. **Check Application**: Visit `https://ai-boxing.com`
2. **Test Registration**: Register a new user
3. **Test Video Upload**: Upload a boxing video
4. **Test Email**: Check if emails are sent
5. **Monitor Logs**: `sudo journalctl -u ai-boxing -f`

---

## 🚨 **Security Checklist**

- [ ] SSL certificate installed
- [ ] Firewall configured (UFW)
- [ ] Regular security updates
- [ ] API keys secured
- [ ] File upload limits set
- [ ] Error messages don't expose sensitive info
- [ ] Database backups (if using database)

---

## 📞 **Support**

If you encounter issues:

1. Check logs: `sudo journalctl -u ai-boxing -f`
2. Check Nginx logs: `sudo tail -f /var/log/nginx/error.log`
3. Test locally first
4. Verify DNS propagation: `nslookup ai-boxing.com`

---

## 💰 **Cost Estimation**

- **DigitalOcean Droplet**: $6/month
- **Domain**: $10-15/year
- **SSL Certificate**: Free (Let's Encrypt)
- **Total**: ~$7-8/month

---

**Choose the deployment method that best fits your needs and technical expertise!** 