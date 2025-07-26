# 🥊 AI Fight Coach - Quick Start

## 🚀 Start the Server (Choose ONE method)

### Method 1: Double-click launcher (Easiest)
```
Double-click: start_server.py
```

### Method 2: PowerShell script
```
Right-click run.ps1 → "Run with PowerShell"
```

### Method 3: Command line
```
python start_server.py
```

### Method 4: Manual (if above don't work)
```
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## 🌐 Use the App

1. **Open your browser** to: http://localhost:8000
2. **Upload a video** (MP4, AVI, MOV, MKV)
3. **Add a prompt** (optional)
4. **Select a voice** (optional)
5. **Click "Upload & Analyze"**
6. **Wait for processing** (shows status updates)
7. **Download the result** when done!

## 📁 What You Get

- **Original video** with AI feedback overlays
- **TTS narration** explaining the analysis
- **Highlight clips** of key moments
- **Complete analysis** with timestamps

## 🔧 If Something Goes Wrong

1. **Check the console** for error messages
2. **Make sure you're in the right directory**: `ai-fight-coach/`
3. **Try Method 1** (double-click launcher)
4. **Check logs** in `debug_logs/` folder

## 🎯 API Keys (Already Set)

- ✅ Google Gemini API: Ready
- ✅ ElevenLabs TTS API: Ready

## 📞 Need Help?

The app logs everything to `debug_logs/` - check there for details! 