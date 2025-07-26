$env:GOOGLE_API_KEY = "AIzaSyDsJRnbA3GZckLE83mK2yA2bIYMmungtQA"
$env:ELEVENLABS_API_KEY = "sk_cce495b4c5d2cf5661ad1645be482965997e6f0fe258588d"

Write-Host "Starting AI Fight Coach..." -ForegroundColor Green
Write-Host "Google API Key: $env:GOOGLE_API_KEY" -ForegroundColor Yellow
Write-Host "ElevenLabs API Key: $env:ELEVENLABS_API_KEY" -ForegroundColor Yellow

python main.py 