#!/bin/bash
# deploy.sh

echo "=== Deploying Personal AI ==="

# 1. Clone atau update kode
if [ ! -d "my-personal-ai" ]; then
    git clone https://github.com/simurazx-ui/simurazx.git
    cd my-personal-ai
else
    cd my-personal-ai
    git pull
fi

# 2. Buat virtual environment
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# 3. Aktifkan virtual environment
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Buat file konfigurasi
cat > config.json << EOF
{
    "ai_name": "MyPersonalAI",
    "version": "1.0.0",
    "secret_key": "$(openssl rand -hex 32)",
    "host": "0.0.0.0",
    "port": 8000,
    "debug": false,
    "max_upload_size": 10485760
}
EOF

# 6. Jalankan aplikasi
echo "Starting AI server..."
nohup uvicorn app:app --host 0.0.0.0 --port 8000 > ai.log 2>&1 &

echo "AI server started on port 8000"
echo "Check logs: tail -f ai.log"