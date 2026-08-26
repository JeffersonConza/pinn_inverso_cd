#!/bin/bash
# ==============================================================================
# AWS EC2 UserData Bootstrap Script for PINN Inverse Drag API (Lightweight & Robust)
# Repositorio: https://github.com/JeffersonConza/pinn_inverso_cd
# Compatible con: Ubuntu 22.04/24.04 LTS & Amazon Linux 2023 (t3.micro / t3.small / t3.medium)
# ==============================================================================

set -e
exec > >(tee -a /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1

echo "=========================================================="
echo "🚀 Iniciando despliegue automatizado: PINN Inverse Drag API"
echo "=========================================================="

# 1. Crear 2 GB de memoria SWAP para evitar OOM (Out-of-Memory) en instancias pequeñas
if [ ! -f /swapfile ]; then
    echo "💾 Configurando 2GB de memoria Swap..."
    fallocate -l 2G /swapfile || dd if=/dev/zero of=/swapfile bs=1M count=2048
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile
    echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

# 2. Configuración de repositorio y variables
REPO_URL="https://github.com/JeffersonConza/pinn_inverso_cd.git"
S3_BUCKET_NAME="${S3_BUCKET:-pinns-models-repository}"

# 3. Detectar sistema operativo e instalar dependencias base
if command -v apt-get &> /dev/null; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -y
    apt-get install -y python3-pip python3-venv git curl
    USER_NAME="ubuntu"
    APP_DIR="/home/ubuntu/pinn_inverso_cd"
else
    dnf update -y
    dnf install -y python3-pip git curl
    USER_NAME="ec2-user"
    APP_DIR="/home/ec2-user/pinn_inverso_cd"
fi

# 4. Clonar el repositorio del proyecto
echo "📥 Clonando repositorio desde ${REPO_URL}..."
mkdir -p $(dirname "$APP_DIR")
if [ -d "$APP_DIR" ]; then
    rm -rf "$APP_DIR"
fi
git clone "$REPO_URL" "$APP_DIR"
cd "$APP_DIR"

# 5. Crear entorno virtual Python e instalar PyTorch CPU optimizado
echo "📦 Configurando entorno virtual Python e instalando dependencias..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# PyTorch versión CPU (~180MB en lugar de ~900MB con CUDA)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install fastapi uvicorn boto3 python-dotenv numpy scipy matplotlib pillow pydantic requests

# 6. Generar archivo .env
cat <<EOF > "$APP_DIR/.env"
PORT=8000
HOST=0.0.0.0
S3_BUCKET=$S3_BUCKET_NAME
S3_MODEL_KEY=models/inverse_drag_pinn.pt
MODEL_PATH=models/inverse_drag_pinn.pt
EOF

# Ajustar permisos
chown -R $USER_NAME:$USER_NAME "$APP_DIR"

# 7. Configurar servicio systemd para FastAPI
echo "⚙️ Configurando servicio systemd para FastAPI..."
cat <<EOF > /etc/systemd/system/pinn-inverse-api.service
[Unit]
Description=FastAPI Service for PINN Inverse Drag Coefficient Discovery
After=network.target

[Service]
User=$USER_NAME
WorkingDirectory=$APP_DIR
Environment="PATH=$APP_DIR/venv/bin:/usr/local/bin:/usr/bin"
EnvironmentFile=$APP_DIR/.env
ExecStart=$APP_DIR/venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# 8. Iniciar y habilitar el servicio
systemctl daemon-reload
systemctl enable pinn-inverse-api
systemctl restart pinn-inverse-api

echo "=========================================================="
echo "✅ Despliegue completado con éxito. PINN API activa en :8000"
echo "🌐 Visualizador Demo disponible en: http://$(curl -s http://checkip.amazonaws.com || echo '<Public-IP>'):8000/demo"
echo "=========================================================="
