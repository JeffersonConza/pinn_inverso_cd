# api/main.py
from contextlib import asynccontextmanager
import os
import time
from typing import Any, AsyncGenerator, Dict, List, Optional
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
import torch
import torch.optim as optim
import numpy as np

from src.config import Config
from src.physics_engine import PhysicsEngine
from src.pinn_inverse import (
    InverseDragPINN,
    compute_inverse_derivatives,
    compute_ode_residual,
)

S3_BUCKET: Optional[str] = os.getenv("S3_BUCKET")
S3_MODEL_KEY: str = os.getenv("S3_MODEL_KEY", "models/inverse_drag_pinn.pt")
MODEL_PATH: str = os.getenv("MODEL_PATH", Config.MODEL_PATH)
DEVICE: torch.device = torch.device("cpu")


def _init_local_model() -> InverseDragPINN:
    model = InverseDragPINN(initial_cd_guess=Config.CD_INITIAL_GUESS).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            model.eval()
        except Exception:
            pass
    return model


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Gestiona el ciclo de vida: descarga pesos desde S3 y los carga en la memoria RAM de EC2."""
    # 1. Sincronización desde S3 (si está configurado)
    if S3_BUCKET:
        try:
            import boto3
            verify_ssl = os.getenv("AWS_VERIFY_SSL", "true").lower() != "false"
            if not verify_ssl:
                import urllib3
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            s3 = boto3.client("s3", verify=verify_ssl)
            os.makedirs(os.path.dirname(MODEL_PATH) or ".", exist_ok=True)
            s3.download_file(S3_BUCKET, S3_MODEL_KEY, MODEL_PATH)
            print(f"✅ Modelo descargado desde s3://{S3_BUCKET}/{S3_MODEL_KEY}")
        except Exception as s3_err:
            print(f"⚠️ Nota de conexión S3 (usando fallback local): {s3_err}")

    # 2. Cargar modelo en memoria RAM
    model = _init_local_model()
    app.state.model = model
    app.state.cd_discovered = model.cd.item()
    print(f"✅ Modelo PINN inverso listo en RAM (Cd = {app.state.cd_discovered:.4f}).")

    yield


app = FastAPI(
    title="PINN Inverse Drag Discovery API",
    description="API de descubrimiento inverso del coeficiente de arrastre Cd en tiempo real para el AWS Community Day.",
    version="2.0.0",
    lifespan=lifespan,
)

# Inicialización predeterminada de app.state (compatible con imports y TestClient)
_default_model = _init_local_model()
app.state.model = _default_model
app.state.cd_discovered = _default_model.cd.item()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Esquemas Pydantic ---
class SensorPoint(BaseModel):
    t: float = Field(..., ge=0.0, description="Tiempo en segundos")
    y: float = Field(..., ge=0.0, description="Altitud medida por sensor (m)")


class DiscoverRequest(BaseModel):
    points: Optional[List[SensorPoint]] = Field(
        None, description="Puntos (t_i, y_i) opcionales. Si se omite, se generan automáticamente."
    )
    preset: Optional[str] = Field("smooth_sphere", description="Preset físico: smooth_sphere, baseball, cylinder, skydiver")
    noise_std: Optional[float] = Field(0.05, ge=0.0, le=2.0, description="Desviación de ruido del sensor")
    n_points: Optional[int] = Field(25, ge=5, le=100, description="Cantidad de observaciones")
    initial_cd_guess: Optional[float] = Field(0.10, ge=0.01, le=5.0, description="Suposición inicial de Cd")
    epochs: Optional[int] = Field(120, ge=30, le=500, description="Épocas de calibración rápida en CPU")


class DiscoverResponse(BaseModel):
    cd_estimated: float
    cd_true: Optional[float]
    relative_error_pct: Optional[float]
    initial_cd_guess: float
    latency_ms: float
    epochs: int
    points_used: int
    sensor_points: List[Dict[str, float]]
    trajectory_fine: List[Dict[str, float]]
    status: str


class PredictResponse(BaseModel):
    t: float
    altitude_y: float
    velocity_v: float
    cd_calibrated: float
    status: str = "computed_via_pinn"


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError) -> JSONResponse:
    detalles = []
    for err in exc.errors():
        field = " -> ".join([str(loc) for loc in err["loc"] if loc != "body"])
        detalles.append({"parametro": field, "error": err["msg"]})
    return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content={"detalles": detalles})


# --- Endpoints REST ---

@app.get("/", tags=["Información"])
def root() -> Dict[str, Any]:
    return {
        "title": "PINN Inverse Drag Discovery API",
        "author": "Jefferson Conza - AWS Community Day 2026",
        "description": "Descubrimiento de parámetros físicos inversos mediante Physics-Informed Neural Networks en AWS EC2.",
        "active_cd": getattr(app.state, "cd_discovered", Config.CD_TRUE),
        "docs_url": "/docs",
        "demo_url": "/demo",
    }


@app.get("/health", tags=["Salud"])
def health_check() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "model_loaded": app.state.model is not None,
        "active_cd": getattr(app.state, "cd_discovered", None),
        "device": str(DEVICE),
        "timestamp": time.time(),
    }


@app.get("/presets", tags=["Física"])
def get_presets() -> Dict[str, Any]:
    """Retorna los objetos físicos preconfigurados para la demostración."""
    return Config.PRESETS


@app.get("/predict", response_model=PredictResponse, tags=["Inferencia"])
def predict_single_point(
    t: float = Query(..., ge=Config.T_MIN, le=Config.T_MAX, description="Instante temporal en segundos")
):
    """Evalúa la trayectoria continua predicha por la PINN y su velocidad mediante autograd."""
    if app.state.model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible en memoria.")

    t_tensor = torch.tensor([[t]], dtype=torch.float32, device=DEVICE).requires_grad_(True)
    y_pred, dy_dt, _ = compute_inverse_derivatives(app.state.model, t_tensor)

    return PredictResponse(
        t=t,
        altitude_y=round(float(y_pred.item()), 4),
        velocity_v=round(float(dy_dt.item()), 4),
        cd_calibrated=round(float(app.state.model.cd.item()), 4),
    )


@app.post("/discover_cd", response_model=DiscoverResponse, tags=["Problema Inverso"])
def discover_drag_coefficient(req: DiscoverRequest) -> DiscoverResponse:
    """Ejecuta una calibración inversa ultrarrápida en CPU (< 400ms) descubriendo Cd a partir de observaciones."""
    start_time = time.perf_counter()

    # 1. Obtener parámetros físicos según preset
    preset_info = Config.PRESETS.get(req.preset, Config.PRESETS["smooth_sphere"])
    cd_true = preset_info["cd_true"]
    mass = preset_info["mass"]
    area = preset_info["area"]
    gravity = Config.GRAVITY
    rho = Config.AIR_DENSITY

    # 2. Preparar puntos de sensor (dados o generados)
    if req.points and len(req.points) >= 5:
        t_arr = np.array([p.t for p in req.points], dtype=np.float32)
        y_arr = np.array([p.y for p in req.points], dtype=np.float32)
    else:
        synthetic = PhysicsEngine.generate_synthetic_sensor_data(
            n_points=req.n_points,
            noise_std=req.noise_std,
            cd_true=cd_true,
            mass=mass,
            gravity=gravity,
            rho=rho,
            area=area,
            t_min=Config.T_MIN,
            t_max=Config.T_MAX,
        )
        t_arr = synthetic["t"].astype(np.float32)
        y_arr = synthetic["y_measured"].astype(np.float32)

    # 3. Tensores PyTorch para calibración rápida
    t_sensor = torch.tensor(t_arr, dtype=torch.float32).view(-1, 1).to(DEVICE)
    y_sensor = torch.tensor(y_arr, dtype=torch.float32).view(-1, 1).to(DEVICE)

    t_max_sample = float(t_arr.max())
    t_col = torch.linspace(0.0, t_max_sample, 25).view(-1, 1).to(DEVICE)

    # Tensor de condición inicial
    t_ic = torch.tensor([[0.0]], dtype=torch.float32).to(DEVICE)
    y_ic_target = torch.tensor([[Config.INITIAL_HEIGHT]], dtype=torch.float32).to(DEVICE)

    # 4. Inicializar modelo ágil de calibración
    calib_model = InverseDragPINN(
        initial_cd_guess=req.initial_cd_guess,
        y0=Config.INITIAL_HEIGHT,
        v0=Config.INITIAL_VELOCITY,
        gravity=gravity,
    ).to(DEVICE)
    optimizer = optim.Adam([
        {"params": calib_model.layers.parameters(), "lr": 1e-2},
        {"params": [calib_model.raw_cd], "lr": 5e-2},
    ])

    # 5. Bucle de calibración acelerado en CPU (< 400 ms)
    for _ in range(req.epochs):
        optimizer.zero_grad()

        # Ajuste a datos de sensor
        y_pred = calib_model(t_sensor)
        loss_data = torch.mean((y_pred - y_sensor) ** 2)

        # Cumplimiento de la ODE
        ode_res = compute_ode_residual(
            model=calib_model,
            t_collocation=t_col,
            mass=mass,
            gravity=gravity,
            rho=rho,
            area=area,
        )
        loss_ode = torch.mean(ode_res ** 2)

        loss = loss_data + 20.0 * loss_ode
        loss.backward()
        optimizer.step()

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    cd_discovered = float(calib_model.cd.item())
    rel_error = abs(cd_discovered - cd_true) / cd_true * 100.0 if cd_true else None

    # Generar curva fina para renderizado
    t_fine = np.linspace(0.0, t_max_sample, 120, dtype=np.float32)
    with torch.no_grad():
        t_fine_tensor = torch.tensor(t_fine).view(-1, 1).to(DEVICE)
        y_fine_pred = calib_model(t_fine_tensor).numpy().flatten()

    sensor_points_list = [{"t": float(t_i), "y": float(y_i)} for t_i, y_i in zip(t_arr, y_arr)]
    trajectory_fine_list = [{"t": float(t_i), "y": round(float(y_i), 3)} for t_i, y_i in zip(t_fine, y_fine_pred)]

    return DiscoverResponse(
        cd_estimated=round(cd_discovered, 4),
        cd_true=round(cd_true, 4) if cd_true else None,
        relative_error_pct=round(rel_error, 2) if rel_error is not None else None,
        initial_cd_guess=req.initial_cd_guess,
        latency_ms=round(elapsed_ms, 2),
        epochs=req.epochs,
        points_used=len(t_arr),
        sensor_points=sensor_points_list,
        trajectory_fine=trajectory_fine_list,
        status="success",
    )


@app.get("/demo", response_class=HTMLResponse, tags=["Visualizador Web"])
def get_interactive_demo():
    """Visualizador web interactivo moderno en tiempo real para el AWS Community Day."""
    html_content = """<!DOCTYPE html>
<html lang="es" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PINN: Descubrimiento Inverso de Coeficiente de Arrastre | AWS Community Day</title>
    <!-- Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <!-- FontAwesome -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        @media print {
            body {
                background-color: #0b0f19 !important;
                color: #f1f5f9 !important;
                -webkit-print-color-adjust: exact !important;
                print-color-adjust: exact !important;
            }
            header {
                position: static !important;
                border-bottom: 1px solid #334155 !important;
            }
            .grid {
                break-inside: avoid;
            }
            #trajectoryChart {
                max-height: 280px !important;
            }
        }
    </style>
    <script>
        tailwind.config = {
            darkMode: 'class',
            theme: {
                extend: {
                    colors: {
                        aws: {
                            orange: '#FF9900',
                            squid: '#232F3E',
                            dark: '#0F172A',
                            card: '#1E293B',
                            accent: '#00A4E4',
                            success: '#10B981',
                        }
                    }
                }
            }
        }
    </script>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen font-sans antialiased">
    <!-- Header -->
    <header class="bg-slate-900 border-b border-slate-800 sticky top-0 z-50 shadow-lg">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3.5 flex items-center justify-between">
            <div class="flex items-center space-x-3">
                <div class="bg-gradient-to-tr from-amber-500 to-orange-600 p-2 rounded-xl text-slate-950 font-black shadow-md">
                    <i class="fa-solid fa-atom text-xl"></i>
                </div>
                <div>
                    <h1 class="text-lg font-bold text-white flex items-center gap-2">
                        PINN Inverso: Descubrimiento de C<sub>d</sub>
                        <span class="text-xs px-2.5 py-0.5 rounded-full bg-amber-500/10 text-amber-400 border border-amber-500/20 font-medium">AWS Community Day 2026</span>
                    </h1>
                    <p class="text-xs text-slate-400">Jefferson Conza · Universidad Yachay Tech & AWS Cloud Institute</p>
                </div>
            </div>
            <div class="flex items-center space-x-3">
                <button onclick="toggleQRModal()" class="px-3 py-1.5 rounded-lg bg-gradient-to-r from-amber-500 to-orange-500 hover:from-amber-400 hover:to-orange-400 text-slate-950 font-bold text-xs transition-all flex items-center gap-1.5 shadow-md shadow-amber-500/20">
                    <i class="fa-solid fa-qrcode"></i> <span>Código QR</span>
                </button>
                <span id="telemetryBadge" class="hidden sm:inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                    <i class="fa-solid fa-bolt mr-1.5 animate-pulse"></i> Latencia EC2: <span id="latencyVal" class="ml-1 font-mono">-- ms</span>
                </span>
                <a href="/docs" target="_blank" class="text-xs bg-slate-800 hover:bg-slate-700 text-slate-200 px-3 py-1.5 rounded-lg border border-slate-700 transition flex items-center gap-1.5 font-medium">
                    <i class="fa-solid fa-book-open"></i> Swagger API
                </a>
            </div>
        </div>
    </header>

    <!-- Main Container -->
    <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
        
        <!-- Top Metrics & Physical Formula Banner -->
        <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div class="bg-slate-900/90 border border-slate-800 p-4 rounded-2xl shadow-sm">
                <div class="flex items-center justify-between text-slate-400 mb-1">
                    <span class="text-xs font-semibold uppercase tracking-wider">Ley Física de Newton</span>
                    <i class="fa-solid fa-wind text-sky-400"></i>
                </div>
                <div class="text-sm font-mono text-slate-200 font-bold bg-slate-950/60 p-2.5 rounded-xl border border-slate-800/80">
                    m·ÿ = -m·g - ½·ρ·A·<span class="text-amber-400 font-black">C<sub>d</sub></span>·ẏ|ẏ|
                </div>
                <p class="text-xs text-slate-400 mt-2">La PINN calcula ẏ y ÿ con <code>autograd</code> y descubre <strong class="text-amber-400">C<sub>d</sub></strong> en vivo.</p>
            </div>

            <div class="bg-slate-900/90 border border-slate-800 p-4 rounded-2xl shadow-sm">
                <div class="flex items-center justify-between text-slate-400 mb-1">
                    <span class="text-xs font-semibold uppercase tracking-wider">Función de Pérdida Compuesta</span>
                    <i class="fa-solid fa-chart-line text-emerald-400"></i>
                </div>
                <div class="text-sm font-mono text-slate-200 font-bold bg-slate-950/60 p-2.5 rounded-xl border border-slate-800/80">
                    L = L<sub>data</sub>(Sensor) + λ·L<sub>ODE</sub>(Física)
                </div>
                <p class="text-xs text-slate-400 mt-2">Equilibrio entre datos experimentales y principios de conservación.</p>
            </div>

            <div class="bg-slate-900/90 border border-slate-800 p-4 rounded-2xl shadow-sm">
                <div class="flex items-center justify-between text-slate-400 mb-1">
                    <span class="text-xs font-semibold uppercase tracking-wider">Ventaja vs Métodos Tradicionales</span>
                    <i class="fa-solid fa-microchip text-purple-400"></i>
                </div>
                <div class="text-2xl font-black text-purple-400 mt-1 font-mono">&lt; 300 ms</div>
                <p class="text-xs text-slate-400 mt-1">Calibración directa en CPU sin mallas ni solvers no lineales costosos.</p>
            </div>
        </div>

        <!-- Interactive Workbench Layout -->
        <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">
            
            <!-- Controls Sidebar (4 cols) -->
            <div class="lg:col-span-4 bg-slate-900/90 border border-slate-800 rounded-2xl p-5 space-y-5 shadow-sm">
                <div class="border-b border-slate-800 pb-3 flex items-center justify-between">
                    <h2 class="font-bold text-base text-white flex items-center gap-2">
                        <i class="fa-solid fa-sliders text-amber-500"></i> Panel de Experimentación
                    </h2>
                </div>

                <!-- Preset Selection -->
                <div class="space-y-1.5">
                    <label class="text-xs font-bold uppercase tracking-wider text-slate-300">1. Objeto / Caso de Prueba</label>
                    <select id="presetSelect" onchange="updatePresetInfo()" class="w-full bg-slate-950 border border-slate-700 rounded-xl px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-amber-500 transition">
                        <option value="smooth_sphere">🏀 Esfera Lisa (Cd Real = 0.47)</option>
                        <option value="baseball">⚾ Pelota de Béisbol (Cd Real = 0.30)</option>
                        <option value="cylinder">🚀 Cilindro / Cohete (Cd Real = 0.82)</option>
                        <option value="skydiver">🪂 Paracaidista (Cd Real = 1.20)</option>
                    </select>
                    <p id="presetDesc" class="text-xs text-slate-400 italic">Flujo laminar alrededor de un cuerpo esférico simétrico.</p>
                </div>

                <!-- Initial Guess Slider -->
                <div class="space-y-1.5">
                    <div class="flex justify-between text-xs">
                        <label class="font-bold uppercase tracking-wider text-slate-300">2. Suposición Inicial (C<sub>d</sub> Guess)</label>
                        <span id="guessDisplay" class="font-mono text-amber-400 font-bold">0.10</span>
                    </div>
                    <input type="range" id="guessSlider" min="0.05" max="2.0" step="0.05" value="0.10" oninput="document.getElementById('guessDisplay').innerText = parseFloat(this.value).toFixed(2)" class="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-amber-500">
                    <span class="text-[10px] text-slate-500">Punto de partida ciego para retar a la red neuronal.</span>
                </div>

                <!-- Sensor Noise Slider -->
                <div class="space-y-1.5">
                    <div class="flex justify-between text-xs">
                        <label class="font-bold uppercase tracking-wider text-slate-300">3. Ruido de Radar (σ Gaussiano)</label>
                        <span id="noiseDisplay" class="font-mono text-rose-400 font-bold">0.05 m</span>
                    </div>
                    <input type="range" id="noiseSlider" min="0.00" max="0.50" step="0.01" value="0.05" oninput="document.getElementById('noiseDisplay').innerText = parseFloat(this.value).toFixed(2) + ' m'" class="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-rose-500">
                    <span class="text-[10px] text-slate-500">Simula perturbaciones e imperfecciones en la telemetría.</span>
                </div>

                <!-- Sensor Points Slider -->
                <div class="space-y-1.5">
                    <div class="flex justify-between text-xs">
                        <label class="font-bold uppercase tracking-wider text-slate-300">4. Muestras de Sensor (N)</label>
                        <span id="pointsDisplay" class="font-mono text-sky-400 font-bold">25 puntos</span>
                    </div>
                    <input type="range" id="pointsSlider" min="10" max="60" step="5" value="25" oninput="document.getElementById('pointsDisplay').innerText = this.value + ' puntos'" class="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-sky-500">
                </div>

                <!-- Run Button -->
                <button id="runBtn" onclick="runDiscovery()" class="w-full bg-gradient-to-r from-amber-500 to-orange-600 hover:from-amber-400 hover:to-orange-500 text-slate-950 font-black py-3 px-4 rounded-xl shadow-lg hover:shadow-orange-500/20 active:scale-[0.98] transition flex items-center justify-center gap-2 text-sm">
                    <i class="fa-solid fa-play"></i> Descubrir C<sub>d</sub> en Vivo
                </button>
            </div>

            <!-- Results & Charts (8 cols) -->
            <div class="lg:col-span-8 space-y-6">
                
                <!-- Results Status Cards -->
                <div class="grid grid-cols-1 sm:grid-cols-3 gap-4">
                    <div class="bg-slate-900/90 border border-slate-800 p-4 rounded-2xl">
                        <div class="text-xs text-slate-400 font-medium">Ground Truth (Real)</div>
                        <div id="resCdTrue" class="text-2xl font-black text-sky-400 mt-1 font-mono">0.4700</div>
                        <div class="text-[10px] text-slate-500">Valor físico exacto</div>
                    </div>

                    <div class="bg-slate-900/90 border border-slate-800 p-4 rounded-2xl ring-1 ring-amber-500/30">
                        <div class="text-xs text-amber-400 font-bold">C<sub>d</sub> Descubierto (PINN)</div>
                        <div id="resCdEst" class="text-2xl font-black text-amber-400 mt-1 font-mono">0.4712</div>
                        <div id="resError" class="text-[10px] text-emerald-400 font-bold">Error Relativo: 0.25%</div>
                    </div>

                    <div class="bg-slate-900/90 border border-slate-800 p-4 rounded-2xl">
                        <div class="text-xs text-slate-400 font-medium">Tiempo de Calibración</div>
                        <div id="resLatency" class="text-2xl font-black text-purple-400 mt-1 font-mono">185 ms</div>
                        <div class="text-[10px] text-slate-500">120 épocas en CPU EC2</div>
                    </div>
                </div>

                <!-- Main Chart -->
                <div class="bg-slate-900/90 border border-slate-800 p-5 rounded-2xl shadow-sm">
                    <div class="flex items-center justify-between mb-4">
                        <h3 class="font-bold text-sm text-white flex items-center gap-2">
                            <i class="fa-solid fa-chart-area text-amber-400"></i>
                            Trayectoria Continua y(t) vs Observaciones Ruidosas
                        </h3>
                        <span class="text-xs text-slate-400 font-mono">Altitud (m) vs Tiempo (s)</span>
                    </div>
                    <div class="h-72 w-full">
                        <canvas id="trajectoryChart"></canvas>
                    </div>
                </div>

            </div>

        </div>

    </main>

    <!-- QR Code Modal (For Audience Interaction) -->
    <div id="qrModal" class="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 hidden flex items-center justify-center p-4" onclick="toggleQRModal()">
        <div class="bg-slate-900 border border-slate-700 rounded-2xl p-6 max-w-sm w-full text-center space-y-4 shadow-2xl" onclick="event.stopPropagation()">
            <div class="flex items-center justify-between border-b border-slate-800 pb-3">
                <h3 class="font-bold text-sm text-white flex items-center gap-2">
                    <i class="fa-solid fa-qrcode text-amber-400"></i> Conéctate a la Demo
                </h3>
                <button onclick="toggleQRModal()" class="text-slate-400 hover:text-white"><i class="fa-solid fa-xmark text-lg"></i></button>
            </div>
            <div class="bg-white p-3 rounded-xl inline-block shadow-inner">
                <img id="qrImage" src="" alt="QR Code" class="w-52 h-52">
            </div>
            <p class="text-xs text-slate-400">Escanea este código con tu teléfono para acceder y experimentar con la calibración PINN en vivo.</p>
            <p id="qrUrlText" class="text-[11px] font-mono text-amber-400 bg-slate-950 p-2 rounded-lg break-all"></p>
            <button onclick="toggleQRModal()" class="w-full py-2 bg-slate-800 hover:bg-slate-700 text-white rounded-xl text-xs font-semibold transition">Cerrar</button>
        </div>
    </div>

    <script>
        const PRESET_DESCRIPTIONS = {
            smooth_sphere: "Flujo laminar alrededor de un cuerpo esférico simétrico.",
            baseball: "Esfera rugosa con costuras que inducen transición turbulenta.",
            cylinder: "Cuerpo alargado de sección cilíndrica con alta resistencia de forma.",
            skydiver: "Cuerpo humano en posición extendida (máximo arrastre de forma)."
        };

        function updatePresetInfo() {
            const val = document.getElementById('presetSelect').value;
            document.getElementById('presetDesc').innerText = PRESET_DESCRIPTIONS[val] || '';
        }

        let chartInstance = null;

        function initChart() {
            const ctx = document.getElementById('trajectoryChart').getContext('2d');
            chartInstance = new Chart(ctx, {
                type: 'line',
                data: {
                    datasets: [
                        {
                            label: 'Mediciones del Sensor (Radar con Ruido)',
                            data: [],
                            borderColor: 'rgba(244, 63, 94, 0.9)',
                            backgroundColor: 'rgba(244, 63, 94, 0.7)',
                            pointRadius: 4.5,
                            showLine: false,
                            type: 'scatter'
                        },
                        {
                            label: 'Ajuste Continuo PINN (Inferencia)',
                            data: [],
                            borderColor: '#FF9900',
                            backgroundColor: 'rgba(255, 153, 0, 0.1)',
                            borderWidth: 3,
                            tension: 0.1,
                            fill: true
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            type: 'linear',
                            title: { display: true, text: 'Tiempo (segundos)', color: '#94a3b8' },
                            grid: { color: 'rgba(51, 65, 85, 0.4)' },
                            ticks: { color: '#94a3b8' }
                        },
                        y: {
                            title: { display: true, text: 'Altitud y(t) (metros)', color: '#94a3b8' },
                            grid: { color: 'rgba(51, 65, 85, 0.4)' },
                            ticks: { color: '#94a3b8' }
                        }
                    },
                    plugins: {
                        legend: {
                            labels: { color: '#f8fafc', font: { size: 11 } }
                        }
                    }
                }
            });
        }

        async function runDiscovery() {
            const btn = document.getElementById('runBtn');
            const originalHTML = btn.innerHTML;
            btn.disabled = true;
            btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Optimizando en EC2...';

            const payload = {
                preset: document.getElementById('presetSelect').value,
                noise_std: parseFloat(document.getElementById('noiseSlider').value),
                n_points: parseInt(document.getElementById('pointsSlider').value),
                initial_cd_guess: parseFloat(document.getElementById('guessSlider').value),
                epochs: 120
            };

            try {
                const res = await fetch('/discover_cd', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await res.json();

                // Actualizar métricas
                document.getElementById('resCdTrue').innerText = data.cd_true.toFixed(4);
                document.getElementById('resCdEst').innerText = data.cd_estimated.toFixed(4);
                document.getElementById('resError').innerText = `Error Relativo: ${data.relative_error_pct}%`;
                document.getElementById('resLatency').innerText = `${data.latency_ms} ms`;
                document.getElementById('latencyVal').innerText = `${data.latency_ms} ms`;
                document.getElementById('telemetryBadge').classList.remove('hidden');

                // Actualizar gráfica
                const scatterData = data.sensor_points.map(p => ({ x: p.t, y: p.y }));
                const lineData = data.trajectory_fine.map(p => ({ x: p.t, y: p.y }));

                chartInstance.data.datasets[0].data = scatterData;
                chartInstance.data.datasets[1].data = lineData;
                chartInstance.update();

            } catch (err) {
                console.error("Error en descubrimiento:", err);
                alert("Error al ejecutar descubrimiento: " + err);
            } finally {
                btn.disabled = false;
                btn.innerHTML = originalHTML;
            }
        }

        // Modal de Código QR
        function toggleQRModal() {
            const modal = document.getElementById('qrModal');
            const qrImg = document.getElementById('qrImage');
            const qrUrlText = document.getElementById('qrUrlText');
            const isHidden = modal.classList.contains('hidden');
            if (isHidden) {
                const currentUrl = window.location.href;
                qrImg.src = `https://api.qrserver.com/v1/create-qr-code/?size=250x250&data=${encodeURIComponent(currentUrl)}`;
                qrUrlText.innerText = currentUrl;
                modal.classList.remove('hidden');
            } else {
                modal.classList.add('hidden');
            }
        }

        // Atajos de Teclado para el Ponente (AWS Community Day 2026)
        window.addEventListener('keydown', (e) => {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;

            if (e.key === ' ' || e.code === 'Space' || e.key === 'Enter') {
                e.preventDefault();
                runDiscovery();
            } else if (e.key === '1') {
                document.getElementById('presetSelect').value = 'smooth_sphere';
                updatePresetInfo();
                runDiscovery();
            } else if (e.key === '2') {
                document.getElementById('presetSelect').value = 'baseball';
                updatePresetInfo();
                runDiscovery();
            } else if (e.key === '3') {
                document.getElementById('presetSelect').value = 'cylinder';
                updatePresetInfo();
                runDiscovery();
            } else if (e.key === '4') {
                document.getElementById('presetSelect').value = 'skydiver';
                updatePresetInfo();
                runDiscovery();
            } else if (e.key === 'q' || e.key === 'Q') {
                toggleQRModal();
            }
        });

        window.addEventListener('DOMContentLoaded', () => {
            initChart();
            runDiscovery();
        });
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)
