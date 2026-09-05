"""
test_api.py
-----------
Suite de pruebas unitarias y de integración automatizadas con pytest y TestClient (FastAPI)
para la PINN de Descubrimiento Inverso de Coeficiente de Arrastre Cd.
AWS Community Day Ecuador 2026 · Jefferson Alfredo Conza Fajardo.
"""

import sys
from pathlib import Path
import pytest
import torch
import numpy as np
from fastapi.testclient import TestClient

# Asegurar que el directorio raíz esté en sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from api.main import app
from src.config import Config
from src.physics_engine import PhysicsEngine
from src.pinn_inverse import (
    InverseDragPINN,
    compute_inverse_derivatives,
    compute_ode_residual,
)

client = TestClient(app)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Pruebas Unitarias del Modelo y Motor Físico (PyTorch Autograd)
# ══════════════════════════════════════════════════════════════════════════════

class TestInversePINNModelAndPhysics:
    def test_model_forward_and_derivatives(self):
        """Verifica que la red y compute_inverse_derivatives calculen posición, velocidad y aceleración."""
        model = InverseDragPINN(initial_cd_guess=0.10)
        model.eval()

        t = torch.tensor([[0.5], [1.0], [2.0]], dtype=torch.float32, requires_grad=True)
        y_pred, dy_dt, d2y_dt2 = compute_inverse_derivatives(model, t)

        assert y_pred.shape == (3, 1)
        assert dy_dt.shape == (3, 1)
        assert d2y_dt2.shape == (3, 1)
        assert not torch.isnan(y_pred).any()
        assert not torch.isnan(dy_dt).any()
        assert not torch.isnan(d2y_dt2).any()

    def test_physics_engine_synthetic_generation(self):
        """Comprueba la generación de trayectorias sintéticas no lineales con ruido gaussiano."""
        data = PhysicsEngine.generate_synthetic_sensor_data(
            n_points=30,
            noise_std=0.05,
            cd_true=0.47,
            mass=1.0,
            gravity=9.81,
            rho=1.225,
            area=0.01,
            t_min=0.0,
            t_max=3.0,
        )

        assert "t" in data
        assert "y_measured" in data
        assert "y_true" in data
        assert len(data["t"]) == 30
        assert len(data["y_measured"]) == 30
        assert abs(data["y_true"][0] - Config.INITIAL_HEIGHT) < 1e-3

    def test_ode_residual_computation(self):
        """Comprueba el cálculo del residuo de la ecuación diferencial de movimiento de Newton."""
        model = InverseDragPINN(initial_cd_guess=0.47)
        model.eval()

        t_col = torch.linspace(0.0, 3.0, 20).view(-1, 1)
        res = compute_ode_residual(
            model=model,
            t_collocation=t_col,
            mass=1.0,
            gravity=9.81,
            rho=1.225,
            area=0.01,
        )

        assert res.shape == (20, 1)
        assert not torch.isnan(res).any()


# ══════════════════════════════════════════════════════════════════════════════
# 2. Pruebas de Endpoints de Inferencia y Descubrimiento FastAPI
# ══════════════════════════════════════════════════════════════════════════════

class TestFastAPIEndpoints:
    def test_root_endpoint(self):
        """Valida el endpoint raíz con metadatos del proyecto y ponente."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "title" in data
        assert "author" in data
        assert "active_cd" in data

    def test_health_endpoint(self):
        """Valida el health check para monitoreo y liveness en AWS EC2."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert "active_cd" in data

    def test_presets_catalog_endpoint(self):
        """Comprueba el catálogo de geometrías físicas y coeficientes Ground Truth."""
        response = client.get("/presets")
        assert response.status_code == 200
        presets = response.json()
        assert "smooth_sphere" in presets
        assert "baseball" in presets
        assert "cylinder" in presets
        assert "skydiver" in presets

    def test_predict_single_point(self):
        """Comprueba la inferencia puntual continua en t=1.5s."""
        response = client.get("/predict?t=1.5")
        assert response.status_code == 200
        data = response.json()
        assert data["t"] == 1.5
        assert "altitude_y" in data
        assert "velocity_v" in data
        assert "cd_calibrated" in data

    def test_discover_cd_smooth_sphere(self):
        """Valida el descubrimiento inverso en vivo para una esfera lisa (Cd = 0.47)."""
        payload = {
            "preset": "smooth_sphere",
            "noise_std": 0.05,
            "n_points": 25,
            "initial_cd_guess": 0.10,
            "epochs": 200,
        }
        response = client.post("/discover_cd", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["cd_true"] == 0.47
        assert abs(data["cd_estimated"] - 0.47) < 0.20  # Converge cerca del valor real
        assert data["latency_ms"] > 0.0

    def test_discover_cd_baseball(self):
        """Valida el descubrimiento inverso para una pelota de béisbol (Cd = 0.30)."""
        payload = {
            "preset": "baseball",
            "noise_std": 0.03,
            "n_points": 25,
            "initial_cd_guess": 0.80,
            "epochs": 200,
        }
        response = client.post("/discover_cd", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["cd_true"] == 0.30
        assert abs(data["cd_estimated"] - 0.30) < 0.20

    def test_validation_error_out_of_bounds(self):
        """Verifica que el schema rechace tiempos fuera del dominio temporal [0.0, 3.0]."""
        response = client.get("/predict?t=-1.0")
        assert response.status_code == 422

    def test_demo_endpoint(self):
        """Comprueba que la interfaz interactiva web se sirva correctamente."""
        response = client.get("/demo")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")


# ══════════════════════════════════════════════════════════════════════════════
# CLI Runner para Ejecución Directa
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    import requests

    parser = argparse.ArgumentParser(description="Test API Descubrimiento Inverso Cd PINN")
    parser.add_argument("--url", default=None, help="URL remota (ej: http://ec2-ip:8000)")
    args = parser.parse_args()

    if args.url:
        base = args.url.rstrip("/")
        print(f"🧪 Probando servidor remoto en: {base}")
        r = requests.get(f"{base}/health", timeout=5)
        print(f"  Status /health: {r.status_code} -> {r.json()}")
    else:
        pytest.main(["-v", __file__])
