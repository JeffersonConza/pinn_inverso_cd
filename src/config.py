# src/config.py
from typing import Dict, Any, List


class Config:
    """Configuración central para el problema inverso de PINN (Descubrimiento de Cd)."""

    # --- Parámetros Físicos Canónicos (Esfera Lisa por Defecto) ---
    MASS: float = 1.0  # Masa del cuerpo (kg)
    GRAVITY: float = 9.81  # Aceleración de la gravedad (m/s^2)
    AIR_DENSITY: float = 1.225  # Densidad del aire al nivel del mar (kg/m^3)
    CROSS_AREA: float = 0.01  # Área de sección transversal frontal (m^2)
    CD_TRUE: float = 0.47  # Coeficiente de arrastre real (Ground Truth de esfera lisa)

    # Condiciones Iniciales y Dominio Temporal
    INITIAL_HEIGHT: float = 100.0  # Altura inicial y(0) en metros
    INITIAL_VELOCITY: float = 0.0  # Velocidad inicial dy/dt(0) en m/s (desde reposo)
    T_MIN: float = 0.0  # Tiempo inicial (s)
    T_MAX: float = 3.0  # Tiempo final de simulación (s)

    # Parámetros de Ruido del Sensor (Simulación de Radar / Altímetro)
    NOISE_STD: float = 0.05  # Desviación estándar del ruido gaussiano epsilon ~ N(0, sigma^2)
    N_SENSOR_POINTS: int = 30  # Cantidad de observaciones ruidosas del sensor

    # --- Arquitectura de la Red Neuronal (PINN) ---
    INPUT_DIM: int = 1  # Entrada: tiempo t
    OUTPUT_DIM: int = 1  # Salida: posición predicha y(t)
    NUM_HIDDEN_LAYERS: int = 3  # Cantidad de capas ocultas
    HIDDEN_NEURONS: int = 64  # Neuronas por capa oculta
    CD_INITIAL_GUESS: float = 0.10  # Suposición inicial lejana para demostrar el descubrimiento

    # --- Parámetros de Entrenamiento Offline ---
    EPOCHS: int = 3000  # Épocas para entrenamiento maestro offline
    LR: float = 1e-3  # Tasa de aprendizaje para Adam
    N_COLLOCATION: int = 200  # Puntos de colocación física en el dominio temporal
    WEIGHT_DATA: float = 1.0  # Peso de la pérdida de ajuste a datos de sensores (MSE)
    WEIGHT_ODE: float = 0.1  # Peso de la penalización del residuo de la ecuación de movimiento
    RANDOM_SEED: int = 42  # Semilla para reproducibilidad matemática

    # --- Parámetros de Calibración Rápida en Vivo (Live Demo EC2) ---
    FAST_CALIBRATION_EPOCHS: int = 250  # Épocas rápidas en CPU (< 400 ms de latencia)
    FAST_CALIBRATION_LR: float = 0.02  # LR agresivo para convergencia instantánea

    # --- Rutas de Persistencia de Modelos y Datos ---
    MODEL_PATH: str = "models/inverse_drag_pinn.pt"
    MODEL_SCRIPT_PATH: str = "models/inverse_drag_script.pt"
    DATA_DIR: str = "data"

    # --- Presets Físicos Predefinidos para Demostraciones ---
    PRESETS: Dict[str, Dict[str, Any]] = {
        "smooth_sphere": {
            "name": "Esfera Lisa (Acero)",
            "cd_true": 0.47,
            "mass": 1.0,
            "area": 0.01,
            "description": "Flujo laminar alrededor de un cuerpo esférico simétrico."
        },
        "baseball": {
            "name": "Pelota de Béisbol",
            "cd_true": 0.30,
            "mass": 0.145,
            "area": 0.0042,
            "description": "Esfera rugosa con costuras que inducen transición turbulenta."
        },
        "cylinder": {
            "name": "Cilindro / Cohete Sonda",
            "cd_true": 0.82,
            "mass": 2.5,
            "area": 0.02,
            "description": "Cuerpo alargado de sección cilíndrica con alta resistencia de forma."
        },
        "skydiver": {
            "name": "Paracaidista en Caída Libre",
            "cd_true": 1.20,
            "mass": 80.0,
            "area": 0.70,
            "description": "Cuerpo humano en posición extendida (máximo arrastre de forma)."
        }
    }
