# src/physics_engine.py
from typing import Dict, Any, Tuple
import numpy as np
from scipy.integrate import solve_ivp
import os
import csv

from src.config import Config


class PhysicsEngine:
    """Motor físico analítico y numérico para dinámica de cuerpos con resistencia aerodinámica cuadrática."""

    @staticmethod
    def ode_system(
        t: float,
        state: np.ndarray,
        cd: float,
        mass: float,
        gravity: float,
        rho: float,
        area: float,
    ) -> np.ndarray:
        """Define el sistema de ecuaciones diferenciales de primer orden:
        dy/dt = v
        dv/dt = -g - (0.5 * rho * A * Cd / m) * v * |v|
        """
        y, v = state
        dydt = v
        # Fuerza de arrastre cuadrática que siempre se opone al vector velocidad
        drag_acceleration = (0.5 * rho * area * cd / mass) * v * np.abs(v)
        dvdt = -gravity - drag_acceleration
        return np.array([dydt, dvdt])

    @classmethod
    def solve_exact_trajectory(
        cls,
        t_eval: np.ndarray,
        cd: float = Config.CD_TRUE,
        mass: float = Config.MASS,
        gravity: float = Config.GRAVITY,
        rho: float = Config.AIR_DENSITY,
        area: float = Config.CROSS_AREA,
        y0: float = Config.INITIAL_HEIGHT,
        v0: float = Config.INITIAL_VELOCITY,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Resuelve con alta precisión la trayectoria mediante Runge-Kutta 45 (RK45).

        Returns:
            Tuple[np.ndarray, np.ndarray]: (posiciones_y, velocidades_v) evaluadas en t_eval.
        """
        t_span = (float(t_eval[0]), float(t_eval[-1]))
        initial_state = [y0, v0]

        sol = solve_ivp(
            fun=cls.ode_system,
            t_span=t_span,
            y0=initial_state,
            t_eval=t_eval,
            method="RK45",
            args=(cd, mass, gravity, rho, area),
            rtol=1e-8,
            atol=1e-10,
        )

        y_exact = sol.y[0]
        v_exact = sol.y[1]
        return y_exact, v_exact

    @classmethod
    def generate_synthetic_sensor_data(
        cls,
        n_points: int = Config.N_SENSOR_POINTS,
        noise_std: float = Config.NOISE_STD,
        cd_true: float = Config.CD_TRUE,
        mass: float = Config.MASS,
        gravity: float = Config.GRAVITY,
        rho: float = Config.AIR_DENSITY,
        area: float = Config.CROSS_AREA,
        t_min: float = Config.T_MIN,
        t_max: float = Config.T_MAX,
        seed: int = Config.RANDOM_SEED,
    ) -> Dict[str, Any]:
        """Genera mediciones sintéticas discretas con perturbación gaussiana simulando sensores de radar.

        Returns:
            Dict[str, Any]: Diccionario con tiempos, mediciones ruidosas, ground truth y metadatos físicos.
        """
        np.random.seed(seed)
        t_sensor = np.linspace(t_min, t_max, n_points)
        y_true, v_true = cls.solve_exact_trajectory(
            t_eval=t_sensor,
            cd=cd_true,
            mass=mass,
            gravity=gravity,
            rho=rho,
            area=area,
        )

        noise = np.random.normal(0.0, noise_std, size=y_true.shape)
        y_measured = y_true + noise

        # Evitar valores negativos si impactara el suelo
        y_measured = np.maximum(y_measured, 0.0)

        return {
            "t": t_sensor,
            "y_measured": y_measured,
            "y_true": y_true,
            "v_true": v_true,
            "cd_true": cd_true,
            "noise_std": noise_std,
            "params": {
                "mass": mass,
                "gravity": gravity,
                "rho": rho,
                "area": area,
            },
        }

    @classmethod
    def export_dataset_to_csv(
        cls,
        data: Dict[str, Any],
        filepath: str = "data/synthetic_measurements.csv",
    ) -> str:
        """Guarda el dataset de mediciones sintéticas en un archivo CSV estructurado."""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["time_s", "altitude_measured_m", "altitude_exact_m", "velocity_exact_m_s"])
            for t_i, y_m, y_e, v_e in zip(
                data["t"], data["y_measured"], data["y_true"], data["v_true"]
            ):
                writer.writerow([round(t_i, 4), round(y_m, 4), round(y_e, 4), round(v_e, 4)])
        return filepath
