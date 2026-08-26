# src/train.py
import os
import time
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

from src.config import Config
from src.physics_engine import PhysicsEngine
from src.pinn_inverse import (
    InverseDragPINN,
    compute_inverse_derivatives,
    compute_ode_residual,
)


def set_seed(seed: int = Config.RANDOM_SEED) -> None:
    """Fija las semillas aleatorias para garantizar reproducibilidad matemática."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def train_inverse_pinn(
    epochs: int = 800,
    lr_net: float = 5e-3,
    lr_cd: float = 2e-2,
) -> Tuple[InverseDragPINN, Dict[str, List[float]]]:
    """Entrena la PINN inversa para descubrir el coeficiente de arrastre Cd a partir de datos de sensor."""
    set_seed(Config.RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Iniciando entrenamiento PINN Inverso en: {device}")
    print(f"🎯 Valor Real Cd (Ground Truth): {Config.CD_TRUE:.4f}")
    print(f"🎲 Suposición Inicial de Cd:    {Config.CD_INITIAL_GUESS:.4f}")
    print("=" * 60)

    # 1. Generación de datos sintéticos con ruido (Sensores de Radar)
    sensor_data = PhysicsEngine.generate_synthetic_sensor_data(
        n_points=Config.N_SENSOR_POINTS,
        noise_std=Config.NOISE_STD,
        seed=Config.RANDOM_SEED,
    )
    csv_path = PhysicsEngine.export_dataset_to_csv(sensor_data)
    print(f"📊 Dataset sintético exportado a: {csv_path}")

    # Preparar tensores de entrenamiento
    t_sensor = torch.tensor(sensor_data["t"], dtype=torch.float32).view(-1, 1).to(device)
    y_sensor = torch.tensor(sensor_data["y_measured"], dtype=torch.float32).view(-1, 1).to(device)

    # Puntos de colocación física en el interior del dominio temporal
    t_col = torch.linspace(Config.T_MIN, Config.T_MAX, Config.N_COLLOCATION).view(-1, 1).to(device)

    # 2. Inicializar Modelo PINN Inverso y Optimizador
    model = InverseDragPINN(initial_cd_guess=Config.CD_INITIAL_GUESS).to(device)
    optimizer = optim.Adam([
        {"params": model.layers.parameters(), "lr": lr_net},
        {"params": [model.raw_cd], "lr": lr_cd},
    ])

    history: Dict[str, List[float]] = {
        "loss_total": [],
        "loss_data": [],
        "loss_ode": [],
        "cd_history": [],
    }

    start_time = time.time()

    # 3. Bucle de Optimización
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        # A. Pérdida de Datos (MSE contra mediciones del sensor)
        y_pred_sensor = model(t_sensor)
        loss_data = torch.mean((y_pred_sensor - y_sensor) ** 2)

        # B. Pérdida Física del Residuo de la ODE
        ode_res = compute_ode_residual(
            model=model,
            t_collocation=t_col,
            mass=Config.MASS,
            gravity=Config.GRAVITY,
            rho=Config.AIR_DENSITY,
            area=Config.CROSS_AREA,
        )
        loss_ode = torch.mean(ode_res ** 2)

        # Pérdida Total Ponderada
        loss = Config.WEIGHT_DATA * loss_data + 10.0 * loss_ode

        loss.backward()
        optimizer.step()

        current_cd = float(model.cd.item())
        history["loss_total"].append(loss.item())
        history["loss_data"].append(loss_data.item())
        history["loss_ode"].append(loss_ode.item())
        history["cd_history"].append(current_cd)

        if epoch % 100 == 0 or epoch == 1:
            error_pct = abs(current_cd - Config.CD_TRUE) / Config.CD_TRUE * 100.0
            print(
                f"Época {epoch:4d}/{epochs} | "
                f"Loss Total: {loss.item():.6f} | "
                f"Loss Data: {loss_data.item():.6f} | "
                f"Loss ODE: {loss_ode.item():.6f} | "
                f"Cd Descubierto: {current_cd:.4f} (Error: {error_pct:.2f}%)"
            )

    elapsed = time.time() - start_time
    final_cd = float(model.cd.item())
    final_error = abs(final_cd - Config.CD_TRUE) / Config.CD_TRUE * 100.0

    print("=" * 60)
    print(f"✅ Entrenamiento completado en {elapsed:.2f} s")
    print(f"🎯 Cd Real (Ground Truth):  {Config.CD_TRUE:.6f}")
    print(f"🔍 Cd Descubierto (PINN):    {final_cd:.6f}")
    print(f"📉 Error Relativo Final:     {final_error:.2f}%")
    print("=" * 60)

    # 4. Guardar Checkpoint y TorchScript
    os.makedirs(os.path.dirname(Config.MODEL_PATH) or ".", exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "cd_discovered": final_cd,
        "cd_true": Config.CD_TRUE,
        "epochs": epochs,
        "final_loss": history["loss_total"][-1],
        "params": {
            "mass": Config.MASS,
            "gravity": Config.GRAVITY,
            "rho": Config.AIR_DENSITY,
            "area": Config.CROSS_AREA,
            "y0": Config.INITIAL_HEIGHT,
        },
    }
    torch.save(checkpoint, Config.MODEL_PATH)
    print(f"💾 Checkpoint guardado en: {Config.MODEL_PATH}")

    # Exportar TorchScript
    model.eval()
    dummy_input = torch.tensor([[1.0]], dtype=torch.float32).to(device)
    try:
        traced = torch.jit.trace(model, dummy_input)
        traced.save(Config.MODEL_SCRIPT_PATH)
        print(f"📦 TorchScript exportado en: {Config.MODEL_SCRIPT_PATH}")
    except Exception as e:
        print(f"⚠️ Nota al exportar TorchScript: {e}")

    # 5. Generar Visualizaciones Analíticas
    generate_training_plots(model, sensor_data, history)

    return model, history


def generate_training_plots(
    model: InverseDragPINN,
    sensor_data: Dict,
    history: Dict[str, List[float]],
) -> None:
    """Genera y guarda en data/ figuras de alta resolución para la charla y el repositorio."""
    os.makedirs(Config.DATA_DIR, exist_ok=True)
    model.eval()

    t_fine = np.linspace(Config.T_MIN, Config.T_MAX, 300)
    with torch.no_grad():
        t_fine_tensor = torch.tensor(t_fine, dtype=torch.float32).view(-1, 1)
        y_pinn_pred = model(t_fine_tensor).numpy().flatten()

    y_exact, v_exact = PhysicsEngine.solve_exact_trajectory(t_fine)

    # --- 1. Gráfica de Convergencia de Cd ---
    plt.figure(figsize=(9, 5), dpi=150)
    plt.plot(history["cd_history"], color="#FF9900", lw=2.2, label=f"PINN Cd estimado (Final: {history['cd_history'][-1]:.4f})")
    plt.axhline(Config.CD_TRUE, color="#00A4E4", linestyle="--", lw=2, label=f"Ground Truth Cd = {Config.CD_TRUE:.2f} (Esfera Lisa)")
    plt.axhline(Config.CD_INITIAL_GUESS, color="#888888", linestyle=":", lw=1.5, label=f"Suposición Inicial = {Config.CD_INITIAL_GUESS:.2f}")
    plt.title(r"Convergencia del Parámetro Físico Desconocido $C_d$ (Problema Inverso)", fontsize=13, fontweight="bold")
    plt.xlabel("Épocas de Entrenamiento", fontsize=11)
    plt.ylabel(r"Coeficiente de Arrastre $C_d$", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, facecolor="white", loc="best")
    plt.tight_layout()
    plt.savefig(f"{Config.DATA_DIR}/cd_convergence.png")
    plt.close()

    # --- 2. Gráfica de Trayectoria Aprendida ---
    plt.figure(figsize=(9, 5), dpi=150)
    plt.plot(t_fine, y_exact, "k-", lw=2, label=r"Trayectoria Exacta RK45 ($C_d = 0.47$)")
    plt.scatter(
        sensor_data["t"],
        sensor_data["y_measured"],
        color="#E74C3C",
        alpha=0.75,
        s=35,
        label=rf"Mediciones Ruidosas de Sensor ($\sigma={Config.NOISE_STD}$ m)",
        zorder=3,
    )
    plt.plot(t_fine, y_pinn_pred, "--", color="#2ECC71", lw=2.5, label="Ajuste Continuo PINN Inverso")
    plt.title("Ajuste de Trayectoria: Sensores vs Solución Física vs PINN", fontsize=13, fontweight="bold")
    plt.xlabel("Tiempo $t$ (segundos)", fontsize=11)
    plt.ylabel("Altitud $y(t)$ (metros)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, facecolor="white")
    plt.tight_layout()
    plt.savefig(f"{Config.DATA_DIR}/trajectory_discovery.png")
    plt.close()

    # --- 3. Gráfica de Pérdidas ---
    plt.figure(figsize=(9, 5), dpi=150)
    plt.semilogy(history["loss_total"], label="Pérdida Total", color="#2C3E50", lw=2)
    plt.semilogy(history["loss_data"], label=r"$\mathcal{L}_{data}$ (MSE Sensor)", color="#E67E22", alpha=0.8)
    plt.semilogy(history["loss_ode"], label=r"$\mathcal{L}_{ODE}$ (Residuo Físico de Arrastre)", color="#3498DB", alpha=0.8)
    plt.title(r"Historial de Pérdidas Multiobjetivo ($\mathcal{L}_{data} + \lambda \mathcal{L}_{ODE}$)", fontsize=13, fontweight="bold")
    plt.xlabel("Épocas", fontsize=11)
    plt.ylabel("Pérdida (Escala Logarítmica)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, facecolor="white")
    plt.tight_layout()
    plt.savefig(f"{Config.DATA_DIR}/loss_history.png")
    plt.close()

    # --- 4. Dashboard Resumen 4 en 1 para el AWS Community Day ---
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    
    # Panel (0,0): Trayectoria
    axs[0, 0].plot(t_fine, y_exact, "k-", lw=2, label="Exacta RK45")
    axs[0, 0].scatter(sensor_data["t"], sensor_data["y_measured"], color="#E74C3C", s=25, alpha=0.7, label="Sensores")
    axs[0, 0].plot(t_fine, y_pinn_pred, "--", color="#2ECC71", lw=2.2, label="PINN Inverso")
    axs[0, 0].set_title(r"A. Trayectoria $y(t)$ vs Observaciones", fontweight="bold")
    axs[0, 0].set_xlabel("Tiempo (s)")
    axs[0, 0].set_ylabel("Altitud (m)")
    axs[0, 0].grid(True, linestyle=":", alpha=0.6)
    axs[0, 0].legend()

    # Panel (0,1): Convergencia Cd
    axs[0, 1].plot(history["cd_history"], color="#FF9900", lw=2.2, label=f"Estimado: {history['cd_history'][-1]:.4f}")
    axs[0, 1].axhline(Config.CD_TRUE, color="#00A4E4", linestyle="--", lw=2, label=f"Real: {Config.CD_TRUE:.2f}")
    axs[0, 1].set_title(r"B. Descubrimiento de Parámetro $C_d$", fontweight="bold")
    axs[0, 1].set_xlabel("Épocas")
    axs[0, 1].set_ylabel(r"Valor de $C_d$")
    axs[0, 1].grid(True, linestyle=":", alpha=0.6)
    axs[0, 1].legend()

    # Panel (1,0): Pérdidas
    axs[1, 0].semilogy(history["loss_total"], color="#2C3E50", lw=1.8, label="Total")
    axs[1, 0].semilogy(history["loss_data"], color="#E67E22", lw=1.5, label="Data")
    axs[1, 0].semilogy(history["loss_ode"], color="#3498DB", lw=1.5, label=r"Física (ODE)")
    axs[1, 0].set_title("C. Convergencia de Pérdidas", fontweight="bold")
    axs[1, 0].set_xlabel("Épocas")
    axs[1, 0].set_ylabel("Loss (Log)")
    axs[1, 0].grid(True, linestyle=":", alpha=0.6)
    axs[1, 0].legend()

    # Panel (1,1): Velocidad continua recuperada dy/dt
    t_fine_torch = torch.tensor(t_fine, dtype=torch.float32).view(-1, 1).requires_grad_(True)
    _, dy_dt_pred, _ = compute_inverse_derivatives(model, t_fine_torch)
    v_pinn = dy_dt_pred.detach().numpy().flatten()
    axs[1, 1].plot(t_fine, v_exact, "k-", lw=2, label="Velocidad Exacta RK45")
    axs[1, 1].plot(t_fine, v_pinn, "r--", lw=2, label="Velocidad Derivada Autograd")
    axs[1, 1].set_title(r"D. Perfil de Velocidad $\dot{y}(t)$ con Autograd", fontweight="bold")
    axs[1, 1].set_xlabel("Tiempo (s)")
    axs[1, 1].set_ylabel("Velocidad (m/s)")
    axs[1, 1].grid(True, linestyle=":", alpha=0.6)
    axs[1, 1].legend()

    plt.suptitle(r"AWS Community Day | Physics-Informed Neural Networks: Descubrimiento Inverso de $C_d$", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{Config.DATA_DIR}/inverse_discovery_dashboard.png")
    plt.close()

    print(f"📊 Gráficos y Dashboard 4 en 1 guardados exitosamente en: '{Config.DATA_DIR}/'")


if __name__ == "__main__":
    train_inverse_pinn()
