# src/pinn_inverse.py
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.autograd as autograd

from src.config import Config


class InverseDragPINN(nn.Module):
    """Physics-Informed Neural Network (PINN) para el descubrimiento inverso del coeficiente de arrastre Cd.

    Incorpora la base cinemática analítica de caída libre y una red neuronal profunda que aprende
    las desviaciones aerodinámicas no lineales, optimizando dinámicamente el parámetro físico Cd
    mediante diferenciación automática (autograd).
    """

    def __init__(
        self,
        initial_cd_guess: float = Config.CD_INITIAL_GUESS,
        y0: float = Config.INITIAL_HEIGHT,
        v0: float = Config.INITIAL_VELOCITY,
        gravity: float = Config.GRAVITY,
    ) -> None:
        """Inicializa la red neuronal y el parámetro aprendible Cd."""
        super().__init__()
        self.y0 = y0
        self.v0 = v0
        self.gravity = gravity

        # 1. Arquitectura MLP para la corrección de arrastre
        layer_dims: List[int] = (
            [Config.INPUT_DIM]
            + [Config.HIDDEN_NEURONS] * Config.NUM_HIDDEN_LAYERS
            + [Config.OUTPUT_DIM]
        )

        self.layers: nn.ModuleList = nn.ModuleList()
        for i in range(len(layer_dims) - 1):
            linear_layer: nn.Linear = nn.Linear(layer_dims[i], layer_dims[i + 1])
            self._init_weights(linear_layer, is_output=(i == len(layer_dims) - 2))
            self.layers.append(linear_layer)
            if i < len(layer_dims) - 2:
                self.layers.append(nn.Tanh())

        # 2. Parámetro físico aprendible (Descubrimiento Inverso)
        self.raw_cd = nn.Parameter(torch.tensor([float(initial_cd_guess)], dtype=torch.float32))

    @property
    def cd(self) -> torch.Tensor:
        """Garantiza positividad física del coeficiente de arrastre."""
        return torch.abs(self.raw_cd)

    @staticmethod
    def _init_weights(layer: nn.Linear, is_output: bool = False) -> None:
        """Inicialización de pesos Xavier para estabilidad y convergencia rápida."""
        gain: float = 0.1 if is_output else nn.init.calculate_gain("tanh")
        nn.init.xavier_normal_(layer.weight, gain=gain)
        nn.init.zeros_(layer.bias)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Forward pass que satisface analíticamente las condiciones iniciales y(0)=y0, v(0)=v0:
        y(t) = y0 + v0*t - 0.5*g*t^2 + net(t) * t^2
        """
        inputs = t
        for layer in self.layers:
            inputs = layer(inputs)

        # Corrección cuadrática suave en el tiempo
        correction = inputs * (t ** 2)
        kinematic_trajectory = self.y0 + self.v0 * t - 0.5 * self.gravity * (t ** 2) + correction
        return kinematic_trajectory


def compute_inverse_derivatives(
    model: nn.Module,
    t: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calcula la posición predicha y las derivadas continuas dy/dt (velocidad) y d²y/dt² (aceleración).

    Args:
        model (nn.Module): Modelo PINN.
        t (torch.Tensor): Tensor de tiempo de forma (N, 1) con requires_grad=True.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (y_pred, dy_dt, d2y_dt2)
    """
    y_pred: torch.Tensor = model(t)
    ones: torch.Tensor = torch.ones_like(y_pred)

    # Primera derivada (Velocidad: dy/dt)
    dy_dt: torch.Tensor = autograd.grad(
        y_pred,
        t,
        grad_outputs=ones,
        create_graph=True,
        retain_graph=True,
    )[0]

    # Segunda derivada (Aceleración: d²y/dt²)
    d2y_dt2: torch.Tensor = autograd.grad(
        dy_dt,
        t,
        grad_outputs=ones,
        create_graph=True,
        retain_graph=True,
    )[0]

    return y_pred, dy_dt, d2y_dt2


def compute_ode_residual(
    model: InverseDragPINN,
    t_collocation: torch.Tensor,
    mass: float = Config.MASS,
    gravity: float = Config.GRAVITY,
    rho: float = Config.AIR_DENSITY,
    area: float = Config.CROSS_AREA,
) -> torch.Tensor:
    """Calcula el residuo normalizado de aceleración de la ecuación de movimiento con resistencia cuadrática.

    Ecuación gobernante:
        d²y/dt² + g + (0.5 * rho * A * Cd / m) * (dy/dt) * |dy/dt| = 0

    Returns:
        torch.Tensor: Residuo puntual de la ODE en m/s² para cada punto de colocación.
    """
    t_col = t_collocation.clone().detach().requires_grad_(True)
    _, dy_dt, d2y_dt2 = compute_inverse_derivatives(model, t_col)

    # Aceleración de arrastre aerodinámico
    drag_acc = (0.5 * rho * area / mass) * model.cd * dy_dt * torch.abs(dy_dt)

    # Residuo físico (debe tender a cero)
    residual = d2y_dt2 + gravity + drag_acc
    return residual
