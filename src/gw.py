"""Algoritmo de aproximação de Goemans-Williamson para o Max Cut.

Resolve o relaxamento por Programação Semidefinida (SDP) via CVXPY e aplica
arredondamento por hiperplano aleatório para obter uma partição em {-1, +1}^n.
"""
from __future__ import annotations

import time
from typing import Any

import cvxpy as cp
import numpy as np

from src.graphs import cut_value, validate_weight_matrix


def solve_max_cut_sdp(
    W: np.ndarray,
    solver: str | None = None,
    eps: float = 1e-7,
) -> tuple[np.ndarray, float]:
    """Parte A: Modela e resolve o relaxamento SDP do Max Cut.

    Formulação:
        maximizar   (1/4) * sum_{i,j} W_{ij} * (1 - X_{ij})
        sujeito a   X >> 0  (X semidefinida positiva)
                    X_{ii} = 1, para todo i = 0, ..., n-1

    Parâmetros:
        W: Matriz de pesos (n x n), simétrica, com diagonal zero.
        solver: Nome do solver CVXPY (ex.: 'CLARABEL', 'SCS').
            Se None, tenta 'CLARABEL' e usa 'SCS' como fallback.
        eps: Tolerância numérica do solver.

    Retorna:
        (X, sdp_value): Matriz de Gram ótima (n x n) e o valor ótimo da SDP
        (limitante superior para o corte máximo).
    """
    W = np.asarray(W, dtype=float)
    validate_weight_matrix(W)

    n = W.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=float), 0.0
    if n == 1:
        return np.ones((1, 1), dtype=float), 0.0

    W_sym = 0.5 * (W + W.T)
    X = cp.Variable((n, n), PSD=True)

    # (1/4) * sum_{i,j} W_{ij} (1 - X_{ij}) = (1/4) * (sum(W) - tr(W @ X))
    objective = cp.Maximize(0.25 * (float(W_sym.sum()) - cp.trace(W_sym @ X)))
    constraints = [cp.diag(X) == 1.0]

    problem = cp.Problem(objective, constraints)

    if solver is not None:
        problem.solve(solver=solver)
    else:
        try:
            problem.solve(
                solver=cp.CLARABEL,
                tol_gap_abs=eps,
                tol_gap_rel=eps,
                tol_feas=eps,
            )
        except (cp.SolverError, Exception):
            problem.solve(solver=cp.SCS, eps=eps)

    if X.value is None or problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"Falha ao resolver a SDP. Status: {problem.status}")

    X_opt = 0.5 * (X.value + X.value.T)
    sdp_value = float(problem.value)

    return X_opt, sdp_value


def extract_unit_vectors(X: np.ndarray, eig_tol: float = 1e-10) -> np.ndarray:
    """Parte B: Fatora X = V @ V.T via decomposição espectral para recuperar v_i.

    Trunca autovalores negativos muito pequenos (ruído numérico do solver) e
    normaliza cada linha para garantir ||v_i||_2 = 1.
    """
    n = X.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=float)

    eigvals, eigvecs = np.linalg.eigh(X)

    min_eig = float(np.min(eigvals))
    if min_eig < -1e-4:
        raise ValueError(
            f"Matriz X não é semidefinida positiva dentro da tolerância (min eig: {min_eig:.2e})."
        )

    eigvals_clipped = np.where(eigvals < eig_tol, 0.0, eigvals)
    V = eigvecs * np.sqrt(eigvals_clipped)

    norms = np.linalg.norm(V, axis=1, keepdims=True)
    zero_rows = norms[:, 0] < 1e-12
    if np.any(zero_rows):
        V[zero_rows] = 0.0
        V[zero_rows, 0] = 1.0
        norms = np.linalg.norm(V, axis=1, keepdims=True)

    return V / norms


def hyperplane_rounding(
    W: np.ndarray,
    V: np.ndarray,
    k: int = 100,
    seed: int | None = None,
) -> tuple[float, np.ndarray, float, np.ndarray]:
    """Parte C: Arredondamento por hiperplano aleatório repetido k vezes.

    Sorteia r ~ N(0, I_n) e define x_i = +1 se <v_i, r> >= 0, senão -1.

    Retorna:
        (best_cut, best_partition, first_cut, first_partition)
    """
    if k < 1:
        raise ValueError("k deve ser >= 1.")

    n = W.shape[0]
    if n == 0:
        empty = np.array([], dtype=int)
        return 0.0, empty, 0.0, empty

    rng = np.random.default_rng(seed)

    best_cut = -np.inf
    best_partition = np.ones(n, dtype=int)
    first_cut = 0.0
    first_partition = np.ones(n, dtype=int)

    for attempt in range(k):
        r = rng.standard_normal(n)
        projections = V @ r
        partition = np.where(projections >= 0.0, 1, -1).astype(int)
        current_cut = cut_value(W, partition)

        if attempt == 0:
            first_cut = current_cut
            first_partition = partition.copy()

        if current_cut > best_cut:
            best_cut = current_cut
            best_partition = partition.copy()

    return best_cut, best_partition, first_cut, first_partition


def goemans_williamson(
    W: np.ndarray,
    k: int = 100,
    seed: int | None = None,
    solver: str | None = None,
    return_details: bool = False,
) -> tuple[float, np.ndarray, float] | tuple[float, np.ndarray, float, dict[str, Any]]:
    """Executa o algoritmo completo de Goemans-Williamson.

    Parâmetros:
        W: Matriz de adjacência (n x n).
        k: Número de hiperplanos aleatórios sorteados (padrão: 100).
        seed: Semente para reprodutibilidade do arredondamento.
        solver: Solver específico do CVXPY (opcional).
        return_details: Se True, retorna também um dicionário com tempos
            separados (sdp_time, rounding_time), resultado k=1 e matriz X.

    Retorna:
        (best_cut, best_partition, sdp_value) ou
        (best_cut, best_partition, sdp_value, details)
    """
    t0_sdp = time.perf_counter()
    X, sdp_value = solve_max_cut_sdp(W, solver=solver)
    V = extract_unit_vectors(X)
    sdp_time = time.perf_counter() - t0_sdp

    t0_round = time.perf_counter()
    best_cut, best_partition, first_cut, first_partition = hyperplane_rounding(
        W, V, k=k, seed=seed
    )
    rounding_time = time.perf_counter() - t0_round

    if return_details:
        details: dict[str, Any] = {
            "k1_cut": first_cut,
            "k1_partition": first_partition,
            "sdp_time": sdp_time,
            "rounding_time": rounding_time,
            "total_time": sdp_time + rounding_time,
            "X": X,
        }
        return best_cut, best_partition, sdp_value, details

    return best_cut, best_partition, sdp_value