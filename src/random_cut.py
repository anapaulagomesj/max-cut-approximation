"""Aproximação aleatorizada 0.5 para Max Cut.

Cada vértice é colocado em um dos lados com probabilidade 1/2,
independentemente. Como cada aresta (i, j) cruza o corte com probabilidade
1/2 (x_i != x_j), temos
        E[corte] = W_total / 2  >=  OPT / 2,
já que OPT <= W_total. Custo: O(n) para sortear + O(n^2) para avaliar.
"""
from __future__ import annotations

import numpy as np

from .graphs import cut_value, total_weight


def random_cut(W: np.ndarray, seed: int | None = None,
               rng: np.random.Generator | None = None) -> tuple[float, np.ndarray]:
    """Uma única tentativa (a versão 'pura' do algoritmo).

    Passe ``seed`` ou um ``rng`` já criado (útil para várias chamadas
    encadeadas sem repetir a mesma sequência).
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    n = W.shape[0]
    x = rng.choice(np.array([-1, 1]), size=n)
    return cut_value(W, x), x


def random_cut_best_of_k(W: np.ndarray, k: int,
                         seed: int | None = None) -> tuple[float, np.ndarray]:
    """Repete ``random_cut`` k vezes e devolve o melhor corte encontrado."""
    if k < 1:
        raise ValueError("k deve ser >= 1.")
    rng = np.random.default_rng(seed)
    best_value, best_x = -1.0, None
    for _ in range(k):
        value, x = random_cut(W, rng=rng)
        if value > best_value:
            best_value, best_x = value, x
    return best_value, best_x


def expected_random_cut(W: np.ndarray) -> float:
    """Valor esperado teórico do corte aleatório: W_total / 2."""
    return total_weight(W) / 2.0
