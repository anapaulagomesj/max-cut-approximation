"""Max Cut exato por força bruta, O(2^n) com atualização incremental.

Ideias de projeto:
1. **Simetria:** trocar todos os lados (x -> -x) não altera o corte. Fixamos
   o vértice 0 no lado +1 e enumeramos só 2^(n-1) partições, sem perder
   nenhuma solução ótima.
2. **Código de Gray:** partições consecutivas diferem em um único vértice.
   Em vez de recalcular o corte (O(n^2)), atualizamos o valor em O(n):
   se o vértice v troca de lado, a variação é
        delta = x_v * (W[v] @ x)     (calculada antes de inverter x_v)
   pois arestas para vértices do mesmo lado passam a cruzar (+w) e as que
   cruzavam deixam de cruzar (-w).
Custo total: O(n * 2^(n-1)).
"""
from __future__ import annotations

import numpy as np

from .graphs import cut_value

MAX_N = 26  # proteção contra execuções que durariam horas


def brute_force_maxcut(W: np.ndarray, max_n: int = MAX_N) -> tuple[float, np.ndarray]:
    """Corte máximo exato.

    Retorna ``(valor_otimo, x)`` com x em {-1, +1}^n e x[0] = +1.
    Levanta ValueError se n > max_n.
    """
    n = W.shape[0]
    if n > max_n:
        raise ValueError(f"n={n} > max_n={max_n}: força bruta inviável.")
    if n == 0:
        return 0.0, np.array([], dtype=int)
    if n == 1:
        return 0.0, np.array([1])

    x = np.ones(n)               # começa com todos no lado +1 (corte = 0)
    value = 0.0
    best_value = 0.0
    best_x = x.copy()

    for k in range(1, 2 ** (n - 1)):
        # bit menos significativo de k = posição do bit que muda no Gray code
        v = (k & -k).bit_length()   # = índice do bit + 1 (pula o vértice 0)
        value += x[v] * (W[v] @ x)
        x[v] = -x[v]
        if value > best_value:
            best_value = value
            best_x = x.copy()

    return float(best_value), best_x.astype(int)


def brute_force_naive(W: np.ndarray) -> tuple[float, np.ndarray]:
    """Versão ingênua O(n^2 * 2^n), só para validar a versão com Gray code."""
    n = W.shape[0]
    best_value, best_x = -1.0, None
    for mask in range(2 ** n):
        x = np.array([1 if (mask >> i) & 1 else -1 for i in range(n)])
        val = cut_value(W, x)
        if val > best_value:
            best_value, best_x = val, x
    return float(best_value), best_x
