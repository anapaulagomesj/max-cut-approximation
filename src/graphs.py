"""Geração de grafos e cálculo do valor de cortes (Max Cut).

Convenções usadas em todo o projeto
* Um grafo é representado por uma matriz de pesos ``W`` (numpy, n x n),
  simétrica, com diagonal zero. ``W[i, j] == 0`` significa que não há aresta.
* Um corte é um vetor ``x`` de tamanho n com valores em {-1, +1}, indicando
  o lado de cada vértice.
* Toda função de corte do projeto retorna ``(valor_do_corte, x)``.
"""
from __future__ import annotations

import numpy as np



# Validação e valor do corte
def validate_weight_matrix(W: np.ndarray) -> None:
    """Levanta ValueError se W não for uma matriz de pesos válida."""
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("W deve ser uma matriz quadrada.")

    if not np.allclose(W, W.T):
        raise ValueError("W deve ser simétrica.")

    if not np.allclose(np.diag(W), 0):
        raise ValueError("A diagonal de W deve ser zero (sem laços).")

def total_weight(W: np.ndarray) -> float:
    """Soma dos pesos de todas as arestas (cada aresta contada uma vez)."""
    return float(W.sum() / 2.0)

def cut_value(W: np.ndarray, x: np.ndarray) -> float:
    """Peso total das arestas que cruzam o corte definido por ``x``.

    Usa a identidade  cut = (1/4) * (sum(W) - x^T W x),  válida para
    x_i em {-1, +1}. Custo O(n^2).
    """
    x = np.asarray(x)
    if x.shape != (W.shape[0],):
        raise ValueError("x deve ter tamanho n.")
    if not np.all(np.abs(x) == 1):
        raise ValueError("x deve conter apenas valores -1 ou +1.")
    return float((W.sum() - x @ W @ x) / 4.0)


# Geradores aleatórios
def random_graph(
    n: int,
    p: float,
    seed: int | None = None,
    weighted: bool = False,
    weight_range: tuple[int, int] = (1, 10),
) -> np.ndarray:
    """Grafo aleatório de Erdős–Rényi G(n, p).

    Cada uma das n(n-1)/2 arestas possíveis existe com probabilidade ``p``.

    Parâmetros:
    n : número de vértices.
    p : probabilidade de existir cada aresta (0 <= p <= 1).
    seed : semente, para reprodutibilidade.
    weighted : se True, pesos inteiros uniformes em ``weight_range``
        (inclusive); se False, todos os pesos são 1.
    """
    if n < 0:
        raise ValueError("n deve ser >= 0.")
    if not 0.0 <= p <= 1.0:
        raise ValueError("p deve estar em [0, 1].")

    rng = np.random.default_rng(seed)
    iu = np.triu_indices(n, k=1)
    exists = rng.random(len(iu[0])) < p

    if weighted:
        lo, hi = weight_range
        weights = rng.integers(lo, hi + 1, size=len(iu[0])).astype(float)
    else:
        weights = np.ones(len(iu[0]))

    W = np.zeros((n, n))
    W[iu] = np.where(exists, weights, 0.0)
    return W + W.T


def sparse_graph(n: int, avg_degree: float = 3.0, seed: int | None = None,
                 weighted: bool = False) -> np.ndarray:
    """Grafo esparso: G(n, p) com p = avg_degree / (n - 1) (grau médio constante)."""
    p = min(1.0, avg_degree / max(n - 1, 1))
    return random_graph(n, p, seed=seed, weighted=weighted)


def dense_graph(n: int, p: float = 0.5, seed: int | None = None,
                weighted: bool = False) -> np.ndarray:
    """Grafo denso: G(n, p) com p alto (padrão 0.5)."""
    return random_graph(n, p, seed=seed, weighted=weighted)


# Grafos com resposta conhecida
def empty_graph(n: int) -> np.ndarray:
    """Grafo sem arestas. OPT = 0."""
    return np.zeros((n, n))


def complete_graph(n: int) -> np.ndarray:
    """K_n. OPT = floor(n/2) * ceil(n/2)."""
    return np.ones((n, n)) - np.eye(n)


def complete_bipartite(a: int, b: int) -> np.ndarray:
    """K_{a,b}. OPT = a * b (todas as arestas)."""
    n = a + b
    W = np.zeros((n, n))
    W[:a, a:] = 1.0
    W[a:, :a] = 1.0
    return W


def cycle_graph(n: int) -> np.ndarray:
    """Ciclo C_n (n >= 3). OPT = n se n é par, n - 1 se n é ímpar."""
    if n < 3:
        raise ValueError("Ciclo exige n >= 3.")
    W = np.zeros((n, n))
    for i in range(n):
        j = (i + 1) % n
        W[i, j] = W[j, i] = 1.0
    return W


def known_optimum(kind: str, *params: int) -> float:
    """Valor ótimo do corte máximo para os grafos de resposta conhecida.

    kind: 'empty', 'complete', 'bipartite', 'cycle'.
    """
    if kind == "empty":
        return 0.0
    if kind == "complete":
        (n,) = params
        return float((n // 2) * (n - n // 2))
    if kind == "bipartite":
        a, b = params
        return float(a * b)
    if kind == "cycle":
        (n,) = params
        return float(n if n % 2 == 0 else n - 1)
    raise ValueError(f"Tipo desconhecido: {kind}")
