"""Testes de src/brute_force.py."""
import numpy as np
import pytest

from src.brute_force import brute_force_maxcut, brute_force_naive
from src.graphs import (
    complete_bipartite, complete_graph, cut_value, cycle_graph, empty_graph,
    known_optimum, random_graph,
)


@pytest.mark.parametrize("n", range(2, 13))
def test_grafo_completo(n):
    valor, _ = brute_force_maxcut(complete_graph(n))
    assert valor == known_optimum("complete", n)


@pytest.mark.parametrize("n", range(3, 13))
def test_ciclos(n):
    valor, _ = brute_force_maxcut(cycle_graph(n))
    assert valor == known_optimum("cycle", n)


@pytest.mark.parametrize("a,b", [(1, 1), (2, 3), (4, 4), (3, 6)])
def test_bipartido_completo(a, b):
    W = complete_bipartite(a, b)
    valor, _ = brute_force_maxcut(W)
    assert valor == a * b


def test_triangulo():
    assert brute_force_maxcut(complete_graph(3))[0] == 2


@pytest.mark.parametrize("seed", range(25))
def test_gray_code_igual_ao_ingenuo(seed):
    n = int(np.random.default_rng(seed).integers(2, 11))
    W = random_graph(n, 0.5, seed=seed, weighted=(seed % 2 == 0))
    rapido, _ = brute_force_maxcut(W)
    lento, _ = brute_force_naive(W)
    assert rapido == pytest.approx(lento)


@pytest.mark.parametrize("seed", range(10))
def test_x_devolvido_e_consistente(seed):
    W = random_graph(12, 0.5, seed=seed, weighted=True)
    valor, x = brute_force_maxcut(W)
    assert set(np.unique(x)) <= {-1, 1}
    assert x[0] == 1                                  # simetria: vértice 0 fixo
    assert cut_value(W, x) == pytest.approx(valor)    # x realmente atinge o valor


def test_otimo_nao_e_superado_por_cortes_aleatorios():
    W = random_graph(14, 0.5, seed=1, weighted=True)
    otimo, _ = brute_force_maxcut(W)
    rng = np.random.default_rng(0)
    for _ in range(500):
        x = rng.choice([-1, 1], size=14)
        assert cut_value(W, x) <= otimo + 1e-9


def test_otimo_limitado_pelo_peso_total():
    W = random_graph(12, 0.5, seed=2)
    assert brute_force_maxcut(W)[0] <= W.sum() / 2


# casos extremos
def test_grafo_sem_arestas():
    valor, x = brute_force_maxcut(empty_graph(6))
    assert valor == 0
    assert len(x) == 6


def test_n_zero_um_dois():
    assert brute_force_maxcut(np.zeros((0, 0)))[0] == 0
    assert brute_force_maxcut(np.zeros((1, 1)))[0] == 0
    W2 = np.array([[0, 3], [3, 0]], dtype=float)
    assert brute_force_maxcut(W2)[0] == 3


def test_pesos_zero_em_grafo_completo_de_indices():
    W = np.zeros((5, 5))
    assert brute_force_maxcut(W)[0] == 0


def test_protecao_de_tamanho():
    with pytest.raises(ValueError):
        brute_force_maxcut(np.zeros((40, 40)))