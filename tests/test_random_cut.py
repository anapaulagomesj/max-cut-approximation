"""Testes de src/random_cut.py."""
import numpy as np
import pytest

from src.brute_force import brute_force_maxcut
from src.graphs import (
    complete_bipartite, complete_graph, cut_value, empty_graph, random_graph,
    total_weight,
)
from src.random_cut import expected_random_cut, random_cut, random_cut_best_of_k


def test_formato_do_retorno():
    W = random_graph(20, 0.5, seed=0)
    valor, x = random_cut(W, seed=0)
    assert x.shape == (20,)
    assert set(np.unique(x)) <= {-1, 1}
    assert cut_value(W, x) == pytest.approx(valor)


def test_reprodutivel_com_seed():
    W = random_graph(30, 0.5, seed=0)
    assert np.array_equal(random_cut(W, seed=5)[1], random_cut(W, seed=5)[1])
    assert not np.array_equal(random_cut(W, seed=5)[1], random_cut(W, seed=6)[1])


@pytest.mark.parametrize("seed", range(10))
def test_nunca_supera_o_otimo(seed):
    W = random_graph(12, 0.5, seed=seed, weighted=True)
    otimo, _ = brute_force_maxcut(W)
    for s in range(50):
        assert random_cut(W, seed=s)[0] <= otimo + 1e-9


def test_media_proxima_de_W_sobre_2():
    """Propriedade estatística: E[corte] = W_total / 2."""
    W = random_graph(40, 0.5, seed=1, weighted=True)
    media = np.mean([random_cut(W, seed=s)[0] for s in range(4000)])
    assert media == pytest.approx(expected_random_cut(W), rel=0.02)


def test_garantia_de_meia_aproximacao_em_media():
    """E[corte] >= OPT/2 (a garantia teórica do algoritmo)."""
    W = random_graph(14, 0.5, seed=4)
    otimo, _ = brute_force_maxcut(W)
    media = np.mean([random_cut(W, seed=s)[0] for s in range(2000)])
    assert media >= 0.5 * otimo


def test_expected_random_cut():
    W = complete_graph(5)
    assert expected_random_cut(W) == total_weight(W) / 2 == 5


def test_best_of_k_nao_pior_que_uma_tentativa():
    W = random_graph(30, 0.5, seed=2)
    unica, _ = random_cut(W, seed=0)
    melhor, _ = random_cut_best_of_k(W, k=200, seed=0)
    assert melhor >= unica


def test_best_of_k_melhora_com_k():
    W = random_graph(40, 0.5, seed=3)
    k1 = np.mean([random_cut_best_of_k(W, 1, seed=s)[0] for s in range(30)])
    k50 = np.mean([random_cut_best_of_k(W, 50, seed=s)[0] for s in range(30)])
    assert k50 > k1


def test_best_of_k_acha_otimo_em_grafo_pequeno():
    W = complete_bipartite(3, 3)
    valor, _ = random_cut_best_of_k(W, k=500, seed=0)
    assert valor == 9


def test_best_of_k_valida_k():
    with pytest.raises(ValueError):
        random_cut_best_of_k(complete_graph(4), k=0)


# ----- casos extremos -----
def test_grafo_sem_arestas():
    valor, x = random_cut(empty_graph(8), seed=0)
    assert valor == 0 and len(x) == 8


def test_n_zero_um_dois():
    assert random_cut(np.zeros((0, 0)), seed=0)[0] == 0
    assert random_cut(np.zeros((1, 1)), seed=0)[0] == 0
    W2 = np.array([[0, 3], [3, 0]], dtype=float)
    assert random_cut(W2, seed=0)[0] in (0, 3)


def test_pesos_zero():
    assert random_cut(np.zeros((6, 6)), seed=0)[0] == 0
