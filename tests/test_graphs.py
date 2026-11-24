"""Testes de src/graphs.py."""
import numpy as np
import pytest

from src.graphs import (
    complete_bipartite, complete_graph, cut_value, cycle_graph, dense_graph,
    empty_graph, known_optimum, random_graph, sparse_graph, total_weight,
    validate_weight_matrix,
)


def test_cut_value_triangulo():
    W = complete_graph(3)
    assert cut_value(W, np.array([1, 1, 1])) == 0       # todos do mesmo lado
    assert cut_value(W, np.array([1, 1, -1])) == 2      # 2 arestas cruzam
    assert cut_value(W, np.array([-1, 1, 1])) == 2


def test_cut_value_bipartido_corta_tudo():
    W = complete_bipartite(3, 4)
    x = np.array([1, 1, 1, -1, -1, -1, -1])
    assert cut_value(W, x) == total_weight(W) == 12


def test_cut_value_simetria_troca_de_lados():
    W = random_graph(15, 0.5, seed=3, weighted=True)
    x = np.random.default_rng(0).choice([-1, 1], size=15)
    assert cut_value(W, x) == pytest.approx(cut_value(W, -x))


def test_cut_value_com_pesos():
    W = np.array([[0, 5, 2], [5, 0, 0], [2, 0, 0]], dtype=float)
    assert cut_value(W, np.array([1, -1, 1])) == 5
    assert cut_value(W, np.array([1, -1, -1])) == 7


def test_cut_value_entrada_invalida():
    W = complete_graph(3)
    with pytest.raises(ValueError):
        cut_value(W, np.array([1, 0, -1]))      # 0 não é lado válido
    with pytest.raises(ValueError):
        cut_value(W, np.array([1, -1]))         # tamanho errado


def test_total_weight():
    assert total_weight(complete_graph(5)) == 10
    assert total_weight(empty_graph(5)) == 0


@pytest.mark.parametrize("weighted", [False, True])
def test_random_graph_valido(weighted):
    W = random_graph(30, 0.4, seed=1, weighted=weighted)
    validate_weight_matrix(W)
    assert W.shape == (30, 30)
    assert np.all(W >= 0)


def test_random_graph_reprodutivel():
    a = random_graph(25, 0.5, seed=42, weighted=True)
    b = random_graph(25, 0.5, seed=42, weighted=True)
    c = random_graph(25, 0.5, seed=43, weighted=True)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_random_graph_extremos_de_p():
    assert total_weight(random_graph(10, 0.0, seed=0)) == 0
    assert total_weight(random_graph(10, 1.0, seed=0)) == 45   # K_10


def test_random_graph_densidade_aproximada():
    n, p = 200, 0.3
    W = random_graph(n, p, seed=7)
    densidade = total_weight(W) / (n * (n - 1) / 2)
    assert densidade == pytest.approx(p, abs=0.03)


def test_random_graph_pesos_no_intervalo():
    W = random_graph(30, 1.0, seed=0, weighted=True, weight_range=(3, 7))
    pesos = W[np.triu_indices(30, k=1)]
    assert pesos.min() >= 3 and pesos.max() <= 7


def test_random_graph_parametros_invalidos():
    with pytest.raises(ValueError):
        random_graph(5, 1.5)
    with pytest.raises(ValueError):
        random_graph(-1, 0.5)


def test_esparso_tem_menos_arestas_que_denso():
    assert total_weight(sparse_graph(100, seed=0)) < total_weight(dense_graph(100, 0.5, seed=0))


def test_grau_medio_esparso():
    n, d = 300, 3.0
    W = sparse_graph(n, avg_degree=d, seed=5)
    assert 2 * total_weight(W) / n == pytest.approx(d, abs=0.5)


def test_casos_extremos_n_pequeno():
    for n in (0, 1, 2):
        W = random_graph(n, 0.5, seed=0)
        assert W.shape == (n, n)
        validate_weight_matrix(W)


def test_validate_rejeita_matrizes_ruins():
    with pytest.raises(ValueError):
        validate_weight_matrix(np.array([[0, 1], [2, 0]], dtype=float))   # assimétrica
    with pytest.raises(ValueError):
        validate_weight_matrix(np.array([[1, 0], [0, 0]], dtype=float))   # diagonal != 0
    with pytest.raises(ValueError):
        validate_weight_matrix(np.zeros((2, 3)))                          # não quadrada


def test_known_optimum_formulas():
    assert known_optimum("complete", 5) == 6
    assert known_optimum("complete", 6) == 9
    assert known_optimum("bipartite", 3, 4) == 12
    assert known_optimum("cycle", 5) == 4
    assert known_optimum("cycle", 6) == 6
    assert known_optimum("empty") == 0
    with pytest.raises(ValueError):
        known_optimum("estrela", 3)


def test_cycle_exige_n_minimo():
    with pytest.raises(ValueError):
        cycle_graph(2)
