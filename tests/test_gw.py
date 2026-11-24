"""Testes unitários e teóricos para o algoritmo Goemans-Williamson (src/gw.py)."""
from __future__ import annotations

import numpy as np
import pytest

from src.graphs import (
    complete_bipartite,
    complete_graph,
    cut_value,
    cycle_graph,
    empty_graph,
    known_optimum,
    random_graph,
)
from src.gw import (
    extract_unit_vectors,
    goemans_williamson,
    hyperplane_rounding,
    solve_max_cut_sdp,
)


def test_sdp_matrix_properties():
    """Verifica se X é simétrica, PSD e tem diagonal unitária."""
    W = random_graph(n=10, p=0.5, seed=42)
    X, sdp_val = solve_max_cut_sdp(W)

    assert X.shape == (10, 10)
    assert np.allclose(X, X.T, atol=1e-7)
    assert np.allclose(np.diag(X), np.ones(10), atol=1e-6)
    eigvals = np.linalg.eigvalsh(X)
    assert np.min(eigvals) >= -1e-6
    assert sdp_val > 0.0


def test_unit_vectors_norm():
    """Verifica se os vetores fatorados têm norma 1 e reconstroem X."""
    W = random_graph(n=12, p=0.6, seed=7)
    X, _ = solve_max_cut_sdp(W)
    V = extract_unit_vectors(X)

    norms = np.linalg.norm(V, axis=1)
    assert np.allclose(norms, np.ones(12), atol=1e-6)
    assert np.allclose(V @ V.T, X, atol=1e-5)


def test_known_graphs():
    """Testa grafos com ótimo conhecido: K_{3,3}, triângulo C_3 e K_4."""
    # Bipartido completo K_{3,3}: todas as 9 arestas podem ser cortadas
    W_bip = complete_bipartite(3, 3)
    opt_bip = known_optimum("bipartite", 3, 3)
    cut_bip, part_bip, sdp_bip = goemans_williamson(W_bip, k=20, seed=10)
    assert np.isclose(sdp_bip, opt_bip, atol=1e-5)
    assert np.isclose(cut_bip, opt_bip, atol=1e-5)
    assert set(part_bip).issubset({-1, 1})

    # Triângulo C_3: OPT = 2, relaxamento SDP = 2.25
    W_tri = cycle_graph(3)
    opt_tri = known_optimum("cycle", 3)
    cut_tri, _, sdp_tri = goemans_williamson(W_tri, k=50, seed=10)
    assert np.isclose(sdp_tri, 2.25, atol=1e-5)
    assert np.isclose(cut_tri, opt_tri, atol=1e-5)

    # Grafo completo K_4: OPT = 4
    W_k4 = complete_graph(4)
    opt_k4 = known_optimum("complete", 4)
    cut_k4, _, sdp_k4 = goemans_williamson(W_k4, k=30, seed=10)
    assert cut_k4 <= opt_k4 <= sdp_k4 + 1e-5


def test_cut_never_exceeds_sdp():
    """Garante que o corte arredondado nunca ultrapassa o limitante SDP."""
    for seed in [1, 2, 3, 4, 5]:
        W = random_graph(n=12, p=0.5, seed=seed, weighted=True)
        gw_cut, gw_part, sdp_val = goemans_williamson(W, k=50, seed=seed)

        assert np.isclose(gw_cut, cut_value(W, gw_part))
        assert gw_cut <= sdp_val + 1e-5


def test_expected_approximation_ratio_0878():
    """Verifica a garantia esperada E[corte] / Z_SDP >= 0.878 no ciclo C_5."""
    W = cycle_graph(5)
    X, sdp_val = solve_max_cut_sdp(W)
    V = extract_unit_vectors(X)

    rng = np.random.default_rng(123)
    cuts = []
    for _ in range(1000):
        _, _, single_cut, _ = hyperplane_rounding(
            W, V, k=1, seed=int(rng.integers(0, 1_000_000))
        )
        cuts.append(single_cut)

    mean_ratio = float(np.mean(cuts)) / sdp_val
    assert mean_ratio >= 0.87


def test_reproducibility_and_timing_details():
    """Verifica reprodutibilidade com seed e medição separada de tempos."""
    W = random_graph(n=8, p=0.4, seed=99)

    cut1, part1, sdp1, det1 = goemans_williamson(W, k=30, seed=2026, return_details=True)
    cut2, part2, sdp2, det2 = goemans_williamson(W, k=30, seed=2026, return_details=True)

    assert cut1 == cut2
    assert np.array_equal(part1, part2)
    assert np.isclose(sdp1, sdp2)
    assert det1["k1_cut"] == det2["k1_cut"]
    assert det1["sdp_time"] >= 0.0
    assert det1["rounding_time"] >= 0.0
    assert det1["k1_cut"] <= cut1


def test_edge_cases():
    """Testa grafo vazio (sem arestas) e n=1."""
    W_zero = empty_graph(5)
    cut, part, sdp_val = goemans_williamson(W_zero, k=10, seed=1)
    assert cut == 0.0
    assert sdp_val == 0.0
    assert len(part) == 5

    W_one = empty_graph(1)
    cut1, part1, sdp1 = goemans_williamson(W_one, k=10, seed=1)
    assert cut1 == 0.0
    assert sdp1 == 0.0
    assert len(part1) == 1