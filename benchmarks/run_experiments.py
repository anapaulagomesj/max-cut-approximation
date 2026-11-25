import csv
import time
from pathlib import Path
from src.graphs import random_graph
from src.brute_force import brute_force_maxcut
from src.random_cut import random_cut_best_of_k
from src.gw import (solve_max_cut_sdp, extract_unit_vectors, hyperplane_rounding,)

# Configurações dos experimentos
SIZES = [10, 14, 18, 20, 50, 100]
DENSITIES = [0.1, 0.5, 0.9]
SEEDS = [0, 1, 2, 3, 4]
K_VALUES = [1, 10, 50, 100]
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results"
MAX_EXACT_N = 20

# Colunas do arquivo de resultados
CSV_FIELDS = [
    "algorithm",
    "n",
    "p",
    "seed",
    "k",
    "cut_value",
    "reference_value",
    "ratio",
    "reference_type",
    "time_seconds",
    "sdp_time_seconds",
    "rounding_time_seconds",
]

# Salva os resultados dos experimentos em um arquivo CSV
def save_results(results, filename="benchmark_results.csv"):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / filename
    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(results)
    print(f"Resultados salvos em: {output_path}")

# Executa a força bruta e mede seu tempo de execução
def run_brute_force(W):
    start = time.perf_counter()
    best_value, _ = brute_force_maxcut(W)
    elapsed_time = time.perf_counter() - start
    return best_value, elapsed_time


# Executa o Random Cut e mede seu tempo de execução
def run_random_cut(W, k, seed):
    start = time.perf_counter()
    best_value, _ = random_cut_best_of_k(W, k=k, seed=seed)
    elapsed_time = time.perf_counter() - start
    return best_value, elapsed_time


# Resolve a SDP uma vez e prepara os vetores para o arredondamento
def run_goemans_williamson(W):
    start = time.perf_counter()
    X, sdp_value = solve_max_cut_sdp(W)
    V = extract_unit_vectors(X)
    sdp_time = time.perf_counter() - start
    return V, sdp_value, sdp_time


# Executa o arredondamento de hiperplano e mede seu tempo
def run_rounding(W, V, k, seed):
    start = time.perf_counter()
    best_value, _, _, _ = hyperplane_rounding(W, V, k=k, seed=seed)
    elapsed_time = time.perf_counter() - start
    return best_value, elapsed_time

# Executa os algoritmos em um grafo e registra seus resultados
def run_experiment(W, n, p, seed):
    results = []

    # Define o tipo de referência
    if n <= MAX_EXACT_N:
        reference_type = "optimum"
        optimum, brute_time = run_brute_force(W)
        results.append({
            "algorithm": "brute_force",
            "n": n,
            "p": p,
            "seed": seed,
            "k": None,
            "cut_value": optimum,
            "reference_value": optimum,
            "ratio": 1.0,
            "reference_type": reference_type,
            "time_seconds": brute_time,
        })
        reference_value = optimum

    else:
        reference_type = "sdp_bound"
        reference_value = None

    # Resolve a SDP uma única vez
    V, sdp_value, sdp_time = run_goemans_williamson(W)
    # Para grafos grandes, usa o limite SDP como referência
    if reference_value is None:
        reference_value = sdp_value
    # Executa os diferentes arredondamentos
    for k in K_VALUES:
        best_value, rounding_time = run_rounding(W, V, k, seed)
        total_time = sdp_time + rounding_time
        results.append({
            "algorithm": "goemans_williamson",
            "n": n,
            "p": p,
            "seed": seed,
            "k": k,
            "cut_value": best_value,
            "reference_value": reference_value,
            "ratio": best_value / reference_value if reference_value > 0 else None,
            "reference_type": reference_type,
            "time_seconds": total_time,
            "sdp_time_seconds": sdp_time,
            "rounding_time_seconds": rounding_time,
        })

    # Executa Random Cut
    for k in K_VALUES:
        best_value, elapsed_time = run_random_cut(W, k, seed)
        results.append({
            "algorithm": "random_cut",
            "n": n,
            "p": p,
            "seed": seed,
            "k": k,
            "cut_value": best_value,
            "reference_value": reference_value,
            "ratio": best_value / reference_value if reference_value > 0 else None,
            "reference_type": reference_type,
            "time_seconds": elapsed_time,
        })

    return results

# Executa todos os experimentos e salva os resultados
def run_all_experiments():
    all_results = []
    for n in SIZES:
        for p in DENSITIES:
            for seed in SEEDS:
                print(f"Executando: n={n}, p={p}, seed={seed}")
                W = random_graph(n, p, seed=seed)
                results = run_experiment(W, n, p, seed)
                all_results.extend(results)
    save_results(all_results)
    print(f"Total de registros: {len(all_results)}")

if __name__ == "__main__":
    run_all_experiments()