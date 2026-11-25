import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Caminhos dos arquivos
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
CSV_PATH = RESULTS_DIR / "benchmark_results.csv"
FIGURES_DIR = RESULTS_DIR / "figures"

# Carrega os resultados dos experimentos
def load_results():
    data = pd.read_csv(CSV_PATH)
    return data

# Gera boxplots comparando a qualidade das soluções
def plot_approximation_quality(data):
    # Considera apenas resultados com ótimo conhecido
    exact_data = data[
        (data["reference_type"] == "optimum") &
        (data["algorithm"].isin(["random_cut", "goemans_williamson"])) &
        (data["k"] == 100)
    ]

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sizes = sorted(exact_data["n"].unique())

    for p in sorted(exact_data["p"].unique()):
        density_data = exact_data[exact_data["p"] == p]
        plt.figure(figsize=(10, 6))
        random_values = []
        gw_values = []
        positions_random = []
        positions_gw = []

        for i, n in enumerate(sizes):
            random_values.append(
                density_data[
                    (density_data["algorithm"] == "random_cut") &
                    (density_data["n"] == n)
                ]["ratio"].dropna()
            )
            gw_values.append(
                density_data[
                    (density_data["algorithm"] == "goemans_williamson") &
                    (density_data["n"] == n)
                ]["ratio"].dropna()
            )
            positions_random.append(i + 1 - 0.18)
            positions_gw.append(i + 1 + 0.18)

        # Boxplots do Random Cut
        plt.boxplot(
            random_values,
            positions=positions_random,
            widths=0.3,
            patch_artist=True,
            boxprops=dict(facecolor="lightblue"),
            medianprops=dict(color="blue")
        )

        # Boxplots do Goemans-Williamson
        plt.boxplot(
            gw_values,
            positions=positions_gw,
            widths=0.3,
            patch_artist=True,
            boxprops=dict(facecolor="lightsalmon"),
            medianprops=dict(color="darkred")
        )

        # Garantias teóricas
        plt.axhline(
            y=0.5,
            color="blue",
            linestyle="--",
            label="Garantia Random Cut (0,5)"
        )
        plt.axhline(
            y=0.878,
            color="red",
            linestyle=":",
            label="Garantia GW (0,878)"
        )

        # Legenda dos algoritmos e garantias teóricas
        legend_elements = [
            Patch(facecolor="lightblue", label="Random Cut"),
            Patch(facecolor="lightsalmon", label="Goemans-Williamson"),
            Line2D([0], [0], color="blue", linestyle="--",
                   label="Garantia Random Cut (0,5)"),
            Line2D([0], [0], color="red", linestyle=":",
                   label="Garantia Goemans-Williamson (0,878)")
        ]

        plt.legend(handles=legend_elements, loc="lower right")
        plt.xticks(range(1, len(sizes) + 1), sizes)
        plt.xlabel("Número de vértices (n)")
        plt.ylabel("Razão em relação ao ótimo")
        plt.title(f"Qualidade das soluções (k=100) - densidade p={p}")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"quality_p_{p}.png")
        plt.close()


# Analisa o efeito do número de tentativas (k) na qualidade
def plot_k_effect(data):
    exact_data = data[
        (data["reference_type"] == "optimum") &
        (data["algorithm"].isin(["random_cut", "goemans_williamson"]))
    ]
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for p in sorted(exact_data["p"].unique()):
        density_data = exact_data[exact_data["p"] == p]
        plt.figure(figsize=(9, 6))

        for algorithm in ["random_cut", "goemans_williamson"]:
            algorithm_data = density_data[
                density_data["algorithm"] == algorithm
            ]
            averages = (
                algorithm_data
                .groupby("k")["ratio"]
                .mean()
                .sort_index()
            )
            plt.plot(
                averages.index,
                averages.values,
                marker="o",
                label=algorithm.replace("_", " ").title()
            )

        plt.xlabel("Número de tentativas (k)")
        plt.ylabel("Razão média em relação ao ótimo")
        plt.title(f"Efeito de k na qualidade média (n=10, 14, 18, 20) - p={p}")
        plt.xticks([1, 10, 50, 100])
        plt.ylim(0, 1.05)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"k_effect_p_{p}.png")
        plt.close()

# Analisa os tempos de execução dos algoritmos
def print_time_summary(data):
    print("\nResumo dos tempos de execução (segundos):")
    summary = (
        data.groupby(["algorithm", "n", "k"], dropna=False)
        .agg(
            mean_time=("time_seconds", "mean"),
            min_time=("time_seconds", "min"),
            max_time=("time_seconds", "max"),
            mean_sdp=("sdp_time_seconds", "mean"),
            mean_rounding=("rounding_time_seconds", "mean")
        )
        .round(6)
    )
    print(summary.to_string())


# Compara os tempos de execução dos algoritmos
def plot_execution_time(data):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Considera apenas k=100 nos algoritmos de aproximação
    selected_data = data[
        (data["algorithm"] == "brute_force") |
        (
            data["algorithm"].isin(["random_cut", "goemans_williamson"]) & (data["k"] == 100)
        )
    ]
    summary = (
        selected_data
        .groupby(["algorithm", "n"])["time_seconds"]
        .mean()
        .reset_index()
    )
    plt.figure(figsize=(10, 6))
    for algorithm in [
        "brute_force",
        "random_cut",
        "goemans_williamson"
    ]:
        algorithm_data = summary[
            summary["algorithm"] == algorithm
        ]
        plt.plot(
            algorithm_data["n"],
            algorithm_data["time_seconds"],
            marker="o",
            label=algorithm.replace("_", " ").title()
        )

    plt.yscale("log")
    plt.xlabel("Número de vértices (n)")
    plt.ylabel("Tempo médio de execução (segundos)")
    plt.title("Comparação do tempo de execução (aproximações com k=100)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "execution_time.png")
    plt.close()


# Gera uma tabela com estatísticas dos resultados
def generate_summary_table(data):
    # Agrupa os resultados por algoritmo, tamanho, densidade e k
    summary = (
        data.groupby(
            ["algorithm", "n", "p", "k", "reference_type"],
            dropna=False
        )
        .agg(
            mean_ratio=("ratio", "mean"),
            std_ratio=("ratio", "std"),
            worst_ratio=("ratio", "min"),
            best_ratio=("ratio", "max"),
            mean_time=("time_seconds", "mean")
        )
        .reset_index()
        .round(6)
    )

    # Salva a tabela em CSV
    output_path = RESULTS_DIR / "summary_statistics.csv"
    summary.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\nTabela estatística salva em: {output_path}")
    print(f"Total de grupos: {len(summary)}")


if __name__ == "__main__":
    results = load_results()

    print("Total de registros:", len(results))
    print("\nAlgoritmos encontrados:")
    print(results["algorithm"].unique())

    print("\nTipos de referência:")
    print(results["reference_type"].unique())

    plot_approximation_quality(results)
    print("\nGráficos de qualidade gerados com sucesso!")

    plot_k_effect(results)
    print("Gráficos do efeito de k gerados com sucesso!")

    plot_execution_time(results)
    print("Gráfico de comparação de tempo gerado com sucesso!")

    generate_summary_table(results)