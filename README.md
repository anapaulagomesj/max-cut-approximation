# Max-Cut: Algoritmos Exatos e de Aproximação

Implementação e análise experimental de algoritmos para o problema **Max-Cut**, que consiste em dividir os vértices de um grafo em dois conjuntos, maximizando o peso das arestas que conectam vértices pertencentes a conjuntos diferentes.

## Algoritmos implementados

O projeto implementa e compara três abordagens:

- **Força Bruta (Brute Force):** algoritmo exato que explora as possíveis partições do grafo para encontrar o corte máximo.
- **Random Cut:** algoritmo aleatorizado que gera diferentes cortes e seleciona o de maior valor entre as tentativas.
- **Goemans-Williamson:** algoritmo de aproximação baseado em programação semidefinida (SDP) e arredondamento por hiperplanos aleatórios.

## Estrutura do projeto

```text
max-cut-approximation/
├── src/
│   ├── brute_force.py
│   ├── graphs.py
│   ├── gw.py
│   └── random_cut.py
├── tests/
│   ├── test_brute_force.py
│   ├── test_graphs.py
│   ├── test_gw.py
│   └── test_random_cut.py
├── benchmarks/
│   ├── run_experiments.py
│   └── make_plots.py
├── results/
│   ├── benchmark_results.csv
│   ├── summary_statistics.csv
│   └── figures/
├── README.md
├── requirements.txt
└── pytest.ini
```

## Instalação

É recomendado utilizar um ambiente virtual Python.

Instale as dependências com:

```bash
pip install -r requirements.txt
```

As principais bibliotecas utilizadas são:

- NumPy
- CVXPY
- Clarabel
- SCS
- Pytest

## Testes automatizados

Para executar os testes:

```bash
python -m pytest -v
```

Os testes verificam o funcionamento dos algoritmos e das funções de geração de grafos.

## Experimentos

Para executar os experimentos:

```bash
python -m benchmarks.run_experiments
```

Os experimentos utilizam grafos aleatórios com as seguintes configurações:

| Parâmetro | Valores |
|---|---|
| Número de vértices (n) | 10, 14, 18, 20, 50 e 100 |
| Densidade (p) | 0,1; 0,5 e 0,9 |
| Sementes aleatórias | 0, 1, 2, 3 e 4 |
| Número de tentativas (k) | 1, 10, 50 e 100 |

A força bruta é executada somente para grafos com até 20 vértices, devido ao seu crescimento exponencial.

Para grafos com até 20 vértices, o valor ótimo encontrado pela força bruta é utilizado como referência. Para grafos maiores, utiliza-se o valor da relaxação SDP como limite de referência.

Os resultados são armazenados em `results/benchmark_results.csv`.

## Análise dos resultados

Para gerar os gráficos e a tabela de estatísticas:

```bash
python -m benchmarks.make_plots
```

São gerados:

- **Gráficos de qualidade:** comparam as razões das soluções dos algoritmos em relação ao ótimo conhecido, para cada densidade, utilizando k = 100.
- **Gráficos do efeito de k:** analisam como o aumento do número de tentativas influencia a qualidade média das soluções.
- **Gráfico de tempo de execução:** compara os tempos médios dos algoritmos, utilizando k = 100 para os algoritmos de aproximação.
- **Tabela de estatísticas:** reúne média, desvio padrão, pior resultado, melhor resultado e tempo médio de execução.

Os gráficos são salvos em `results/figures/`, e a tabela estatística em `results/summary_statistics.csv`.

## Garantias de aproximação

Os gráficos de qualidade também apresentam as garantias teóricas de aproximação:

- **Random Cut:** fator de aproximação esperado de 0,5.
- **Goemans-Williamson:** fator de aproximação esperado de aproximadamente 0,878.

Essas garantias são teóricas e não representam limites obrigatórios para cada execução individual dos algoritmos aleatorizados.

## Tecnologias

- Python
- NumPy
- CVXPY
- Clarabel
- SCS
- Matplotlib
- Pandas
- Pytest