import csv
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DADOS_DIR = BASE_DIR / "dados"
DADOS_DIR.mkdir(exist_ok=True)

LIM_SUP = 1000.0
BASE_SEED = 42


def gerar_pontos(n: int, lim_sup: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    xs = rng.uniform(0, lim_sup, size=n)
    ys = rng.uniform(0, lim_sup, size=n)
    return np.column_stack((xs, ys)).astype(np.float32)


def salvar_csv(pontos: np.ndarray, caminho: Path) -> None:
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "x", "y"])
        for i, (x, y) in enumerate(pontos):
            writer.writerow([i, float(x), float(y)])


def main():
    tamanhos = [10, 20, 30, 40, 50, 100, 500]
    instancias_por_tamanho = 10

    for n in tamanhos:
        pasta_n = DADOS_DIR / f"n_{n}"
        pasta_n.mkdir(exist_ok=True)

        for inst_idx in range(instancias_por_tamanho):
            seed = BASE_SEED + n * 1000 + inst_idx
            pontos = gerar_pontos(n, LIM_SUP, seed)
            caminho = pasta_n / f"instance_{inst_idx:02d}.csv"
            salvar_csv(pontos, caminho)

    print("Instâncias geradas com sucesso.")


if __name__ == "__main__":
    main()