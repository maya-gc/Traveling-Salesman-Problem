import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DADOS_DIR = BASE_DIR / 'dados'
IMAGENS_DIR = BASE_DIR / 'imagens'
IMAGENS_DIR.mkdir(exist_ok=True)


def carregar_pontos(caminho):
    pts = []
    with open(caminho, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pts.append((float(row['x']), float(row['y'])))
    return np.array(pts, dtype=float)


def salvar_imagem_pontos(pontos, caminho, titulo):
    plt.figure(figsize=(7, 7))
    plt.scatter(pontos[:, 0], pontos[:, 1], s=40)
    for i, (x, y) in enumerate(pontos):
        plt.text(x + 2, y + 2, str(i), fontsize=8)
    plt.title(titulo)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(caminho, dpi=180)
    plt.close()


def main():
    for pasta_n in sorted(DADOS_DIR.glob('n_*')):
        n = pasta_n.name.replace('n_', '')
        out_dir = IMAGENS_DIR / pasta_n.name
        out_dir.mkdir(exist_ok=True)
        for arq in sorted(pasta_n.glob('instance_*.csv')):
            pontos = carregar_pontos(arq)
            nome = arq.stem + '.png'
            salvar_imagem_pontos(pontos, out_dir / nome, f'Pontos - n={n} - {arq.stem}')
            print(f'Imagem salva: {out_dir / nome}')


if __name__ == '__main__':
    main()