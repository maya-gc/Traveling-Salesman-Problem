from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.stats import wilcoxon
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

BASE_DIR = Path(__file__).resolve().parent
RESULTADOS_DIR = BASE_DIR / "resultados"


def bootstrap_ci(values, n_boot=2000, seed=123):
    vals = np.asarray([v for v in values if pd.notna(v)], dtype=float)
    if len(vals) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(vals, size=len(vals), replace=True)
        means.append(sample.mean())
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def save_line_chart(df_plot, x, y, hue, title, ylabel, out_path):
    plt.figure(figsize=(10, 6))
    for algo, g in df_plot.groupby(hue):
        g = g.sort_values(x)
        plt.plot(g[x], g[y], marker='o', linewidth=2, label=algo)
    plt.title(title)
    plt.xlabel('Número de pontos')
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_boxplot(df, n_value, metric, out_path, title):
    sub = df[df['n_points'] == n_value].copy()
    algos = sorted(sub['algorithm'].unique())
    data = [sub[sub['algorithm'] == a][metric].dropna().values for a in algos]
    plt.figure(figsize=(10, 6))
    plt.boxplot(data, labels=algos, showfliers=False)
    plt.title(title)
    plt.ylabel(metric)
    plt.xticks(rotation=15)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main():
    df = pd.read_csv(RESULTADOS_DIR / 'raw_runs.csv')

    summary = (
        df.groupby(['n_points', 'algorithm'])
        .agg(
            runs=('distance_final', 'size'),
            mean_distance=('distance_final', 'mean'),
            median_distance=('distance_final', 'median'),
            std_distance=('distance_final', 'std'),
            min_distance=('distance_final', 'min'),
            max_distance=('distance_final', 'max'),
            mean_runtime_s=('runtime_total_s', 'mean'),
            median_runtime_s=('runtime_total_s', 'median'),
            std_runtime_s=('runtime_total_s', 'std'),
            mean_gap_to_best_pct=('gap_to_best_pct', 'mean'),
            median_gap_to_best_pct=('gap_to_best_pct', 'median'),
            mean_improvement_pct=('improvement_pct', 'mean'),
        )
        .reset_index()
    )

    cis = []
    for (n, algo), g in df.groupby(['n_points', 'algorithm']):
        lo, hi = bootstrap_ci(g['distance_final'].tolist(), seed=42 + int(n))
        cis.append({
            'n_points': n,
            'algorithm': algo,
            'distance_mean_ci95_low': lo,
            'distance_mean_ci95_high': hi,
        })

    ci_df = pd.DataFrame(cis)
    summary = summary.merge(ci_df, on=['n_points', 'algorithm'], how='left')
    summary.to_csv(RESULTADOS_DIR / 'summary_by_algorithm.csv', index=False)

    tabela_mediana = summary.pivot(index='n_points', columns='algorithm', values='median_distance').reset_index()
    tabela_runtime = summary.pivot(index='n_points', columns='algorithm', values='median_runtime_s').reset_index()

    tabela_mediana.to_csv(RESULTADOS_DIR / 'median_distance_table.csv', index=False)
    tabela_runtime.to_csv(RESULTADOS_DIR / 'median_runtime_table.csv', index=False)

    tabela_simplificada = summary[[
        'n_points', 'algorithm', 'runs', 'median_distance', 'median_runtime_s', 'median_gap_to_best_pct'
    ]].copy()
    tabela_simplificada = tabela_simplificada.sort_values(['n_points', 'algorithm'])
    tabela_simplificada.to_csv(RESULTADOS_DIR / 'tabela_simplificada.csv', index=False)

    testes = []
    if SCIPY_AVAILABLE:
        for n in sorted(df['n_points'].unique()):
            sub = df[df['n_points'] == n]
            algos = sorted(sub['algorithm'].unique())
            for i in range(len(algos)):
                for j in range(i + 1, len(algos)):
                    a = algos[i]
                    b = algos[j]
                    xa = sub[sub['algorithm'] == a]['distance_final'].dropna().reset_index(drop=True)
                    xb = sub[sub['algorithm'] == b]['distance_final'].dropna().reset_index(drop=True)
                    m = min(len(xa), len(xb))
                    if m >= 10:
                        try:
                            stat, p = wilcoxon(xa.iloc[:m], xb.iloc[:m])
                        except Exception:
                            stat, p = np.nan, np.nan
                    else:
                        stat, p = np.nan, np.nan
                    testes.append({
                        'n_points': n,
                        'algo_a': a,
                        'algo_b': b,
                        'wilcoxon_stat': stat,
                        'p_value': p,
                        'paired_n': m,
                    })

    pd.DataFrame(testes).to_csv(RESULTADOS_DIR / 'wilcoxon_tests.csv', index=False)

    save_line_chart(summary, 'n_points', 'median_distance', 'algorithm', 'Mediana da distância por algoritmo', 'Distância', RESULTADOS_DIR / 'grafico_mediana_distancia.png')
    save_line_chart(summary, 'n_points', 'median_runtime_s', 'algorithm', 'Mediana do tempo por algoritmo', 'Tempo (s)', RESULTADOS_DIR / 'grafico_mediana_tempo.png')
    save_line_chart(summary, 'n_points', 'median_gap_to_best_pct', 'algorithm', 'Gap mediano para o melhor resultado', 'Gap (%)', RESULTADOS_DIR / 'grafico_gap_mediano.png')

    max_n = int(df['n_points'].max())
    save_boxplot(df, max_n, 'distance_final', RESULTADOS_DIR / f'boxplot_distance_n{max_n}.png', f'Distribuição de distância final em n={max_n}')
    save_boxplot(df, max_n, 'runtime_total_s', RESULTADOS_DIR / f'boxplot_runtime_n{max_n}.png', f'Distribuição de tempo total em n={max_n}')

    print('Análise concluída com gráficos e tabela simplificada.')


if __name__ == '__main__':
    main()