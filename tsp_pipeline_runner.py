import csv
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import cupy as cp
    GPU_AVAILABLE = True
except Exception:
    cp = None
    GPU_AVAILABLE = False

BASE_DIR = Path(__file__).resolve().parent
DADOS_DIR = BASE_DIR / "dados"
RESULTADOS_DIR = BASE_DIR / "resultados"
RESULTADOS_DIR.mkdir(exist_ok=True)

BASE_SEED = 42
LIM_SUP = 1000.0
TAMANHOS = [10, 20, 30, 40, 50, 100, 500]
INSTANCIAS_POR_TAMANHO = 5
ML_REPEATS = 5
CLASSIC_REPEATS_MAX50 = 20
CHUNK_PAIRS = 2048


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if GPU_AVAILABLE:
        cp.random.seed(seed)


def carregar_pontos(caminho):
    pontos = []
    with open(caminho, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pontos.append((float(row["x"]), float(row["y"])))
    return np.array(pontos, dtype=np.float32)


def dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def matriz_distancias_cpu(pontos):
    diff = pontos[:, None, :] - pontos[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=2)).astype(np.float32)


def rota_distancia_cpu(rota, dist_mat):
    idx = np.array(rota, dtype=np.int32)
    prox = np.roll(idx, -1)
    return float(dist_mat[idx, prox].sum())


def nearest_neighbor_cpu(pontos, start=0):
    n = len(pontos)
    nao_visitados = set(range(n))
    rota = [start]
    nao_visitados.remove(start)
    atual = start
    while nao_visitados:
        prox = min(nao_visitados, key=lambda j: dist(pontos[atual], pontos[j]))
        rota.append(prox)
        nao_visitados.remove(prox)
        atual = prox
    return rota


def two_opt_cpu(rota, dist_mat):
    melhor = rota[:]
    melhor_dist = rota_distancia_cpu(melhor, dist_mat)
    n = len(melhor)
    melhorado = True
    while melhorado:
        melhorado = False
        for i in range(1, n - 2):
            for j in range(i + 2, n):
                a = melhor[i - 1]
                b = melhor[i]
                c = melhor[j]
                d = melhor[(j + 1) % n]
                ganho = (dist_mat[a, b] + dist_mat[c, d]) - (dist_mat[a, c] + dist_mat[b, d])
                if ganho > 1e-10:
                    melhor[i:j + 1] = reversed(melhor[i:j + 1])
                    melhor_dist -= float(ganho)
                    melhorado = True
                    break
            if melhorado:
                break
    return melhor, float(melhor_dist)


def run_classic_cpu(pontos, seed):
    dist_mat = matriz_distancias_cpu(pontos)
    start = seed % len(pontos)
    t0 = time.perf_counter()
    rota_nn = nearest_neighbor_cpu(pontos, start=start)
    dist_inicial = rota_distancia_cpu(rota_nn, dist_mat)
    _, dist_final = two_opt_cpu(rota_nn, dist_mat)
    t1 = time.perf_counter()
    return {
        "distance_initial": float(dist_inicial),
        "distance_final": float(dist_final),
        "runtime_total_s": t1 - t0,
        "runtime_setup_s": 0.0,
        "runtime_solve_s": t1 - t0,
        "device": "cpu"
    }


def matriz_distancias_numpy(pontos_np):
    diff = pontos_np[:, None, :] - pontos_np[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=2)).astype(np.float32)


def matriz_distancias_gpu(pontos_np):
    return cp.asarray(matriz_distancias_numpy(pontos_np))


def nearest_neighbor_gpu(dist_mat, start=0):
    n = dist_mat.shape[0]
    visitado = cp.zeros(n, dtype=cp.bool_)
    rota = []
    atual = int(start)
    visitado[atual] = True
    rota.append(atual)
    for _ in range(n - 1):
        linha = dist_mat[atual].copy()
        linha[visitado] = cp.inf
        prox = int(cp.argmin(linha))
        visitado[prox] = True
        rota.append(prox)
        atual = prox
    return rota


def rota_distancia_gpu(rota, dist_mat):
    rota_gpu = cp.array(rota, dtype=cp.int32)
    prox = cp.roll(rota_gpu, -1)
    return float(dist_mat[rota_gpu, prox].sum())


def gerar_pares_gpu(n):
    i_list, j_list = [], []
    for i in range(1, n - 2):
        for j in range(i + 2, n):
            i_list.append(i)
            j_list.append(j)
    return cp.array(i_list, dtype=cp.int32), cp.array(j_list, dtype=cp.int32)


def two_opt_gpu(rota, dist_mat):
    n = len(rota)
    melhor = list(rota)
    melhor_dist = rota_distancia_gpu(melhor, dist_mat)
    i_gpu, j_gpu = gerar_pares_gpu(n)
    total_pares = i_gpu.shape[0]
    melhorado = True
    while melhorado:
        melhorado = False
        rota_gpu = cp.array(melhor, dtype=cp.int32)
        melhor_ganho = cp.float32(1e-10)
        melhor_i = -1
        melhor_j = -1
        for start in range(0, total_pares, CHUNK_PAIRS):
            end = min(start + CHUNK_PAIRS, total_pares)
            i_c = i_gpu[start:end]
            j_c = j_gpu[start:end]
            a = rota_gpu[i_c - 1]
            b = rota_gpu[i_c]
            c = rota_gpu[j_c]
            d = rota_gpu[(j_c + 1) % n]
            ganho = (dist_mat[a, b] + dist_mat[c, d]) - (dist_mat[a, c] + dist_mat[b, d])
            idx_local = int(cp.argmax(ganho))
            g_local = float(ganho[idx_local])
            if g_local > float(melhor_ganho):
                melhor_ganho = cp.float32(g_local)
                melhor_i = int(i_c[idx_local])
                melhor_j = int(j_c[idx_local])
        if melhor_i >= 0:
            melhor[melhor_i:melhor_j + 1] = reversed(melhor[melhor_i:melhor_j + 1])
            melhor_dist -= float(melhor_ganho)
            melhorado = True
    return melhor, float(melhor_dist)


def run_classic_gpu(pontos, seed):
    cp.get_default_memory_pool().free_all_blocks()
    t0 = time.perf_counter()
    dist_np = matriz_distancias_numpy(pontos)
    t_setup = time.perf_counter()
    dist_mat = cp.asarray(dist_np)
    start = seed % len(pontos)
    rota_nn = nearest_neighbor_gpu(dist_mat, start=start)
    dist_inicial = rota_distancia_gpu(rota_nn, dist_mat)
    _, dist_final = two_opt_gpu(rota_nn, dist_mat)
    cp.cuda.Stream.null.synchronize()
    t1 = time.perf_counter()
    return {
        "distance_initial": float(dist_inicial),
        "distance_final": float(dist_final),
        "runtime_total_s": t1 - t0,
        "runtime_setup_s": t_setup - t0,
        "runtime_solve_s": t1 - t_setup,
        "device": "gpu"
    }


def run_ml_cpu(pontos, seed, episodios=1200, alpha=0.1, gamma=0.9, epsilon=0.25):
    set_seed(seed)
    dist_mat = matriz_distancias_cpu(pontos)
    n = dist_mat.shape[0]
    q = np.zeros((n, n), dtype=float)
    melhor_dist = float("inf")
    t0 = time.perf_counter()
    for ep in range(episodios):
        inicio = random.randrange(n)
        nao_visitadas = set(range(n))
        nao_visitadas.remove(inicio)
        atual = inicio
        total = 0.0
        while nao_visitadas:
            if random.random() < epsilon:
                prox = random.choice(list(nao_visitadas))
            else:
                idxs = list(nao_visitadas)
                valores = [q[atual, j] for j in idxs]
                prox = idxs[int(np.argmax(valores))]
            recompensa = -dist_mat[atual, prox]
            total += dist_mat[atual, prox]
            antigo = q[atual, prox]
            futuro = max([q[prox, j] for j in nao_visitadas if j != prox], default=0.0)
            q[atual, prox] = antigo + alpha * (recompensa + gamma * futuro - antigo)
            nao_visitadas.remove(prox)
            atual = prox
        total += dist_mat[atual, inicio]
        if total < melhor_dist:
            melhor_dist = total
        if ep % 300 == 0 and ep > 0:
            epsilon = max(0.05, epsilon * 0.92)
    t1 = time.perf_counter()
    return {
        "distance_initial": np.nan,
        "distance_final": float(melhor_dist),
        "runtime_total_s": t1 - t0,
        "runtime_setup_s": 0.0,
        "runtime_solve_s": t1 - t0,
        "device": "cpu",
        "episodes": episodios,
        "alpha": alpha,
        "gamma": gamma,
        "epsilon_init": 0.25
    }


def run_ml_gpu(pontos, seed, n_chains=512, episodios=30, alpha=0.12, gamma=0.92,
               epsilon=0.35, epsilon_decay=0.94, epsilon_min=0.02, decay_freq=5):
    set_seed(seed)
    cp.get_default_memory_pool().free_all_blocks()
    t0 = time.perf_counter()
    dist_np = matriz_distancias_numpy(pontos)
    t_setup = time.perf_counter()
    dist_mat = cp.asarray(dist_np)
    n = int(dist_mat.shape[0])
    INF = cp.float32(1e30)
    Q = cp.zeros((n, n), dtype=cp.float32)
    melhor_dist = float("inf")
    arange = cp.arange(n_chains)
    eps = epsilon
    for ep in range(episodios):
        if ep > 0 and ep % decay_freq == 0:
            eps = max(epsilon_min, eps * epsilon_decay)
        cidade_atual = cp.random.randint(0, n, (n_chains,), dtype=cp.int32)
        visitado = cp.zeros((n_chains, n), dtype=cp.bool_)
        visitado[arange, cidade_atual] = True
        rota_idx = cp.zeros((n_chains, n), dtype=cp.int32)
        rota_idx[:, 0] = cidade_atual
        dist_acumulada = cp.zeros(n_chains, dtype=cp.float32)
        for passo in range(1, n):
            q_linha = Q[cidade_atual]
            q_linha = cp.where(visitado, -INF, q_linha)
            usar_random = cp.random.rand(n_chains) < eps
            prox_greedy = cp.argmax(q_linha, axis=1).astype(cp.int32)
            ruido = cp.random.rand(n_chains, n).astype(cp.float32)
            ruido = cp.where(visitado, -INF, ruido)
            prox_random = cp.argmax(ruido, axis=1).astype(cp.int32)
            proxima = cp.where(usar_random, prox_random, prox_greedy)
            d_passo = dist_mat[cidade_atual, proxima]
            dist_acumulada += d_passo
            recompensa = -d_passo
            visitado_futuro = visitado.copy()
            visitado_futuro[arange, proxima] = True
            q_futuro = Q[proxima]
            q_futuro = cp.where(visitado_futuro, -INF, q_futuro)
            melhor_futuro = cp.max(q_futuro, axis=1)
            melhor_futuro = cp.where(melhor_futuro < -1e29, cp.float32(0.0), melhor_futuro)
            alvo = recompensa + gamma * melhor_futuro
            delta = alpha * (alvo - Q[cidade_atual, proxima])
            idx_flat = cidade_atual.astype(cp.int64) * n + proxima.astype(cp.int64)
            cp.add.at(Q.ravel(), idx_flat, delta)
            visitado = visitado_futuro
            rota_idx[:, passo] = proxima
            cidade_atual = proxima
        origem = rota_idx[:, 0]
        dist_acumulada += dist_mat[cidade_atual, origem]
        idx_ep = int(cp.argmin(dist_acumulada))
        d_ep = float(dist_acumulada[idx_ep])
        if d_ep < melhor_dist:
            melhor_dist = d_ep
    cp.cuda.Stream.null.synchronize()
    t1 = time.perf_counter()
    return {
        "distance_initial": np.nan,
        "distance_final": float(melhor_dist),
        "runtime_total_s": t1 - t0,
        "runtime_setup_s": t_setup - t0,
        "runtime_solve_s": t1 - t_setup,
        "device": "gpu",
        "episodes": episodios,
        "n_chains": n_chains,
        "alpha": alpha,
        "gamma": gamma,
        "epsilon_init": epsilon
    }


def main():
    rows = []
    for n in TAMANHOS:
        pasta_n = DADOS_DIR / f"n_{n}"
        classic_repeats = CLASSIC_REPEATS_MAX50 if n <= 50 else 1
        for inst_idx in range(INSTANCIAS_POR_TAMANHO):
            instance_id = f"n{n}_i{inst_idx}"
            instance_seed = BASE_SEED + n * 1000 + inst_idx
            arquivo = pasta_n / f"instance_{inst_idx:02d}.csv"
            pontos = carregar_pontos(arquivo)
            for run_idx in range(classic_repeats):
                run_seed = instance_seed + run_idx
                res = run_classic_cpu(pontos, run_seed)
                rows.append({"instance_id": instance_id, "n_points": n, "instance_seed": instance_seed, "algorithm": "classic_cpu", "run_idx": run_idx, "run_seed": run_seed, "repeats_planned": classic_repeats, **res})
            if GPU_AVAILABLE:
                for run_idx in range(classic_repeats):
                    run_seed = instance_seed + 10000 + run_idx
                    res = run_classic_gpu(pontos, run_seed)
                    rows.append({"instance_id": instance_id, "n_points": n, "instance_seed": instance_seed, "algorithm": "classic_gpu", "run_idx": run_idx, "run_seed": run_seed, "repeats_planned": classic_repeats, **res})
            for run_idx in range(ML_REPEATS):
                run_seed = instance_seed + 20000 + run_idx
                res = run_ml_cpu(pontos, run_seed)
                rows.append({"instance_id": instance_id, "n_points": n, "instance_seed": instance_seed, "algorithm": "ml_cpu", "run_idx": run_idx, "run_seed": run_seed, "repeats_planned": ML_REPEATS, **res})
            if GPU_AVAILABLE:
                for run_idx in range(ML_REPEATS):
                    run_seed = instance_seed + 30000 + run_idx
                    res = run_ml_gpu(pontos, run_seed)
                    rows.append({"instance_id": instance_id, "n_points": n, "instance_seed": instance_seed, "algorithm": "ml_gpu", "run_idx": run_idx, "run_seed": run_seed, "repeats_planned": ML_REPEATS, **res})
            print(f"[ok] n={n} inst={inst_idx}")
    df = pd.DataFrame(rows)
    df["improvement_pct"] = ((df["distance_initial"] - df["distance_final"]) / df["distance_initial"]) * 100.0
    melhor_por_instancia = df.groupby("instance_id")["distance_final"].transform("min")
    df["gap_to_best_pct"] = ((df["distance_final"] - melhor_por_instancia) / melhor_por_instancia) * 100.0
    df.to_csv(RESULTADOS_DIR / "raw_runs.csv", index=False)
    print("Arquivo salvo:", RESULTADOS_DIR / "raw_runs.csv")


if __name__ == "__main__":
    main()
