import os

import numpy as np
import matplotlib.pyplot as plt

from base_func import base_func


def main():
    # Parameters (matching main.m)
    K = int(os.environ.get("BB_K", 3))
    T = int(os.environ.get("BB_T", 50000))
    M = int(os.environ.get("BB_M", 3))
    m = int(os.environ.get("BB_ITER", 200))
    eps = 1.1
    # eps = 1.1，1.7

    T_set = np.floor(np.logspace(np.log10(500), np.log10(5e4), 6) + 0.5).astype(int)
    M_set = np.arange(1, 8, dtype=int)

    mu_max = 1
    mu_min = 0.8
    gamma = 0.5

    seed_env = os.environ.get("BB_SEED")
    rng = np.random.default_rng(int(seed_env)) if seed_env is not None else np.random.default_rng()


    # dependence on T
    regretMinimax_T = np.zeros((m, len(T_set)))

    mu = np.concatenate(([mu_max], mu_min * np.ones(K - 1)))
    for iter_idx in range(m):
        for iter_T, temp_T in enumerate(T_set):
            temp_T = int(temp_T)
            regretMinimax_T[iter_idx, iter_T], _ = base_func(
                mu, K, temp_T, M, "minimax", gamma, eps=eps, rng=rng
            )
            regretMinimax_T[iter_idx, iter_T] = regretMinimax_T[iter_idx, iter_T] / temp_T

    #regretMinimax_T_sum = regretMinimax_T.sum(axis=0)
    #np.savetxt("data.txt", regretMinimax_T_sum)
    regretMinimax_T_mean = regretMinimax_T.mean(axis=0)
    np.savetxt("data.txt", regretMinimax_T_mean)

if __name__ == "__main__":
    main()

