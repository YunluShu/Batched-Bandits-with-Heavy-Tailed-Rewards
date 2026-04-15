import os

import numpy as np
#import matplotlib.pyplot as plt

from base_func_H import base_func_H


def main():
    # Parameters (matching main.m)
    K = int(os.environ.get("BB_K", 3))
    T = int(os.environ.get("BB_T", 50000))
    M = int(os.environ.get("BB_M", 3))
    m = int(os.environ.get("BB_ITER", 400))
    eps = 1.1
    # eps = 1.1，1.7

    T_set = np.floor(np.logspace(np.log10(500), np.log10(5e4), 6) + 0.5).astype(int)
    M_set = np.arange(1, 8, dtype=int)

    mu_max = 1
    mu_min = 0.8

    seed_env = os.environ.get("BB_SEED")
    rng = np.random.default_rng(int(seed_env)) if seed_env is not None else np.random.default_rng()

    # dependence on M: run the following code with eps = 1.1, 1.7
    regretinstance_dependent_M = np.zeros((m, len(M_set)))

    mu = np.concatenate(([mu_max], mu_min * np.ones(K - 1)))
    for iter_idx in range(m):
        for iter_M, temp_M in enumerate(M_set):
            regretinstance_dependent_M[iter_idx, iter_M], _ = base_func_H(
                mu, K, T, int(temp_M), "instance-dependent", eps=eps, rng=rng
            )
            regretinstance_dependent_M[iter_idx, iter_M] = regretinstance_dependent_M[iter_idx, iter_M] / T

    regretinstance_dependent_M_mean = regretinstance_dependent_M.mean(axis=0)
    np.savetxt("data.txt", regretinstance_dependent_M_mean)


    # dependence on T: run the following code with eps = 1.1, 1.7
    # regretinstance_dependent_T = np.zeros((m, len(T_set)))

    # mu = np.concatenate(([mu_max], mu_min * np.ones(K - 1)))
    # for iter_idx in range(m):
    #     for iter_T, temp_T in enumerate(T_set):
    #         temp_T = int(temp_T)
    #         regretinstance_dependent_T[iter_idx, iter_T], _ = base_func_H(
    #             mu, K, temp_T, M, "instance-dependent", eps=eps, rng=rng
    #         )
    #         regretinstance_dependent_T[iter_idx, iter_T] = regretinstance_dependent_T[iter_idx, iter_T] / temp_T
    
    # regretinstance_dependent_T_mean = regretinstance_dependent_T.mean(axis=0)
    # np.savetxt("data.txt", regretinstance_dependent_T_mean)



if __name__ == "__main__":
    main()

