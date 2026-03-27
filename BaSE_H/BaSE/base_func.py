import numpy as np


def base_func(mu, K, T, M, grid_type, gamma, eps=1.5, rng=None):
    """
    Batched Multi-armed Bandit baseline (BaSE) translated from BASEFunc.m.

    Parameters
    ----------
    mu : array_like
        Arm means. Length should be K.
    K : int
        Number of arms (batch reward streams).
    T : int
        Horizon.
    M : int
        Number of batches / grid segments.
    grid_type : str
        'minimax' | 'geometric' | 'arithmetic'
    gamma : float
        Tuning parameter for elimination threshold.
    rng : numpy.random.Generator, optional
        RNG for reproducibility.

    Returns
    -------
    (regret, active_set) :
        regret : float
        active_set : np.ndarray of shape (K,)
            0/1 array indicating which arms remain active at the end.
    """
    if rng is None:
        rng = np.random.default_rng()

    mu = np.asarray(mu, dtype=float).reshape(-1)
    if K is None:
        K = int(mu.shape[0])
    if mu.shape[0] != K:
        raise ValueError(f"mu length ({mu.shape[0]}) must equal K ({K}).")

    regret = 0.0

    if grid_type == "minimax":
        a = T ** (1.0 / (2.0 - 2.0 ** (1.0 - M)))
        exponents = 2.0 - 1.0 / (2.0 ** np.arange(0, M, dtype=float))
        tgrid = np.floor(a ** exponents)
        tgrid[-1] = T
        TGrid = np.concatenate(([0.0], tgrid)).astype(float)
    elif grid_type == "geometric":
        b = T ** (1.0 / M)
        tgrid = np.floor(b ** np.arange(1, M + 1, dtype=float))
        tgrid[-1] = T
        TGrid = np.concatenate(([0.0], tgrid)).astype(float)
    elif grid_type == "arithmetic":
        TGrid = np.floor(np.linspace(0.0, float(T), M + 1)).astype(float)
    else:
        raise ValueError(f"Unsupported grid_type: {grid_type!r}")

    active = np.ones(K, dtype=bool)
    numberPull = np.zeros(K, dtype=float)
    averageReward = np.zeros(K, dtype=float)

    for i_py in range(1, M + 1):
        availableK = int(active.sum())
        if availableK <= 0:
            break

        pullNumber = (TGrid[i_py] - TGrid[i_py - 1]) / availableK
        pullNumber = int(max(np.floor(pullNumber), 1))
        TGrid[i_py] = availableK * pullNumber + TGrid[i_py - 1]

        active_idx = np.where(active)[0]

        for j in active_idx:
            # Incremental mean update using counts.
            old_count = numberPull[j]
            new_count = old_count + pullNumber

            sample_mean = rng.pareto(eps, size=pullNumber).mean()
            observed = sample_mean + mu[j]

            averageReward[j] = averageReward[j] * (old_count / new_count) + observed * (
                pullNumber / new_count
            )

            regret += pullNumber * (mu[0] - mu[j])
            numberPull[j] = new_count

        if active_idx.size == 0:
            break

        maxArm = float(averageReward[active].max())
        thresh_base = np.sqrt(gamma * np.log(T * K))
        for j in active_idx:
            thresh = thresh_base / np.sqrt(numberPull[j])
            if (maxArm - averageReward[j]) >= thresh:
                active[j] = False

    active_set = active.astype(int)
    return float(regret), active_set

