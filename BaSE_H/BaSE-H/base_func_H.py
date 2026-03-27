import numpy as np


def median_of_means(x, T, K):
    """
    Robust mean estimator using median-of-means.

    Parameters
    ----------
    x : array_like
        Input samples.
    T : int
        Horizon used to set delta = 1 / (T * K).
    K : int
        Number of arms used to set delta = 1 / (T * K).
    v : float
        Centered (1+eps)-moment bound constant. Fixed to 2 by default.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    n = int(x.shape[0])
    if n == 0:
        return 0.0

    delta = 1 / max(T*K, 1)
    raw_k = 24* np.log(np.exp(1 / 8) / delta)
    #constant can be modified according to meadian of means

    k = int(np.ceil(min(raw_k, n / 2)))
    k = max(1, k)

    groups = np.array_split(x, k)
    means = np.array([g.mean() for g in groups if g.size > 0], dtype=float)
    return float(np.median(means))


def base_func_H(mu, K, T, M, grid_type, eps, rng=None):
    """
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

    # Build initial batch grid 
    if grid_type == "instance-dependent":
        b = T ** (1.0 / M)
        tgrid = np.floor(b ** np.arange(1, M + 1, dtype=float))
        tgrid[-1] = T
        TGrid = np.concatenate(([0.0], tgrid)).astype(float)
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

            samples = rng.pareto(eps, size=pullNumber)
            sample_mm = median_of_means(samples, T=T, K=K)
            observed = sample_mm + mu[j]

            #averageReward[j] = averageReward[j] * (old_count / new_count) + observed * (
            #    pullNumber / new_count
            #)
            averageReward[j] = observed

            regret += pullNumber * (mu[0] - mu[j])
            numberPull[j] = new_count

        if active_idx.size == 0:
            break


        maxArm = float(averageReward[active].max())
        thresh_base = (1/4) * (np.log(T * K) ) ** ((eps - 1) / (eps))
        #constanrt can change according to eps

        for j in active_idx:
            thresh = thresh_base / ((numberPull[j])** ( (eps - 1) / (eps)))
            if (maxArm - averageReward[j]) >= thresh:
                active[j] = False

    active_set = active.astype(int)
    return float(regret), active_set

