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
    raw_k = 1* np.log(np.exp(1 / 8) / delta)
    #constant can be modified according to meadian of means

    k = int(np.ceil(min(raw_k, n / 2)))
    k = max(1, k)

    groups = np.array_split(x, k)
    means = np.array([g.mean() for g in groups if g.size > 0], dtype=float)
    return float(np.median(means))


def base_func_H(mu, K, T, M, grid_type, eps, rng=None, return_trace=False):
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
    If return_trace is False:
        (regret, active_set)
            regret : float
            active_set : np.ndarray of shape (K,)
                0/1 array indicating which arms remain active at the end.
    If return_trace is True:
        (
            regret,
            active_set,
            regret_trace,
            time_trace,
            active_set_trace,
            active_set_by_batch,
            batch_index_trace,
        )
            regret_trace : np.ndarray
                Cumulative regret at recorded intermediate time points.
            time_trace : np.ndarray
                Time indices corresponding to regret_trace.
            active_set_trace : np.ndarray
                Active-set indicator (0/1) at each recorded time point.
            active_set_by_batch : np.ndarray
                Active-set indicator (0/1) recorded once per batch.
            batch_index_trace : np.ndarray
                Batch indices corresponding to active_set_by_batch.
    """
    if rng is None:
        rng = np.random.default_rng()

    mu = np.asarray(mu, dtype=float).reshape(-1)
    if K is None:
        K = int(mu.shape[0])
    if mu.shape[0] != K:
        raise ValueError(f"mu length ({mu.shape[0]}) must equal K ({K}).")

    regret = 0.0
    time_now = 0
    regret_trace = [0.0]
    time_trace = [0]

    # Build initial batch grid 
    if grid_type == "instance-dependent":
        b = T ** (1.0 / M)
        tgrid = np.floor(b ** np.arange(1, M + 1, dtype=float))
        tgrid[-1] = T
        TGrid = np.concatenate(([0.0], tgrid)).astype(float)
    else:
        raise ValueError(f"Unsupported grid_type: {grid_type!r}")

    active = np.ones(K, dtype=bool)
    active_set_by_batch = []
    batch_index_trace = []
    numberPull = np.zeros(K, dtype=float)
    averageReward = np.zeros(K, dtype=float)

    for i_py in range(1, M + 1):
        availableK = int(active.sum())
        if availableK <= 0:
            break
        batch_index_trace.append(int(i_py))
        active_set_by_batch.append(active.astype(int).copy())

        pullNumber = (TGrid[i_py] - TGrid[i_py - 1]) / availableK
        pullNumber = int(max(np.floor(pullNumber), 1))
        TGrid[i_py] = availableK * pullNumber + TGrid[i_py - 1]

        active_idx = np.where(active)[0]

        # First update each active arm's observed reward for this batch.
        per_pull_regret = {}
        for j in active_idx:
            old_count = numberPull[j]
            new_count = old_count + pullNumber

            samples = rng.pareto(eps, size=pullNumber)
            sample_mm = median_of_means(samples, T=T, K=K)
            averageReward[j] = sample_mm + mu[j]
            numberPull[j] = new_count
            per_pull_regret[j] = float(mu[0] - mu[j])

        # Then record regret in round-robin order
        for _ in range(pullNumber):
            for j in active_idx:
                regret += per_pull_regret[j]
                time_now += 1
                regret_trace.append(float(regret))
                time_trace.append(int(time_now))

        if active_idx.size == 0:
            break

        maxArm = float(averageReward[active].max())
        thresh_base = (2/eps) * (np.log(T * K) ) ** ((eps - 1) / (eps))
        #constanrt can change according to eps

        for j in active_idx:
            thresh = thresh_base / ((numberPull[j])** ( (eps - 1) / (eps)))
            if (maxArm - averageReward[j]) >= thresh:
                active[j] = False

    active_set = active.astype(int)
    if return_trace:
        return (
            float(regret),
            active_set,
            np.asarray(regret_trace, dtype=float),
            np.asarray(time_trace, dtype=int),
            np.asarray(active_set_by_batch, dtype=int),
            np.asarray(batch_index_trace, dtype=int),
        )
    return float(regret), active_set

