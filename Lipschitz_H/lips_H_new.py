import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


def mu(x: float, y: float) -> float:
    f =  0.5*((x - 0.8) ** 2 + (y - 0.7) ** 2) ** 0.5 + 0.1 * ((x - 0.1) ** 2 + (y - 0.2) ** 2 ) ** 0.5
    return 1 - f


@dataclass
class LipsHResult:
    regret: float
    regretlist: List[float]
    grid: List[int]
    cubes: List[List[float]]
    cubes_eli: List[List[List[float]]]


def _ace_sequence(T: int, eps: float) -> Tuple[List[float], List[int]]:
    edge_length: List[float] = []

    ACE_c: List[float] = [1 / eps * 1 / (3 * eps - 2) * math.log2(T / math.log2(T))]
    c_cumu = ACE_c[-1]
    eta = 3 / (3 + 1 / (eps - 1))
    B_star = math.ceil(math.log2(math.log2(T)) / math.log2(4 / 3))

    ACE_d: List[int] = []
    for _ in range(B_star):
        ACE_d.append(math.floor(c_cumu))
        ACE_d.append(math.ceil(c_cumu))
        ACE_c.append(ACE_c[-1] * eta)
        c_cumu = c_cumu + ACE_c[-1]

    ACE_d = sorted(list(set(ACE_d)))
    for i in ACE_d:
        edge_length.append(pow(2, -i))

    return edge_length, ACE_d


def _init_cubes(edge_len0: float) -> List[List[float]]:
    cubes: List[List[float]] = []
    m = round(1 / edge_len0)
    for i in range(m):
        for j in range(m):
            # x_location, y_location, edge_length, rewards_within_batch (list)
            cubes.append([i * edge_len0, j * edge_len0, edge_len0, []])
    return cubes


def _sample(
    rng: np.random.Generator,
    cubes: List[List[float]],
    p: int,
    noise_std: float,
    eps: float,
) -> Tuple[float, float]:
    x = rng.uniform(cubes[p][0], cubes[p][0] + cubes[p][2])
    y = rng.uniform(cubes[p][1], cubes[p][1] + cubes[p][2])
    mean = mu(x, y)
    if abs(eps - 2.0) < 0.1:
        reward = rng.normal(mean, noise_std)
    else:
        pareto_pos = rng.pareto(eps)
        pareto_neg = -rng.pareto(eps)
        reward = mean + (noise_std) * (pareto_pos + pareto_neg)/2.0
        #divide by 2.0 to make the variance of the reward smaller
    return float(reward), float(mean)

def _sample_median(x, T):
    x = np.asarray(x, dtype=float).reshape(-1)
    n = int(x.shape[0])
    if n == 0:
        return 0.0

    delta = 1 / max(T**3, 1)
    raw_k = 8 * np.log(np.exp(1 / 8) / delta)
    #constant can be modified according to meadian of means

    k = int(np.ceil(min(raw_k, n / 2)))
    k = max(1, k)

    groups = np.array_split(x, k)
    means = np.array([g.mean() for g in groups if g.size > 0], dtype=float)
    return float(np.median(means))


def run_lips_h(
    *,
    T: int = 50000,
    noise_std: float = 0.1,
    eps: float = 2,
    seed: Optional[int] = 0,
) -> LipsHResult:
    """
    Port of the core logic from `A-BLiN.ipynb`.
    Returns traces for plotting (no plotting inside).
    """

    rng = np.random.default_rng(seed)

    edge_length, _ = _ace_sequence(T, eps)
    B_total = len(edge_length)

    num: List[int] = []
    for r in edge_length:
        num.append(math.ceil(0.75 * math.log(T) / (r) ** 2))

    cubes = _init_cubes(edge_length[0])

    cubes_eli: List[List[List[float]]] = []
    cubes_eli_grid: List[int] = [-1, len(cubes_eli) - 1]

    mu_m = mu(0.8, 0.7)

    regret = 0.0
    regretlist: List[float] = [0.0]
    grid: List[int] = [0]

    pointer_h = 0
    pointer_t = round(1 / edge_length[0]) ** 2 - 1

    def sample_final(time: int, p_h: int, p_t: int) -> None:
        nonlocal regret
        pointer = p_h
        for _ in range(time):
            r, mu_s = _sample(rng, cubes, pointer, noise_std, eps)
            cubes[pointer][3].append(r)
            pointer += 1
            if pointer > p_t:
                pointer = p_h
            regret = regret + mu_m - mu_s
            regretlist.append(regret)

    def sample_fullbatch(num_samples: int, p_h: int, p_t: int) -> None:
        nonlocal regret
        for _ in range(num_samples):
            for cube_now in range(p_h, p_t + 1):
                r, mu_s = _sample(rng, cubes, cube_now, noise_std, eps)
                cubes[cube_now][3].append(r)
                regret = regret + mu_m - mu_s
                regretlist.append(regret)

    def eli_par(r: float, mu_max: float, p_h: int, p_t: int, num_p: int) -> int:
        p_new = p_t
        for p in range(p_h, p_t + 1):
            if mu_max - cubes[p][3] > 0.25 * r:
                cubes_eli.append(
                    [
                        [
                            cubes[p][0],
                            cubes[p][0],
                            cubes[p][0] + cubes[p][2],
                            cubes[p][0] + cubes[p][2],
                        ],
                        [
                            cubes[p][1],
                            cubes[p][1] + cubes[p][2],
                            cubes[p][1] + cubes[p][2],
                            cubes[p][1],
                        ],
                    ]
                )
                continue
            for i in range(num_p):
                for j in range(num_p):
                    cubes.append(
                        [
                            cubes[p][0] + i * cubes[p][2] / num_p,
                            cubes[p][1] + j * cubes[p][2] / num_p,
                            cubes[p][2] / num_p,
                            [],
                        ]
                    )
            p_new = p_new + num_p**2
        cubes_eli_grid.append(len(cubes_eli) - 1)
        return p_new

    T_accu = 0

    for B in range(T):
        if B == B_total:
            sample_final(T - T_accu, pointer_h, pointer_t)
            grid.append(T)
            break

        num_cube = pointer_t - pointer_h + 1
        num_round = num_cube * num[B]

        if (T_accu + num_round >= T) or (B == B_total):
            sample_final(T - T_accu, pointer_h, pointer_t)
            grid.append(T)
            break

        if T_accu + num_round < T:
            T_accu = T_accu + num_round
            sample_fullbatch(num[B], pointer_h, pointer_t)
            grid.append(T_accu)

            hat_mu_max = -float("inf")
            for p in range(pointer_h, pointer_t + 1):
                cubes[p][3] = _sample_median(cubes[p][3], T)
                if cubes[p][3] > hat_mu_max:
                    hat_mu_max = cubes[p][3]

            if B == B_total - 1:
                num_p = 2
            else:
                num_p = round(edge_length[B] / edge_length[B + 1])

            p_t = eli_par(edge_length[B], hat_mu_max, pointer_h, pointer_t, num_p)
            pointer_h = pointer_t + 1
            pointer_t = p_t
    return LipsHResult(
        regret=float(regret),
        regretlist=regretlist,
        grid=grid,
        cubes=cubes,
        cubes_eli=cubes_eli,
    )


def as_dict(res: LipsHResult) -> Dict[str, object]:
    return {
        "regret": res.regret,
        "regretlist": res.regretlist,
        "grid": res.grid,
        "cubes": res.cubes,
        "cubes_eli": res.cubes_eli,
    }

