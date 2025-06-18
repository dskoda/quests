from typing import List

import numpy as np
from quests.entropy import DEFAULT_BANDWIDTH, DEFAULT_BATCH, kernel_sum


def sequential_px(
    descriptors: List[np.ndarray],
    selected: List[int],
    h: float = DEFAULT_BANDWIDTH,
    batch_size: int = DEFAULT_BATCH,
) -> List[int]:
    """Computes the p_x of an increasingly large dataset given the indices
    of the structures that are selected. This is useful when computing metrics
    such as entropy, diversity, or dH, but without having to recompute everything
    again for each single data point.
    """

    last_px = []
    last_x = None

    for i, n in enumerate(selected):
        # to avoid recomputing kernel matrices, we will perform this only once
        # per reference structure
        current_x = descriptors[n]
        current_px = kernel_sum(current_x, current_x, h=h, batch_size=batch_size)

        # treat first element of the list
        if len(last_px) == 0:
            last_x = current_x
            last_px.append(current_px)
            continue

        # compute the diagonal terms regarding the p_x
        old_px = last_px[-1]
        old_x = last_x.copy()

        upper_px = kernel_sum(old_x, current_x, h=h, batch_size=batch_size)
        lower_px = kernel_sum(current_x, old_x, h=h, batch_size=batch_size)

        old_px = old_px + upper_px
        new_px = current_px + lower_px

        last_px.append(np.concatenate([old_px, new_px]))
        last_x = np.concatenate([old_x, current_x], axis=0)

    return last_px


def sequential_metrics(
    descriptors: List[np.ndarray],
    selected: List[int],
    h: float = DEFAULT_BANDWIDTH,
    batch_size: int = DEFAULT_BATCH,
) -> List[dict]:
    all_px = sequential_px(descriptors, selected, h=h, batch_size=batch_size)

    results = []
    for i, p_x in enumerate(all_px):
        x = np.concatenate([descriptors[n] for n in selected[: i + 1]])
        N = x.shape[0]

        results.append(
            {
                "n_envs": len(p_x),
                "n_structs": i + 1,
                "entropy": float(-np.mean(np.log(p_x / N))),
                "diversity": float(np.log(np.sum(1 / p_x))),
            }
        )

    return results
