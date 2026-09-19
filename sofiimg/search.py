"""Search over the space of segment rankings.

Two procedures are available and they answer different questions.

The sequential ranking of Algorithm 3 selects, at every position, the segment whose
perturbation lowers the probability of the explained class the most, given
everything already perturbed. Theorem 3 proves that this ranking is optimal when
the black box is modular, and it costs a quadratic number of queries in the
number of segments. It also serves as a strong starting point for the search.

Hill climbing improves an incumbent ranking through a local operator that swaps
two randomly selected positions. A candidate is accepted when it raises the
degradation score, and the run ends after a budget of proposals or once the
patience expires. Modularity rarely holds in practice, since the effect of
perturbing one segment depends on what was perturbed before, so the search
usually finds rankings the greedy procedure cannot reach.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

__all__ = ["hill_climbing", "greedy_ranking"]

_TOLERANCE = 1e-12


def _progress_bar(total: int, enabled: bool, description: str):
    if not enabled:
        return None
    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover - progress reporting is optional
        return None
    return tqdm(total=total, desc=description, leave=False)


def greedy_ranking(objective) -> List[int]:
    """Ranking built by the greedy selection of Algorithm 3.

    At every position the remaining segment whose perturbation yields the lowest
    probability for the explained class is appended, and the set of perturbed
    segments grows accordingly. The procedure is optimal under the modularity
    assumption of Theorem 3, and it is a reasonable heuristic otherwise.

    Parameters
    ----------
    objective : Objective
        Evaluator bound to one instance, one segmentation and one operator.

    Returns
    -------
    list of int
        Positions of the segments, from the most to the least relevant.
    """
    remaining = list(range(objective.n_segments))
    chosen: List[int] = []
    state = objective.template.copy()

    while remaining:
        states = []
        for position in remaining:
            candidate = state.copy()
            objective._perturb(candidate, position)
            states.append(candidate)
        scores, _ = objective._predict_states(states)
        best = int(np.argmin(scores))
        position = remaining.pop(best)
        objective._perturb(state, position)
        chosen.append(position)
    return chosen


def hill_climbing(
    n_segments: int,
    evaluate: Callable,
    initial_order: Sequence[int],
    max_iterations: int = 100,
    patience: Optional[int] = 10,
    n_restarts: int = 0,
    accept_equal: bool = False,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = False,
    description: str = "SOFI search",
    record_history: bool = True,
) -> Dict:
    """Maximize the degradation score over the rankings of a segmentation.

    Parameters
    ----------
    n_segments : int
        Length of a ranking.
    evaluate : callable
        Receives a ranking and, optionally, the evaluation of the incumbent
        together with the swapped positions. It returns an object exposing
        ``order`` and ``ds``.
    initial_order : sequence of int
        Ranking the first climb starts from.
    max_iterations : int, default=100
        Upper bound on the proposed swaps across every restart.
    patience : int, optional
        Consecutive proposals without improvement tolerated before a restart or
        the end of the run. ``None`` spends the whole budget.
    n_restarts : int, default=0
        Restarts from a random ranking granted once the patience expires.
    accept_equal : bool, default=False
        Whether swaps that leave the score unchanged are accepted, which lets
        the search drift along plateaus.
    rng : Generator, optional
        Source of randomness of the operator and of the restarts.
    verbose : bool, default=False
        Whether a progress bar is displayed.
    description : str, default="SOFI search"
        Label placed on the progress bar.
    record_history : bool, default=True
        Whether one record per proposal is kept, which the optimization trace
        figure consumes.

    Returns
    -------
    dict
        The best evaluation found, the history and a few counters.
    """
    if n_segments < 2:
        evaluation = evaluate([int(p) for p in initial_order])
        return {
            "evaluation": evaluation,
            "history": [],
            "n_iterations": 0,
            "n_restarts_used": 0,
            "n_evaluations": 1,
        }

    rng = np.random.default_rng() if rng is None else rng
    patience = max_iterations if patience is None else int(patience)

    current = evaluate([int(p) for p in initial_order])
    cache = {tuple(current.order): current}
    best = current

    history: List[Dict] = []
    if record_history:
        history.append(
            {
                "iteration": 0,
                "swap": None,
                "order": list(current.order),
                "candidate_ds": current.ds,
                "current_ds": current.ds,
                "best_ds": best.ds,
                "accepted": True,
                "restart": False,
                "prior": True,
            }
        )

    iteration = 0
    stagnation = 0
    restarts_used = 0
    evaluations = 1

    bar = _progress_bar(max_iterations, verbose, description)
    try:
        while iteration < max_iterations:
            first, second = sorted(rng.choice(n_segments, size=2, replace=False))
            candidate_order = list(current.order)
            candidate_order[first], candidate_order[second] = (
                candidate_order[second],
                candidate_order[first],
            )
            key = tuple(candidate_order)

            if key in cache:
                candidate = cache[key]
            else:
                candidate = evaluate(candidate_order, current, (int(first), int(second)))
                cache[key] = candidate
                evaluations += 1

            improved = candidate.ds > current.ds + _TOLERANCE
            accepted = improved or (
                accept_equal and candidate.ds >= current.ds - _TOLERANCE
            )
            if accepted:
                current = candidate
            stagnation = 0 if improved else stagnation + 1
            if candidate.ds > best.ds + _TOLERANCE:
                best = candidate

            iteration += 1
            if record_history:
                history.append(
                    {
                        "iteration": iteration,
                        "swap": (int(first), int(second)),
                        "order": list(candidate.order),
                        "candidate_ds": candidate.ds,
                        "current_ds": current.ds,
                        "best_ds": best.ds,
                        "accepted": bool(accepted),
                        "restart": False,
                        "prior": False,
                    }
                )
            if bar is not None:
                bar.update(1)
                bar.set_postfix(ds=f"{best.ds:.4f}", refresh=False)

            if stagnation >= patience:
                if restarts_used >= n_restarts:
                    break
                restarts_used += 1
                stagnation = 0
                restart_order = [int(p) for p in rng.permutation(n_segments)]
                key = tuple(restart_order)
                if key in cache:
                    current = cache[key]
                else:
                    current = evaluate(restart_order)
                    cache[key] = current
                    evaluations += 1
                if current.ds > best.ds + _TOLERANCE:
                    best = current
                if record_history:
                    history.append(
                        {
                            "iteration": iteration,
                            "swap": None,
                            "order": list(current.order),
                            "candidate_ds": current.ds,
                            "current_ds": current.ds,
                            "best_ds": best.ds,
                            "accepted": True,
                            "restart": True,
                            "prior": False,
                        }
                    )
    finally:
        if bar is not None:
            bar.close()

    return {
        "evaluation": best,
        "history": history,
        "n_iterations": iteration,
        "n_restarts_used": restarts_used,
        "n_evaluations": evaluations,
    }
