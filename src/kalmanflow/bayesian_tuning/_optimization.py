from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import pi, sqrt

import numpy as np
from scipy.special import ndtr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

from ..reservoir_config import ReservoirConfig
from ._diagnostics import _hourly_increment_sd_to_q
from ._types import _PARAMETER_NAMES, BayesianTuningSettings

Array = np.ndarray


def _normal_pdf(value: Array) -> Array:
    return np.exp(-0.5 * value * value) / sqrt(2.0 * pi)


def _weighted_standard_error(values: Array, weights: Array) -> float:
    finite = np.isfinite(values)
    if finite.sum() < 2:
        return 0.0
    values = values[finite]
    weights = np.asarray(weights, dtype=float)[finite]
    weights = weights / weights.sum()
    mean = float(np.dot(weights, values))
    denominator = 1.0 - float(np.sum(weights * weights))
    if denominator <= 0.0:
        return 0.0
    variance = float(np.dot(weights, (values - mean) ** 2) / denominator)
    return sqrt(max(0.0, variance * float(np.sum(weights * weights))))


def _base_parameters(base: ReservoirConfig) -> np.ndarray:
    q = np.asarray(base.q, dtype=float)
    r = np.asarray(base.r, dtype=float)
    return np.asarray([q[0, 0], q[1, 1], q[2, 2], r[0, 0], r[1, 1]], dtype=float)


def _parameter_bounds(
    base: ReservoirConfig,
    inflow_increment_sd_seeds: Sequence[float],
    options: BayesianTuningSettings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    seeds = sorted(
        {_hourly_increment_sd_to_q(value) for value in inflow_increment_sd_seeds}
    )
    if len(seeds) < 3:
        raise ValueError("at least three positive finite candidates are required")
    base_values = _base_parameters(base)
    if not np.isfinite(base_values).all() or np.any(base_values <= 0.0):
        raise ValueError(
            "all tuned diagonal q and r values must be positive and finite"
        )
    scales = base_values.copy()
    multiplier_bounds = (
        options.q_storage_multiplier_bounds,
        (1.0, 1.0),
        options.q_outflow_multiplier_bounds,
        options.r_storage_multiplier_bounds,
        options.r_outflow_multiplier_bounds,
    )
    lows = np.asarray(
        [
            scale * bound[0]
            for scale, bound in zip(scales, multiplier_bounds, strict=True)
        ]
    )
    highs = np.asarray(
        [
            scale * bound[1]
            for scale, bound in zip(scales, multiplier_bounds, strict=True)
        ]
    )
    lows[1] = min(seeds)
    highs[1] = max(seeds)
    # Seed values define the exact q_inflow interval.
    return scales, lows, highs, seeds


def _initial_design(
    seeds: Sequence[float],
    initial_trials: int,
    lows: Array,
    highs: Array,
    base_values: Array,
    rng: np.random.Generator,
) -> list[Array]:
    # A base value need not lie inside a caller-supplied multiplier interval
    # (for example, all multipliers may intentionally be above 1.0).  Seed
    # points are still useful in that case, but must be projected into the
    # declared log-space box before they are evaluated.
    base_normalized = np.clip(
        (np.log(base_values) - np.log(lows)) / (np.log(highs) - np.log(lows)),
        0.0,
        1.0,
    )
    normalized_seed = [
        np.asarray(
            [
                base_normalized[0],
                (np.log(value) - np.log(lows[1]))
                / (np.log(highs[1]) - np.log(lows[1])),
                base_normalized[2],
                base_normalized[3],
                base_normalized[4],
            ],
            dtype=float,
        ).clip(0.0, 1.0)
        for value in seeds
    ]
    design = normalized_seed[:initial_trials]
    if len(design) >= initial_trials:
        return design
    # Greedy maximin selection makes the additional initial evaluations
    # reproducible and spreads them across the five-dimensional box.
    pool = rng.random((max(256, 32 * initial_trials), 5))
    while len(design) < initial_trials:
        distances = np.min(
            np.linalg.norm(pool[:, None, :] - np.asarray(design)[None, :, :], axis=2),
            axis=1,
        )
        selected = int(np.argmax(distances))
        design.append(pool[selected].copy())
        pool = np.delete(pool, selected, axis=0)
    return design


def _decode(normalized: Array, lows: Array, highs: Array) -> dict[str, float]:
    values = np.exp(
        np.log(lows) + np.asarray(normalized) * (np.log(highs) - np.log(lows))
    )
    return {
        name: float(value) for name, value in zip(_PARAMETER_NAMES, values, strict=True)
    }


def _encode(parameters: Mapping[str, float], lows: Array, highs: Array) -> Array:
    values = np.asarray([parameters[name] for name in _PARAMETER_NAMES], dtype=float)
    result = (np.log(values) - np.log(lows)) / (np.log(highs) - np.log(lows))
    return np.clip(result, 0.0, 1.0)


def _fit_and_acquire(
    x_values: list[Array],
    y_values: list[float],
    excluded_x_values: list[Array],
    rng: np.random.Generator,
    options: BayesianTuningSettings,
) -> tuple[Array, str]:
    if not x_values or not y_values:
        pool = rng.random((options.acquisition_pool_size, 5))
        if excluded_x_values:
            excluded = np.asarray(excluded_x_values, dtype=float)
            distances = np.min(
                np.linalg.norm(pool[:, None, :] - excluded[None, :, :], axis=2),
                axis=1,
            )
            return pool[int(np.argmax(distances))], "space-filling-fallback"
        return pool[0], "space-filling-fallback"
    global_count = max(1, options.acquisition_pool_size * 3 // 4)
    pool = rng.random((global_count, 5))
    existing = np.asarray(x_values, dtype=float)
    excluded = np.asarray(excluded_x_values, dtype=float)
    best_point = existing[int(np.argmin(np.asarray(y_values, dtype=float)))]
    local = np.clip(
        best_point
        + rng.normal(
            0.0,
            0.1,
            size=(options.acquisition_pool_size - global_count, 5),
        ),
        0.0,
        1.0,
    )
    pool = np.vstack((pool, local))
    distance = np.min(
        np.linalg.norm(pool[:, None, :] - excluded[None, :, :], axis=2), axis=1
    )
    # Avoid resampling both valid and invalid evaluated neighborhoods.
    pool = pool[distance > 0.02]
    if not len(pool):
        return rng.random(5), "random-fallback"
    try:
        model = GaussianProcessRegressor(
            kernel=Matern(length_scale=np.ones(5), nu=2.5)
            + WhiteKernel(noise_level=1e-6, noise_level_bounds="fixed"),
            normalize_y=True,
            random_state=options.random_seed,
            n_restarts_optimizer=0,
        )
        y = np.asarray(y_values, dtype=float)
        model.fit(existing, y)
        mean, std = model.predict(pool, return_std=True)
        best = float(np.min(y))
        improvement = best - mean - options.expected_improvement_xi
        z = np.divide(improvement, std, out=np.zeros_like(improvement), where=std > 0)
        expected = improvement * ndtr(z) + std * _normal_pdf(z)
        expected[std <= 0.0] = 0.0
        return pool[int(np.argmax(expected))], "expected-improvement"
    except ValueError, RuntimeError, np.linalg.LinAlgError:
        # A numerical GP failure should not prevent a bounded deterministic
        # search from completing.
        distances = np.min(
            np.linalg.norm(pool[:, None, :] - existing[None, :, :], axis=2), axis=1
        )
        return pool[int(np.argmax(distances))], "space-filling-fallback"
