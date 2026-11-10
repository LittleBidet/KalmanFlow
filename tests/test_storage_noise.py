from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from kalmanflow import (
    NoiseEvaluationSettings,
    NoiseTuningSettings,
    ReservoirConfig,
    UnitSystem,
    ValidationWindow,
    estimate_storage_noise,
    tune_inflow_noise,
)
from kalmanflow.noise_tuning import _storage_noise, _types


def _noisy_storage(
    count: int = 1200, noise_variance: float = 4.0, seed: int = 3
) -> pd.Series:
    """Smooth storage plus white reading noise of known variance."""

    index = pd.date_range("2025-01-01", periods=count, freq="15min", tz="UTC")
    rng = np.random.default_rng(seed)
    smooth = 1000.0 + 50.0 * np.sin(np.linspace(0.0, 6.0, count))
    noise = rng.normal(0.0, np.sqrt(noise_variance), count)
    return pd.Series(smooth + noise, index=index)


def _halves(index: pd.DatetimeIndex) -> tuple[ValidationWindow, ValidationWindow]:
    middle = index[len(index) // 2]
    return (
        ValidationWindow("first", index[0], middle),
        ValidationWindow("second", middle, index[-1] + pd.Timedelta(minutes=15)),
    )


def test_estimate_recovers_known_reading_noise_per_window_and_pooled() -> None:
    storage = _noisy_storage()
    frame = estimate_storage_noise(storage, _halves(storage.index))

    assert frame["window"].tolist() == ["first", "second", "all"]
    np.testing.assert_allclose(frame["r_storage"], 4.0, rtol=0.15)
    np.testing.assert_allclose(frame["reading_sd"], 2.0, rtol=0.08)
    pooled = frame.set_index("window").loc["all"]
    assert pooled["pairs"] == len(storage) - 2
    assert pooled["blocks"] == 2  # 12.5 days of readings span two weeks
    assert (frame["unchanged_share"] == 0.0).all()


def _stormy_storage() -> tuple[pd.Series, tuple[ValidationWindow, ValidationWindow]]:
    """Four calm weeks after four weeks that include a two-day storm."""

    count = 2 * 28 * 96
    index = pd.date_range("2025-01-01", periods=count, freq="15min", tz="UTC")
    rng = np.random.default_rng(5)
    true_changes = np.zeros(count)
    storm = np.zeros(2 * 96)
    for step in range(1, len(storm)):
        storm[step] = 0.98 * storm[step - 1] + rng.normal(0.0, 3.0)
    true_changes[5 * 96 : 7 * 96] = storm
    storage = pd.Series(
        1000.0 + np.cumsum(true_changes) + rng.normal(0.0, 2.0, count), index=index
    )
    middle = index[count // 2]
    windows = (
        ValidationWindow("stormy", index[0], middle),
        ValidationWindow("calm", middle, index[-1] + pd.Timedelta(minutes=15)),
    )
    return storage, windows


def test_block_median_is_not_dragged_to_zero_by_a_storm() -> None:
    storage, windows = _stormy_storage()
    stormy = np.diff(storage.to_numpy()[: len(storage) // 2])
    # One covariance over the whole window is swamped by the storm...
    assert _storage_noise._negative_lag1_covariance(stormy[1:], stormy[:-1]) < 0.0

    # ...but the median of weekly estimates still finds the true noise of 4.
    frame = estimate_storage_noise(storage, windows)
    np.testing.assert_allclose(frame["r_storage"], 4.0, rtol=0.15)
    assert _storage_noise._storage_noise_warnings(frame) == []


def test_estimate_without_windows_returns_only_the_pooled_row() -> None:
    storage = _noisy_storage()
    frame = estimate_storage_noise(storage)
    assert frame["window"].tolist() == ["all"]
    assert frame.loc[0, "start"] == storage.index[0]
    assert frame.loc[0, "end"] == storage.index[-1]


def test_estimate_needs_enough_evenly_spaced_pairs() -> None:
    short = _noisy_storage(count=50)
    frame = estimate_storage_noise(short)
    assert np.isnan(frame.loc[0, "r_storage"])
    assert frame.loc[0, "pairs"] == 48

    # Doubling every other interval leaves no evenly spaced triple.
    storage = _noisy_storage(count=400)
    steps = np.cumsum(np.tile([15, 30], 200))
    storage.index = storage.index[0] + pd.to_timedelta(steps, unit="min")
    assert estimate_storage_noise(storage).loc[0, "pairs"] == 0


def test_daily_readings_have_too_few_pairs_in_each_week() -> None:
    daily = _noisy_storage(count=200)
    daily.index = pd.date_range("2025-01-01", periods=200, freq="D", tz="UTC")
    frame = estimate_storage_noise(daily)
    assert frame.loc[0, "pairs"] == 198
    assert frame.loc[0, "blocks"] == 0
    assert np.isnan(frame.loc[0, "r_storage"])


def test_smooth_storage_without_noise_estimates_zero() -> None:
    storage = _noisy_storage(noise_variance=0.0)
    assert estimate_storage_noise(storage).loc[0, "r_storage"] == 0.0


def test_estimate_rejects_non_series_input() -> None:
    with pytest.raises(TypeError, match="pandas Series"):
        estimate_storage_noise([1.0, 2.0, 3.0])  # type: ignore[arg-type]


def _frame(*estimates: float, pooled: float = 1.0) -> pd.DataFrame:
    rows = [
        {"window": f"window-{i}", "r_storage": value}
        for i, value in enumerate(estimates)
    ]
    return pd.DataFrame([*rows, {"window": "all", "r_storage": pooled}])


def test_disagreeing_windows_point_to_the_public_estimator() -> None:
    warnings = _storage_noise._storage_noise_warnings(_frame(8.0, 0.1, 6.0))
    assert len(warnings) == 1
    assert "window-1: r_storage 0.1" in warnings[0]
    assert "window-0: 8" in warnings[0]
    assert "kalmanflow.estimate_storage_noise(storage, validation_windows)" in (
        warnings[0]
    )


def test_zero_windows_are_named_but_not_compared() -> None:
    warnings = _storage_noise._storage_noise_warnings(_frame(0.0, 2.0, 2.5))
    assert len(warnings) == 1
    assert warnings[0].startswith("window-0 gave no reading-noise estimate")
    assert "not compared" in warnings[0]


def test_noiseless_window_is_named_rather_than_silently_dropped() -> None:
    noisy = _noisy_storage(count=1400)
    quiet = _noisy_storage(count=1400, noise_variance=0.0)
    storage = pd.concat([noisy, quiet.set_axis(noisy.index + pd.Timedelta(days=15))])
    windows = _halves(storage.index)
    frame = estimate_storage_noise(storage, windows)
    assert frame.set_index("window").loc["second", "r_storage"] == 0.0
    warnings = _storage_noise._storage_noise_warnings(frame)
    assert any(warning.startswith("second gave no") for warning in warnings)
    assert not any("differs between validation windows" in w for w in warnings)


def test_similar_or_sparse_windows_do_not_warn() -> None:
    warn = _storage_noise._storage_noise_warnings
    assert warn(_frame(4.0, 3.0, 5.0)) == []
    assert warn(_frame(4.0, np.nan)) == []
    assert warn(pd.DataFrame()) == []


def test_unusable_pooled_estimates_explain_why() -> None:
    warn = _storage_noise._storage_noise_warnings
    too_few = warn(_frame(pooled=np.nan))[0]
    assert "fewer than 100 evenly spaced storage change pairs" in too_few
    no_signal = warn(_frame(pooled=0.0))[0]
    assert "no reading-noise signal" in no_signal
    assert "storm-driven" in no_signal


def _config(r_storage: float = 0.25) -> ReservoirConfig:
    return ReservoirConfig(
        reservoir_id="noise",
        reservoir_name="Noise",
        q=np.diag([0.002, 1.0, 0.02]),
        r=np.diag([r_storage, 4.0]),
        p0=np.diag([10.0, 20.0, 30.0]),
        smoothing_lag=timedelta(hours=1),
        model_version="model-v1",
        configuration_version="config-v1",
        unit_system=UnitSystem.us_customary(),
    )


def _tune(storage: pd.Series, **tuning: object):
    index = storage.index
    thirds = np.array_split(np.arange(len(index)), 3)
    windows = tuple(
        ValidationWindow(
            f"window-{number}",
            index[rows[0]],
            index[rows[-1]] + pd.Timedelta(minutes=15),
        )
        for number, rows in enumerate(thirds)
    )
    return tune_inflow_noise(
        storage,
        pd.Series(4.0, index=index),
        _config(),
        [2.0, 5.0],
        windows,
        settings=NoiseEvaluationSettings(warmup=timedelta(0)),
        tuning_settings=NoiseTuningSettings(
            max_evaluations_per_search=15, check_identifiability=False, **tuning
        ),
        proposed_configuration_version="config-v2",
    )


def test_tuner_centers_r_storage_bounds_on_the_data_estimate() -> None:
    result = _tune(_noisy_storage(count=600))
    estimate = result.selected_config.metadata["selection"]["r_storage_estimate"]

    assert estimate == pytest.approx(4.0, rel=0.25)
    assert estimate / 2 <= result.selected_parameters["r_storage"] <= estimate * 2
    summary = result.candidate_summary.set_index("candidate")
    assert summary.loc["base", "r_storage"] == 0.25
    assert list(result.storage_noise["window"]) == [
        "window-0",
        "window-1",
        "window-2",
        "all",
    ]
    assert not any("could not be estimated" in w for w in result.warnings)


def test_tuner_keeps_configured_bounds_when_estimation_is_off() -> None:
    result = _tune(_noisy_storage(count=600), estimate_r_storage=False)
    assert result.storage_noise.empty
    assert np.isnan(result.selected_config.metadata["selection"]["r_storage_estimate"])
    assert 0.125 <= result.selected_parameters["r_storage"] <= 0.25


def test_tuner_falls_back_when_storage_is_too_short_to_estimate() -> None:
    result = _tune(_noisy_storage(count=90))
    assert any("could not be estimated" in w for w in result.warnings)
    assert 0.125 <= result.selected_parameters["r_storage"] <= 0.25


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("estimate_r_storage", "yes", TypeError),
        ("r_storage_estimate_factor", 1.0, ValueError),
        ("r_storage_estimate_factor", np.inf, ValueError),
    ],
)
def test_storage_estimate_settings_are_validated(
    field: str, value: object, error: type[Exception]
) -> None:
    with pytest.raises(error):
        NoiseTuningSettings(**{field: value})


def test_result_requires_a_storage_noise_dataframe() -> None:
    with pytest.raises(TypeError, match="storage_noise"):
        _types.NoiseTuningResult(
            selected_config=_config(),
            selected_parameters={},
            selected_objective=0.0,
            base_objective=0.0,
            improvement_standard_error=0.0,
            candidate_summary=pd.DataFrame(),
            window_diagnostics=pd.DataFrame(),
            storage_noise=[],  # type: ignore[arg-type]
        )
