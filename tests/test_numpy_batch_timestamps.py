"""Timestamp release boundaries and vectorized batch preparation parity."""

from datetime import UTC, datetime, timedelta
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from kalmanflow import core, get_reservoir_inflow
from kalmanflow.models import ReservoirStateSpaceModel
from kalmanflow.time_utils import elapsed_seconds


@pytest.mark.parametrize("unit", ["ns", "us", "ms", "s"])
@pytest.mark.parametrize("irregular", [False, True])
def test_preparation_preserves_scalar_matrices_and_does_not_mutate(unit, irregular):
    index = pd.date_range(
        "2024-03-10 00:00", periods=10, freq="15min", tz="America/Los_Angeles"
    ).as_unit(unit)
    if irregular:
        index = index + pd.to_timedelta(np.arange(len(index)) ** 2, unit="s")
        index = index.as_unit(unit)
    storage = pd.Series(100.0 + np.arange(len(index)), index=index)
    discharge = pd.Series(np.full(len(index), 4.0), index=index)
    storage.iloc[1] = np.nan  # initialization skips a row, even on a regular index
    originals = storage.copy(deep=True), discharge.copy(deep=True)
    actual_filter = core.kalman_filter
    captured = []

    def check_filter(observations, **kwargs):
        retained = index[np.r_[0, np.arange(2, len(index))]]
        model = ReservoirStateSpaceModel(q_continuous=np.diag([0.002, 1.0, 0.02]))
        deltas = [
            (b - a).total_seconds()
            for a, b in zip(retained[:-1], retained[1:], strict=True)
        ]
        for key, operation in (
            ("transition_matrix", model.transition_matrix),
            ("process_covariance", model.process_covariance),
        ):
            expected = np.asarray([operation(value) for value in deltas])
            assert kwargs[key].tobytes() == expected.tobytes()
        captured.append(True)
        return actual_filter(observations, **kwargs)

    with patch.object(core, "kalman_filter", side_effect=check_filter):
        get_reservoir_inflow(storage, discharge, 0.002, 1.0, 0.02, 0.25, 4.0)
    assert captured == [True]
    pd.testing.assert_series_equal(storage, originals[0], check_exact=True)
    pd.testing.assert_series_equal(discharge, originals[1], check_exact=True)


@pytest.mark.parametrize("offset_us", [-1, 0, 1])
@pytest.mark.parametrize("unit", ["ns", "us"])
@pytest.mark.parametrize("pandas_lag", [False, True])
def test_integer_release_matches_pandas_one_microsecond_around_lag(
    offset_us, unit, pandas_lag
):
    lag = pd.Timedelta(hours=1) if pandas_lag else timedelta(hours=1)
    start = pd.Timestamp("2024-11-03 00:30", tz="America/Los_Angeles")
    index = pd.DatetimeIndex(
        [
            start,
            start + timedelta(minutes=15),
            start + lag + timedelta(microseconds=offset_us),
        ]
    ).as_unit(unit)
    storage = pd.Series([100.0, 101.0, 103.0], index=index)
    discharge = pd.Series([4.0, 4.0, 4.0], index=index)
    output = get_reservoir_inflow(
        storage, discharge, 0.002, 1.0, 0.02, 0.25, 4.0, smoothing_lag=lag
    )
    expected = index[-1] - index[0] >= lag
    assert pd.notna(output.revised_inflow.iloc[0]) == expected
    assert output.revised_inflow.iloc[1:].isna().all()
    assert output.index.equals(index)


@pytest.mark.parametrize(
    "index,message",
    [
        (pd.DatetimeIndex(["2024-01-01"]), "timestamp must be timezone-aware"),
        (
            pd.DatetimeIndex(["2024-01-01", pd.NaT], tz="UTC"),
            "timestamp must be timezone-aware",
        ),
        (
            pd.DatetimeIndex(["2024-01-02", "2024-01-01"], tz="UTC"),
            "observation timestamps must be strictly increasing",
        ),
        (
            pd.DatetimeIndex(["2024-01-01", "2024-01-01"], tz="UTC"),
            "observation timestamps must be strictly increasing",
        ),
        (
            pd.DatetimeIndex(["2024-01-01T00:00:00.000000001Z"]),
            "timestamps finer than microsecond precision are not supported",
        ),
        # Preserve first-error ordering when several contracts are violated.
        (
            pd.to_datetime(
                [
                    "2024-01-02T00:00:00Z",
                    "2024-01-01T00:00:00Z",
                    "2024-01-03T00:00:00.000000001Z",
                ],
                format="mixed",
            ),
            "observation timestamps must be strictly increasing",
        ),
        (
            pd.to_datetime(
                ["2024-01-02T00:00:00Z", "2024-01-01T00:00:00.000000001Z"],
                format="mixed",
            ),
            "timestamps finer than microsecond precision are not supported",
        ),
    ],
)
def test_fast_timestamp_validation_preserves_errors(index, message):
    with pytest.raises(ValueError) as error:
        core._validate_batch_timestamps(index)
    assert str(error.value) == message


def test_timestamp_validation_empty_singleton_and_dst_fold():
    core._validate_batch_timestamps(pd.DatetimeIndex([]))
    core._validate_batch_timestamps(pd.DatetimeIndex([], tz="UTC"))
    core._validate_batch_timestamps(
        pd.DatetimeIndex([datetime(2024, 1, 1, tzinfo=UTC)])
    )
    zone = ZoneInfo("America/Los_Angeles")
    index = pd.DatetimeIndex(
        [datetime(2024, 11, 3, 1, 30, tzinfo=zone, fold=fold) for fold in (0, 1)]
    )
    core._validate_batch_timestamps(index)


def test_vector_elapsed_preserves_pandas_rounding_for_large_intervals():
    index = pd.DatetimeIndex(
        [
            pd.Timestamp("1900-01-01T00:00:00Z"),
            pd.Timestamp("2200-01-01T00:00:00.000001Z"),
            pd.Timestamp("2500-01-01T00:00:00.000002Z"),
        ]
    ).as_unit("us")
    expected = np.asarray(
        [(b - a).total_seconds() for a, b in zip(index[:-1], index[1:], strict=True)]
    )
    actual = core._batch_elapsed_seconds(index)
    assert actual.tobytes() == expected.tobytes()


def test_documented_preexisting_stream_float_release_discrepancy():
    earlier = datetime(2024, 1, 1, tzinfo=UTC)
    age = timedelta(days=109500, microseconds=1)
    lag = timedelta(days=109500, microseconds=2)
    assert age < lag
    assert elapsed_seconds(earlier + age, earlier) >= lag.total_seconds()


@pytest.mark.parametrize("kind", ["object", "far_seconds"])
def test_timestamp_fast_path_preserves_previously_accepted_index_ranges(kind):
    regular = pd.date_range("2024-01-01", periods=4, freq="15min", tz="UTC")
    if kind == "object":
        index = pd.Index(regular.to_pydatetime(), dtype=object)
    else:
        seconds = np.array([10**13 + 900 * k for k in range(4)], dtype=np.int64)
        index = pd.DatetimeIndex(seconds.view("datetime64[s]"), tz="UTC")

    def batch(index):
        return get_reservoir_inflow(
            pd.Series([100.0, 101.0, 102.0, 103.0], index=index),
            pd.Series([4.0, 4.0, 4.0, 4.0], index=index),
            0.002,
            1.0,
            0.02,
            0.25,
            4.0,
            smoothing_lag=timedelta(minutes=15),
        )

    expected, actual = batch(regular), batch(index)
    pd.testing.assert_frame_equal(
        actual.reset_index(drop=True), expected.reset_index(drop=True), check_exact=True
    )
    assert actual.index.equals(index)


def test_microsecond_conversion_empty_singleton_and_overflow_fallback():
    for index in (
        pd.DatetimeIndex([], tz="UTC"),
        pd.date_range("2024-01-01", periods=1, tz="UTC"),
    ):
        assert core._batch_elapsed_seconds(index).shape == (0,)
    values = np.array([-5 * 10**18, 5 * 10**18], dtype=np.int64)
    index = pd.DatetimeIndex(values.view("datetime64[us]"), tz="UTC")
    # Preserve pandas' subtraction exception for a span beyond its resolution.
    with pytest.raises(
        (pd.errors.OutOfBoundsDatetime, pd.errors.OutOfBoundsTimedelta, OverflowError)
    ) as original:
        index[1] - index[0]
    with pytest.raises(type(original.value)) as actual:
        core._batch_elapsed_seconds(index)
    assert str(actual.value) == str(original.value)


def test_regular_cadence_constructs_only_one_distinct_matrix_pair(monkeypatch):
    from unittest.mock import Mock

    model = ReservoirStateSpaceModel
    transition = Mock(side_effect=model.transition_matrix)
    process = Mock(side_effect=model.process_covariance)
    # Preserve method binding while counting the original pure computations.
    monkeypatch.setattr(
        model, "transition_matrix", lambda self, dt: transition(self, dt)
    )
    monkeypatch.setattr(model, "process_covariance", lambda self, dt: process(self, dt))
    index = pd.date_range("2024-01-01", periods=30, freq="15min", tz="UTC")
    get_reservoir_inflow(
        pd.Series(100.0 + np.arange(30), index=index),
        pd.Series(4.0, index=index),
        0.002,
        1.0,
        0.02,
        0.25,
        4.0,
    )
    assert transition.call_count == process.call_count == 1


def test_submicrosecond_pandas_lag_keeps_original_batch_release_boundary():
    index = pd.date_range("2024-01-01", periods=3, freq="us", tz="UTC")
    output = get_reservoir_inflow(
        pd.Series([100.0, 101.0, 102.0], index=index),
        pd.Series(4.0, index=index),
        0.002,
        1.0,
        0.02,
        0.25,
        4.0,
        smoothing_lag=pd.Timedelta(1500, unit="ns"),
    )
    assert output.revised_inflow.notna().tolist() == [True, False, False]


@pytest.mark.parametrize("lag", [pd.Timedelta(10**15, unit="s"), pd.NaT])
def test_exceptional_pandas_lags_retain_original_comparison(lag):
    index = pd.date_range("2024-01-01", periods=4, freq="15min", tz="UTC")
    storage = pd.Series([100.0, 101.0, 102.0, 103.0], index=index)
    discharge = pd.Series(4.0, index=index)
    expected = get_reservoir_inflow(
        storage,
        discharge,
        0.002,
        1.0,
        0.02,
        0.25,
        4.0,
        smoothing_lag=timedelta(days=999999999),
    )
    actual = get_reservoir_inflow(
        storage, discharge, 0.002, 1.0, 0.02, 0.25, 4.0, smoothing_lag=lag
    )
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
