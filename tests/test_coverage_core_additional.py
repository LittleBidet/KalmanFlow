"""Contract and failure-path coverage for the core public adapters."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from kalmanflow import (
    InflowUnits,
    InitializationStrategy,
    OnlineReservoirInflow,
    OutputFlag,
    ReservoirConfig,
    add_inflow_uncertainty_intervals,
    core,
    get_reservoir_inflow,
    get_reservoir_inflow_from_config,
)

TIMESTAMP = datetime(2024, 1, 1, tzinfo=UTC)


def _config() -> ReservoirConfig:
    return ReservoirConfig(
        reservoir_id="coverage",
        reservoir_name="Coverage Reservoir",
        q=np.diag([0.1, 0.1, 0.1]),
        r=np.diag([0.25, 0.5]),
        p0=np.diag([100.0, 1000.0, 1000.0]),
        smoothing_lag=timedelta(hours=1),
        initialization_strategy=InitializationStrategy.FIRST_TWO_VALID_STORAGE,
        inflow_units=InflowUnits.CUBIC_FEET_PER_SECOND,
        model_version="coverage-model",
        configuration_version="coverage-config",
    )


def _backend() -> object:
    return core._build_default_backend(
        q_storage=0.1,
        q_inflow=0.1,
        q_outflow=0.1,
        r_storage=0.25,
        r_outflow=0.5,
    )


def test_core_scalar_validation_and_interval_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(TypeError, match="include_uncertainty must be a bool"):
        core._validate_include_uncertainty(1)

    class _InfiniteNormal:
        def inv_cdf(self, value: float) -> float:
            del value
            return float("inf")

    monkeypatch.setattr(core, "NormalDist", lambda: _InfiniteNormal())
    with pytest.raises(ValueError, match="finite normal interval"):
        core._normal_interval_multiplier(0.95)

    with pytest.raises(ValueError, match="inflow covariance entry"):
        core._inflow_standard_deviation(np.ones((1, 1)), name="covariance")
    with pytest.raises(ValueError, match="only finite"):
        core._inflow_standard_deviation(
            np.array([[1.0, 0.0], [0.0, np.inf]]), name="covariance"
        )
    with pytest.raises(ValueError, match="materially negative"):
        core._inflow_standard_deviation(
            np.array([[1.0, 0.0], [0.0, -1.0]]), name="covariance"
        )

    with pytest.raises(ValueError, match="inflow covariance entries"):
        core._inflow_standard_deviations(np.ones((2, 1, 1)), name="covariances")
    with pytest.raises(ValueError, match="only finite"):
        core._inflow_standard_deviations(
            np.array([[[1.0, 0.0], [0.0, np.inf]]]), name="covariances"
        )
    with pytest.raises(ValueError, match="materially negative"):
        core._inflow_standard_deviations(
            np.array([[[1.0, 0.0], [0.0, -1.0]]]), name="covariances"
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"timestamp": object()}, "timestamp must be a datetime"),
        ({"timestamp": datetime(2024, 1, 1)}, "timezone-aware"),
        ({"prediction_flag": OutputFlag.SMOOTHED}, "prediction_flag"),
        ({"smoothing_flag": OutputFlag.PREDICTED}, "smoothing_flag"),
        ({"value": float("inf")}, "value must be finite"),
        ({"standard_deviation": float("inf")}, "standard_deviation must be finite"),
        ({"standard_deviation": -1.0}, "standard_deviation must be non-negative"),
    ],
)
def test_reservoir_flow_estimate_rejects_invalid_fields(
    kwargs: dict[str, object], message: str
) -> None:
    values: dict[str, object] = {"timestamp": TIMESTAMP, "value": 2.0}
    values.update(kwargs)
    with pytest.raises((TypeError, ValueError), match=message):
        core.ReservoirFlowEstimate(**values)


def test_online_stream_argument_forms_and_preprocessing_errors() -> None:
    stream = OnlineReservoirInflow(
        q_storage=0.1,
        q_inflow=0.1,
        q_outflow=0.1,
        r_storage=0.25,
        r_outflow=0.5,
    )
    with pytest.raises(TypeError, match="either an observation object"):
        stream.process(
            SimpleNamespace(timestamp=TIMESTAMP, storage=100.0, discharge=4.0),
            timestamp=TIMESTAMP,
        )
    with pytest.raises(TypeError, match="requires timestamp"):
        stream.process()
    with pytest.raises(TypeError, match="must provide timestamp"):
        stream.process(SimpleNamespace(timestamp=TIMESTAMP, storage=100.0))
    with pytest.raises(TypeError, match="timestamp must be a datetime"):
        stream.process(timestamp="bad", storage=100.0, discharge=4.0)

    assert stream.process_many(()) == ()
    stream.process(timestamp=TIMESTAMP, storage=100.0, discharge=4.0)
    with pytest.raises(ValueError, match="strictly increasing"):
        stream.process(timestamp=TIMESTAMP, storage=101.0, discharge=4.0)
    with pytest.raises(RuntimeError, match="successful process result"):
        stream.checkpoint()


def test_online_stream_checkpoint_and_reentrancy_guards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = OnlineReservoirInflow(
        q_storage=0.1,
        q_inflow=0.1,
        q_outflow=0.1,
        r_storage=0.25,
        r_outflow=0.5,
    )
    stream.process(timestamp=TIMESTAMP, storage=100.0, discharge=4.0)
    with pytest.raises(RuntimeError, match="reservoir_id"):
        stream.checkpoint()

    stream._processing = True
    stream._checkpoint_ready = True
    stream._mark_preprocessing_failure()
    assert stream._checkpoint_ready is True
    with pytest.raises(RuntimeError, match="may not be re-entered"):
        stream.process_many(
            (
                SimpleNamespace(
                    timestamp=TIMESTAMP + timedelta(minutes=15),
                    storage=101.0,
                    discharge=4.0,
                ),
            )
        )
    stream._processing = True
    with pytest.raises(RuntimeError, match="may not be re-entered"):
        stream._process_one_value((TIMESTAMP + timedelta(minutes=15), 101.0, 4.0))
    stream._processing = False

    with pytest.raises(TypeError, match="timestamp must be a datetime"):
        stream._validate_complete_order((("bad", 1.0, 2.0),))

    monkeypatch.setattr(
        stream._pipeline,
        "export_state",
        lambda: (_ for _ in ()).throw(ValueError("snapshot failed")),
    )
    with pytest.raises(ValueError, match="snapshot failed"):
        stream._process_many_values(((TIMESTAMP + timedelta(minutes=15), 101.0, 4.0),))

    monkeypatch.setattr(stream._pipeline, "export_state", lambda: object())
    monkeypatch.setattr(
        stream._pipeline,
        "process",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("processing failed")),
    )
    monkeypatch.setattr(
        stream._pipeline,
        "restore_state",
        lambda state: (_ for _ in ()).throw(RuntimeError("restore failed")),
    )
    with pytest.raises(RuntimeError, match="could not be restored"):
        stream._process_many_values(((TIMESTAMP + timedelta(minutes=15), 101.0, 4.0),))


def test_batch_validation_and_empty_initialization_paths() -> None:
    index = pd.date_range(TIMESTAMP, periods=2, freq="h")
    storage = pd.Series([100.0, 101.0], index=index)
    discharge = pd.Series([4.0, 4.0], index=index)
    model_args = (0.1, 0.1, 0.1, 0.25, 0.5)

    with pytest.raises(ValueError, match="indexes must match"):
        get_reservoir_inflow(
            storage, discharge.set_axis(index + timedelta(hours=1)), *model_args
        )
    with pytest.raises(ValueError, match="indexes must match"):
        get_reservoir_inflow_from_config(
            storage, discharge.set_axis(index + timedelta(hours=1)), _config()
        )

    with pytest.raises(ValueError, match="lag must be positive"):
        core._process_reservoir_batch_raw(
            index,
            storage.to_numpy(),
            discharge.to_numpy(),
            backend=_backend(),
            smoothing_lag=timedelta(0),
            max_window_steps=2,
            include_uncertainty=False,
        )
    with pytest.raises(ValueError, match="matching lengths"):
        core._process_reservoir_batch_raw(
            index,
            storage.to_numpy()[:1],
            discharge.to_numpy(),
            backend=_backend(),
            smoothing_lag=timedelta(hours=1),
            max_window_steps=2,
            include_uncertainty=False,
        )

    no_storage = core._process_reservoir_batch_raw(
        index,
        np.array([np.nan, np.nan]),
        discharge.to_numpy(),
        backend=_backend(),
        smoothing_lag=timedelta(hours=1),
        max_window_steps=2,
        include_uncertainty=False,
    )
    assert no_storage["estimated_inflow"].isna().all()

    one_storage = core._process_reservoir_batch_raw(
        index,
        np.array([100.0, np.nan]),
        discharge.to_numpy(),
        backend=_backend(),
        smoothing_lag=timedelta(hours=1),
        max_window_steps=2,
        include_uncertainty=False,
    )
    assert one_storage["estimated_inflow"].isna().all()

    with pytest.raises(ValueError, match="needs a finite discharge"):
        core._process_reservoir_batch_raw(
            index,
            np.array([np.nan, 100.0]),
            np.array([4.0, np.nan]),
            backend=_backend(),
            smoothing_lag=timedelta(hours=1),
            max_window_steps=2,
            include_uncertainty=False,
        )

    duplicate_index = pd.DatetimeIndex([TIMESTAMP, TIMESTAMP])
    with pytest.raises(ValueError, match="strictly increasing"):
        core._process_reservoir_batch_raw(
            duplicate_index,
            np.array([100.0, 101.0]),
            np.array([4.0, 4.0]),
            backend=_backend(),
            smoothing_lag=timedelta(hours=1),
            max_window_steps=2,
            include_uncertainty=False,
        )


def test_batch_interval_validation_and_configuration_fingerprint() -> None:
    frame = pd.DataFrame(
        {
            "estimated_inflow": [1.0, np.nan],
            "revised_inflow": [1.5, np.nan],
            "estimated_inflow_standard_deviation": [0.1, np.nan],
            "revised_inflow_standard_deviation": [0.2, np.nan],
        }
    )
    output = add_inflow_uncertainty_intervals(frame)
    assert output is not frame
    assert output.loc[0, "estimated_inflow_lower"] < 1.0
    assert pd.isna(output.loc[1, "revised_inflow_upper"])
    with pytest.raises(TypeError, match="DataFrame"):
        add_inflow_uncertainty_intervals(object())
    with pytest.raises(ValueError, match="uncertainty columns"):
        add_inflow_uncertainty_intervals(frame.drop(columns="revised_inflow"))

    with pytest.raises(ValueError, match="matching lengths"):
        core._validate_batch_interval_inputs(
            np.array([1.0]), np.array([0.1, 0.2]), prefix="estimated_inflow"
        )
    with pytest.raises(ValueError, match="only finite values"):
        core._validate_batch_interval_inputs(
            np.array([np.inf]), np.array([0.1]), prefix="estimated_inflow"
        )
    with pytest.raises(ValueError, match="only finite values"):
        core._validate_batch_interval_inputs(
            np.array([1.0]), np.array([np.inf]), prefix="estimated_inflow"
        )
    with pytest.raises(ValueError, match="missing together"):
        core._validate_batch_interval_inputs(
            np.array([1.0]), np.array([np.nan]), prefix="estimated_inflow"
        )
    with pytest.raises(ValueError, match="non-negative"):
        core._validate_batch_interval_inputs(
            np.array([1.0]), np.array([-0.1]), prefix="estimated_inflow"
        )

    with pytest.raises(TypeError, match="ReservoirStateSpaceModel"):
        core._configuration_fingerprint(
            SimpleNamespace(model=object()), timedelta(hours=1)
        )
