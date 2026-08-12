"""Black-box tests for batch adapter and tuning APIs."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from kalmone import (
    InflowUnits,
    InitializationStrategy,
    NoiseTuningData,
    ReservoirConfig,
    ReservoirFlowEstimate,
    UnitSystem,
    get_reservoir_inflow,
    get_reservoir_inflow_from_config,
    run_filter_with_noise,
    tune_noise,
)
from kalmone.core import _estimates_to_frame


def _tuning_config() -> ReservoirConfig:
    return ReservoirConfig(
        reservoir_id="test",
        reservoir_name="Test Reservoir",
        q=np.diag([1.0, 0.1, 0.2]),
        r=np.diag([1.0, 2.0]),
        p0=np.diag([100.0, 1000.0, 1000.0]),
        smoothing_lag=timedelta(hours=1),
        initialization_strategy=InitializationStrategy.FIRST_TWO_VALID_STORAGE,
        inflow_units=InflowUnits.CUBIC_FEET_PER_SECOND,
        model_version="test-model",
        configuration_version="test-config",
        unit_system=UnitSystem.us_customary(),
    )


def _aligned_series(count: int = 4) -> tuple[pd.Series, pd.Series]:
    index = pd.date_range("2024-01-01", periods=count, freq="15min", tz="UTC")
    storage = pd.Series([100.0 + index for index in range(count)], index=index)
    outflow = pd.Series([4.0] * count, index=index)
    return storage, outflow


class TestBatchAdapterPartitions:
    """EP/BVA for get_reservoir_inflow."""

    def test_valid_aligned_series_return_expected_columns(self) -> None:
        storage, outflow = _aligned_series()
        result = get_reservoir_inflow(
            storage,
            outflow,
            q_storage=0.1,
            q_inflow=0.1,
            q_outflow=0.1,
            r_storage=0.25,
            r_outflow=0.5,
            smoothing_lag=timedelta(minutes=15),
        )

        assert result.index.equals(storage.index)
        assert list(result.columns) == [
            "estimated_inflow",
            "estimated_outflow",
            "estimated_inflow_flag",
            "estimated_outflow_flag",
            "estimated_inflow_smoothing_flag",
            "estimated_outflow_smoothing_flag",
        ]
        assert result["estimated_inflow"].notna().all()
        assert result["estimated_outflow"].iloc[:-1].notna().all()
        assert result["estimated_outflow"].iloc[-1:].isna().all()

    def test_missing_storage_partition_is_supported(self) -> None:
        storage, outflow = _aligned_series(count=5)
        storage.iloc[2] = np.nan
        result = get_reservoir_inflow(
            storage,
            outflow,
            q_storage=0.1,
            q_inflow=0.1,
            q_outflow=0.1,
            r_storage=0.25,
            r_outflow=0.5,
            smoothing_lag=timedelta(minutes=15),
        )
        assert result.shape == (5, 6)

    def test_missing_observation_sets_predicted_flag(self) -> None:
        storage, outflow = _aligned_series(count=5)
        storage.iloc[2] = np.nan
        outflow.iloc[3] = np.nan
        result = get_reservoir_inflow(
            storage,
            outflow,
            q_storage=0.1,
            q_inflow=0.1,
            q_outflow=0.1,
            r_storage=0.25,
            r_outflow=0.5,
            smoothing_lag=timedelta(minutes=15),
        )

        assert result.loc[storage.index[0], "estimated_inflow_flag"] == "NORMAL"
        assert result.loc[storage.index[1], "estimated_inflow_flag"] == "NORMAL"
        assert result.loc[storage.index[2], "estimated_inflow_flag"] == "PREDICTED"
        assert result.loc[storage.index[3], "estimated_inflow_flag"] == "PREDICTED"
        assert result.loc[storage.index[2], "estimated_outflow_flag"] == "PREDICTED"
        assert result.loc[storage.index[3], "estimated_outflow_flag"] == "PREDICTED"
        assert (
            result["estimated_inflow_smoothing_flag"] == "NON_SMOOTHED"
        ).all()
        assert (
            result["estimated_outflow_smoothing_flag"].iloc[:-1] == "SMOOTHED"
        ).all()
        assert result["estimated_outflow_smoothing_flag"].iloc[-1] == "NON_SMOOTHED"

    def test_unequal_length_series_rejected(self) -> None:
        storage, outflow = _aligned_series(count=4)
        shorter = outflow.iloc[:3]
        with pytest.raises(ValueError):
            get_reservoir_inflow(
                storage,
                shorter,
                q_storage=0.1,
                q_inflow=0.1,
                q_outflow=0.1,
                r_storage=0.25,
                r_outflow=0.5,
            )

    def test_mismatched_series_indexes_rejected(self) -> None:
        storage, outflow = _aligned_series(count=4)
        outflow.index = outflow.index + timedelta(minutes=15)
        with pytest.raises(ValueError, match="indexes must match exactly"):
            get_reservoir_inflow(
                storage,
                outflow,
                q_storage=0.1,
                q_inflow=0.1,
                q_outflow=0.1,
                r_storage=0.25,
                r_outflow=0.5,
            )

    @pytest.mark.parametrize("r_storage", [0.0, -1.0, float("nan"), float("inf")])
    def test_invalid_measurement_variance_rejected(self, r_storage: float) -> None:
        storage, outflow = _aligned_series(count=2)
        with pytest.raises(ValueError, match="observation_covariance"):
            get_reservoir_inflow(
                storage,
                outflow,
                q_storage=0.1,
                q_inflow=0.1,
                q_outflow=0.1,
                r_storage=r_storage,
                r_outflow=0.5,
            )

    def test_initial_covariance_is_not_public_parameter(self) -> None:
        storage, outflow = _aligned_series(count=2)
        with pytest.raises(TypeError, match="unexpected keyword argument 'p0'"):
            get_reservoir_inflow(
                storage,
                outflow,
                q_storage=0.1,
                q_inflow=0.1,
                q_outflow=0.1,
                r_storage=0.25,
                r_outflow=0.5,
                p0=np.diag([10.0, 20.0]),
            )

    def test_configured_batch_api_honors_si_units(self) -> None:
        index = pd.date_range("2024-01-01", periods=3, freq="30s", tz="UTC")
        config = _tuning_config()
        config = ReservoirConfig(
            reservoir_id=config.reservoir_id,
            reservoir_name=config.reservoir_name,
            q=np.diag([1e-12, 1e-12, 1e-12]),
            r=np.diag([1e-12, 1e-12]),
            p0=config.p0,
            smoothing_lag=timedelta(seconds=30),
            initialization_strategy=config.initialization_strategy,
            inflow_units=InflowUnits.SYSTEM_FLOW_RATE,
            model_version=config.model_version,
            configuration_version=config.configuration_version,
            unit_system=UnitSystem.si(),
        )
        result = get_reservoir_inflow_from_config(
            pd.Series([100.0, 160.0, 220.0], index=index),
            pd.Series([2.0, 2.0, 2.0], index=index),
            config,
        )

        assert result.loc[index[0], "estimated_inflow"] == pytest.approx(4.0)
        assert result["estimated_outflow"].iloc[:-1].notna().all()
        assert pd.isna(result["estimated_outflow"].iloc[-1])


class TestEstimatesToFrameBoundaries:
    """Direct black-box checks for batch estimate mapping."""

    def test_estimate_timestamp_must_exist_in_index(self) -> None:
        index = pd.date_range("2024-01-01", periods=1, freq="15min", tz="UTC")
        estimate = ReservoirFlowEstimate(
            timestamp=datetime(2024, 1, 2, tzinfo=UTC),
            value=4.0,
        )
        with pytest.raises(ValueError, match="outside the input index"):
            _estimates_to_frame(
                index,
                filtered_inflows=[estimate],
                estimated_outflows=[],
            )


class TestTuningDataPartitions:
    """EP/BVA for NoiseTuningData."""

    def _valid_data(self, *, count: int = 4) -> NoiseTuningData:
        start = datetime(2024, 1, 1, tzinfo=UTC)
        timestamps = tuple(
            start + timedelta(minutes=15 * index) for index in range(count)
        )
        return NoiseTuningData(
            timestamps=timestamps,
            storage=np.array([100.0, 100.5, np.nan, 101.5][:count]),
            discharge=np.array([4.0, 4.5, 5.0, 4.0][:count]),
        )

    def test_single_timestamp_rejected(self) -> None:
        start = datetime(2024, 1, 1, tzinfo=UTC)
        with pytest.raises(ValueError, match="at least two timestamps"):
            NoiseTuningData(
                timestamps=(start,),
                storage=np.array([100.0]),
                discharge=np.array([4.0]),
            )

    def test_non_increasing_timestamps_rejected(self) -> None:
        start = datetime(2024, 1, 1, tzinfo=UTC)
        with pytest.raises(ValueError, match="strictly increasing"):
            NoiseTuningData(
                timestamps=(start, start),
                storage=np.array([100.0, 101.0]),
                discharge=np.array([4.0, 4.0]),
            )

    def test_shape_mismatch_rejected(self) -> None:
        start = datetime(2024, 1, 1, tzinfo=UTC)
        with pytest.raises(ValueError, match="storage must have shape"):
            NoiseTuningData(
                timestamps=(start, start + timedelta(minutes=15)),
                storage=np.array([100.0]),
                discharge=np.array([4.0, 4.0]),
            )


class TestRunFilterWithNoisePartitions:
    """EP for candidate noise parameters."""

    def test_positive_candidate_runs_without_scipy(self) -> None:
        data = NoiseTuningData(
            timestamps=(
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 1, 1, 0, 15, tzinfo=UTC),
                datetime(2024, 1, 1, 0, 30, tzinfo=UTC),
            ),
            storage=np.array([100.0, 100.5, 101.0]),
            discharge=np.array([4.0, 4.5, 4.0]),
        )
        result = run_filter_with_noise(
            _tuning_config(),
            data,
            q_storage=1.0,
            q_inflow=0.1,
            q_outflow=0.2,
            r_storage=1.0,
            r_outflow=2.0,
        )
        assert result.filtered_means.shape == (3, 3)

    def test_missing_discharge_is_a_missing_observation(self) -> None:
        timestamps = (
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 1, 0, 15, tzinfo=UTC),
            datetime(2024, 1, 1, 0, 30, tzinfo=UTC),
        )
        data_with_missing_discharge = NoiseTuningData(
            timestamps=timestamps,
            storage=np.array([100.0, 100.5, 101.0]),
            discharge=np.array([4.0, np.nan, 4.0]),
        )
        missing_result = run_filter_with_noise(
            _tuning_config(),
            data_with_missing_discharge,
            q_storage=1.0,
            q_inflow=0.1,
            q_outflow=0.2,
            r_storage=1.0,
            r_outflow=2.0,
        )

        assert np.isnan(missing_result.innovations[1, 1])
        assert np.isfinite(missing_result.innovations[1, 0])

    def test_missing_initial_storage_is_rejected(self) -> None:
        data = NoiseTuningData(
            timestamps=(
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 1, 1, 0, 15, tzinfo=UTC),
            ),
            storage=np.array([np.nan, 101.0]),
            discharge=np.array([4.0, 4.0]),
        )
        with pytest.raises(ValueError, match="two finite initial storage"):
            run_filter_with_noise(
                _tuning_config(),
                data,
                q_storage=1.0,
                q_inflow=0.1,
                q_outflow=0.2,
                r_storage=1.0,
                r_outflow=2.0,
            )

    @pytest.mark.parametrize(
        ("q_storage", "q_inflow", "q_outflow", "r_storage", "r_outflow"),
        [
            (0.0, 1.0, 0.2, 1.0, 2.0),
            (-1.0, 1.0, 0.2, 1.0, 2.0),
            (1.0, float("nan"), 0.2, 1.0, 2.0),
        ],
    )
    def test_nonpositive_or_nonfinite_candidates_rejected(
        self,
        q_storage: float,
        q_inflow: float,
        q_outflow: float,
        r_storage: float,
        r_outflow: float,
    ) -> None:
        data = NoiseTuningData(
            timestamps=(
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 1, 1, 0, 15, tzinfo=UTC),
            ),
            storage=np.array([100.0, 101.0]),
            discharge=np.array([4.0, 4.0]),
        )
        with pytest.raises(ValueError, match="positive and finite"):
            run_filter_with_noise(
                _tuning_config(),
                data,
                q_storage=q_storage,
                q_inflow=q_inflow,
                q_outflow=q_outflow,
                r_storage=r_storage,
                r_outflow=r_outflow,
            )


class TestTuneNoiseBoundaries:
    """BVA for tune_noise public contract."""

    @staticmethod
    def _mock_scipy_minimize() -> MagicMock:
        mock_result = MagicMock(
            x=np.log([1.0, 0.1, 0.2, 1.0, 2.0]),
            fun=1.0,
            success=True,
            nit=0,
            message="ok",
        )
        return MagicMock(return_value=mock_result)

    def test_invalid_objective_rejected(self) -> None:
        data = NoiseTuningData(
            timestamps=(
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 1, 1, 0, 15, tzinfo=UTC),
            ),
            storage=np.array([100.0, 101.0]),
            discharge=np.array([4.0, 4.0]),
        )
        with pytest.raises(ValueError, match="objective must be"):
            tune_noise(_tuning_config(), data, objective="mse")  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "initial",
        [
            (1.0, 1.0),
            (0.0, 1.0, 0.2, 1.0, 2.0),
            (-1.0, 1.0, 0.2, 1.0, 2.0),
            (float("nan"), 1.0, 0.2, 1.0, 2.0),
            (float("inf"), 1.0, 0.2, 1.0, 2.0),
        ],
    )
    def test_invalid_initial_tuple_rejected(self, initial: tuple[float, ...]) -> None:
        data = NoiseTuningData(
            timestamps=(
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 1, 1, 0, 15, tzinfo=UTC),
            ),
            storage=np.array([100.0, 101.0]),
            discharge=np.array([4.0, 4.0]),
        )
        scipy_optimize = MagicMock(minimize=self._mock_scipy_minimize())
        with patch.dict(
            "sys.modules",
            {"scipy": MagicMock(), "scipy.optimize": scipy_optimize},
        ):
            with pytest.raises(ValueError, match="initial must contain five positive"):
                tune_noise(_tuning_config(), data, initial=initial)

    def test_rmse_objective_uses_one_step_innovations(self) -> None:
        data = NoiseTuningData(
            timestamps=(
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 1, 1, 0, 15, tzinfo=UTC),
                datetime(2024, 1, 1, 0, 30, tzinfo=UTC),
            ),
            storage=np.array([100.0, 101.0, 100.5]),
            discharge=np.array([4.0, 4.0, 4.0]),
        )
        expected_filter = run_filter_with_noise(
            _tuning_config(),
            data,
            q_storage=1.0,
            q_inflow=0.1,
            q_outflow=0.2,
            r_storage=1.0,
            r_outflow=2.0,
        )
        expected_innovations = expected_filter.innovations[1:]
        expected_variances = np.diagonal(
            expected_filter.innovation_covariances[1:],
            axis1=1,
            axis2=2,
        )
        expected_innovations = expected_innovations / np.sqrt(expected_variances)
        expected_innovations = expected_innovations.ravel()
        finite_innovations = expected_innovations[np.isfinite(expected_innovations)]
        expected = float(np.sqrt(np.mean(finite_innovations**2)))

        def minimize(objective, x0, *, method, options):
            return SimpleNamespace(
                x=x0,
                fun=objective(x0),
                success=True,
                nit=1,
                message="ok",
            )

        scipy_optimize = SimpleNamespace(minimize=minimize)
        with patch.dict(
            "sys.modules",
            {"scipy": MagicMock(), "scipy.optimize": scipy_optimize},
        ):
            result = tune_noise(
                _tuning_config(),
                data,
                objective="rmse",
            )

        assert result.objective_value == pytest.approx(expected)

    def test_tune_noise_requires_scipy(self) -> None:
        data = NoiseTuningData(
            timestamps=(
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 1, 1, 0, 15, tzinfo=UTC),
            ),
            storage=np.array([100.0, 101.0]),
            discharge=np.array([4.0, 4.0]),
        )
        with patch.dict(
            "sys.modules",
            {"scipy": None, "scipy.optimize": None},
        ):
            with pytest.raises(ImportError, match="requires SciPy"):
                tune_noise(_tuning_config(), data)

    def test_tune_noise_does_not_mutate_config(self) -> None:
        config = _tuning_config()
        original_q = np.array(config.q, copy=True)
        data = NoiseTuningData(
            timestamps=(
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 1, 1, 0, 15, tzinfo=UTC),
            ),
            storage=np.array([100.0, 101.0]),
            discharge=np.array([4.0, 4.0]),
        )
        scipy_optimize = SimpleNamespace(minimize=self._mock_scipy_minimize())
        with patch.dict(
            "sys.modules",
            {"scipy": MagicMock(), "scipy.optimize": scipy_optimize},
        ):
            tune_noise(config, data)
        npt.assert_allclose(config.q, original_q)
