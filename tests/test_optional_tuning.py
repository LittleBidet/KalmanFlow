"""Verify optional tuning imports in fresh Python processes."""

import subprocess
import sys
import textwrap

import pytest


def _run_python(source: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_filtering_and_evaluation_do_not_import_optimizer_dependencies() -> None:
    _run_python("""
        import importlib.abc
        import sys

        class NoOptimizerImports(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in {'scipy', 'sklearn'}:
                    raise AssertionError(f'Unexpected optional import: {fullname}')

        sys.meta_path.insert(0, NoOptimizerImports())
        from datetime import timedelta
        import numpy as np
        import pandas as pd
        from kalmanflow import (
            BayesianEvaluationSettings, InflowUnits, InitializationStrategy,
            ReservoirConfig, evaluate_configuration, get_reservoir_inflow,
            tune_inflow_noise_bayesian,
        )

        index = pd.date_range('2026-01-01', periods=6, freq='h', tz='UTC')
        storage = pd.Series(1000.0 + np.arange(6), index=index)
        discharge = pd.Series(10.0, index=index)
        result = get_reservoir_inflow(storage, discharge, 1, 1, 1, 1, 1)
        assert result.estimated_inflow.notna().all()
        config = ReservoirConfig(
            'demo', 'Demo', np.eye(3), np.eye(2), np.eye(3), timedelta(hours=1),
            InitializationStrategy.FIRST_TWO_VALID_STORAGE,
            InflowUnits.CUBIC_FEET_PER_SECOND, 'v1', 'c1',
        )
        evaluation = evaluate_configuration(
            storage, discharge, config,
            settings=BayesianEvaluationSettings(warmup=timedelta(0)),
        )
        assert evaluation.window_diagnostics.iloc[0]['storage_count'] == 4
        assert 'kalmanflow.bayesian_tuning._optimization' not in sys.modules
    """)


@pytest.mark.parametrize("missing", ["scipy", "sklearn"])
def test_missing_tuning_dependency_has_install_guidance(missing: str) -> None:
    _run_python(f"""
        import importlib.abc
        import sys

        class MissingDependency(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == {missing!r}:
                    raise ModuleNotFoundError('simulated missing dependency',
                                              name=fullname)

        sys.meta_path.insert(0, MissingDependency())
        from kalmanflow import tune_inflow_noise_bayesian
        try:
            tune_inflow_noise_bayesian(None, None, None, [], [])
        except ImportError as error:
            assert "pip install 'kalmanflow[tuning]'" in str(error)
            assert error.__cause__.name == {missing!r}
        else:
            raise AssertionError('Expected an actionable missing-extra error')
    """)
