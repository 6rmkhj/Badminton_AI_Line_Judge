"""Fit reuse preserves measured trajectories and isolates cached arrays."""
import numpy as np

from badminton.tracking import _flight_fit, _flight_coefficients


def samples(times):
    return [{'time': t, 'x': 7 + 2 * t + .2 * t * t,
             'y': 4 + 3 * t + .5 * t * t} for t in times]


def test_cached_fit_uses_exact_timestamps_and_predicts_without_changing_observations():
    observations = samples([0., .1, .2, .3, .5, .7])
    original = [dict(o) for o in observations]
    first, velocity, error = _flight_fit(observations, 1.1)
    np.testing.assert_allclose(first, [9.442, 7.905], atol=1e-12, rtol=0)
    assert error < 1e-12
    velocity[:] = 999  # Caller-owned output must not poison later hypotheses.
    second, next_velocity, next_error = _flight_fit(observations, 1.3)
    np.testing.assert_allclose(second, [9.938, 8.745], atol=1e-12, rtol=0)
    np.testing.assert_allclose(next_velocity, [2.28, 3.7], atol=1e-12, rtol=0)
    assert next_error == error
    assert observations == original
    changed_times = [{**o, 'time': o['time'] * 2} for o in observations]
    changed, _, _ = _flight_fit(changed_times, 1.3)
    assert not np.allclose(changed, second)


def test_fit_reuse_matches_uncached_least_squares_for_irregular_noisy_histories():
    rng = np.random.default_rng(128)
    for length in (2, 3, 4, 6, 12):
        times = np.cumsum(rng.uniform(.01, .08, length))
        points = rng.normal(0, 1, (length, 2)) + times[:, None] * np.array([4., 50.])
        observations = [dict(time=float(t), x=float(p[0]), y=float(p[1])) for t, p in zip(times, points)]
        degree = 2 if length >= 4 else 1
        local_times = times - times[-1]
        design = np.column_stack([local_times ** i for i in range(degree + 1)])
        fit = np.linalg.lstsq(design, points, rcond=None)[0]
        stamp = float(times[-1] + .04)
        expected = np.array([(stamp - times[-1]) ** i for i in range(degree + 1)]) @ fit
        for _ in range(2):
            actual, velocity, residual = _flight_fit(observations, stamp, window=length)
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(velocity, fit[1])
            assert residual == float(np.sqrt(np.mean(np.sum((design @ fit - points) ** 2, axis=1))))
    assert _flight_coefficients.cache_info().currsize <= 4096
