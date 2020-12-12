import numpy as np
from numpy.testing import assert_allclose
import pytest

from sgp4_vec.io import twoline2rv as vec_twoline2rv
from sgp4_vec.model import minutes_per_day
from sgp4_vec.ext import jday
from sgp4_vec.propagation import sgp4
from sgp4_vec.earth_gravity import wgs72


def datetime_components(epoch):
    return (
        epoch.year,
        epoch.month,
        epoch.day,
        epoch.hour,
        epoch.minute,
        epoch.second + epoch.microsecond * 1e-6,
    )


def jday_from_epochs(epochs):
    jd_l = []
    for epoch in epochs:
        jd = jday(*datetime_components(epoch))
        jd_l.append(jd)

    return np.array(jd_l)


# Custom function, not present in the original implementation
def numpy_sgp4_many(satellites, jd, whichconst=None):
    n = len(satellites)
    m = len(jd)

    r_array = np.zeros((n, m, 3))
    v_array = np.zeros((n, m, 3))

    for ii in range(n):
        satellite = satellites[ii]

        (rx, ry, rz), (vx, vy, vz) = sgp4(
            satellite,
            (jd - satellite.jdsatepoch) * minutes_per_day,
            whichconst,
        )
        r_array[ii] = np.array([rx, ry, rz]).T
        v_array[ii] = np.array([vx, vy, vz]).T

    return r_array, v_array


def test_single_satellite_single_date(single_satellite_single_date_data, benchmark):
    (line1, line2), epoch, expected_r, expected_v = single_satellite_single_date_data
    components = datetime_components(epoch)

    satellite = vec_twoline2rv(line1, line2, wgs72)

    def f():
        return satellite.propagate(*components)

    r, v = benchmark(f)

    assert satellite.error == 0
    assert r == pytest.approx(expected_r)
    assert v == pytest.approx(expected_v)


def test_single_satellite_multiple_dates_medium(
    single_satellite_multiple_dates_data_medium, benchmark
):
    (
        (line1, line2),
        epochs,
        expected_rs,
        expected_vs,
    ) = single_satellite_multiple_dates_data_medium
    jd = jday_from_epochs(epochs)

    satellite = vec_twoline2rv(line1, line2, wgs72)
    (rx, ry, rz), (vx, vy, vz) = benchmark(
        sgp4, satellite, (jd - satellite.jdsatepoch) * minutes_per_day
    )
    r = np.array([rx, ry, rz]).T
    v = np.array([vx, vy, vz]).T

    assert_allclose(satellite.error, 0)
    assert_allclose(r, expected_rs, rtol=1e-5)  # Default rtol=1e-7 makes test fail
    assert_allclose(v, expected_vs, rtol=1e-5)  # Default rtol=1e-7 makes test fail


def test_single_satellite_multiple_dates_large(
    single_satellite_multiple_dates_data_large, benchmark
):
    (
        (line1, line2),
        epochs,
        expected_shape,
    ) = single_satellite_multiple_dates_data_large
    jd = jday_from_epochs(epochs)

    satellite = vec_twoline2rv(line1, line2, wgs72)
    (rx, ry, rz), (vx, vy, vz) = benchmark(
        sgp4, satellite, (jd - satellite.jdsatepoch) * minutes_per_day
    )
    r = np.array([rx, ry, rz]).T
    v = np.array([vx, vy, vz]).T

    assert_allclose(satellite.error, 0)
    assert r.shape == expected_shape
    assert v.shape == expected_shape


def test_multiple_satellites_multiple_dates_medium(
    multiple_satellites_multiple_dates_data_medium, benchmark
):
    (
        tles,
        epochs,
        expected_rs,
        expected_vs,
    ) = multiple_satellites_multiple_dates_data_medium
    jd = jday_from_epochs(epochs)

    satellites = [vec_twoline2rv(*tle, wgs72) for tle in tles]
    r, v = benchmark(numpy_sgp4_many, satellites, jd)

    assert all(satellite.error == 0 for satellite in satellites)
    assert_allclose(r, expected_rs, rtol=1e-5)  # Default rtol=1e-7 makes test fail
    assert_allclose(v, expected_vs, rtol=1e-5)  # Default rtol=1e-7 makes test fail


def test_multiple_satellites_multiple_dates_large(
    multiple_satellites_multiple_dates_data_large, benchmark
):
    (tles, epochs, expected_shape) = multiple_satellites_multiple_dates_data_large
    jd = jday_from_epochs(epochs)

    satellites = [vec_twoline2rv(*tle, wgs72) for tle in tles]
    r, v = benchmark(numpy_sgp4_many, satellites, jd)

    assert all(satellite.error == 0 for satellite in satellites)
    assert r.shape == expected_shape
    assert v.shape == expected_shape
