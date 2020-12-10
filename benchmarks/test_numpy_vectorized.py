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


def test_single_satellite_single_date(single_satellite_single_date_data, benchmark):
    (line1, line2), epoch, expected_r, expected_v = single_satellite_single_date_data
    components = datetime_components(epoch)

    satellite = vec_twoline2rv(line1, line2, wgs72)
    r, v = benchmark(satellite.propagate, *components)

    assert satellite.error == 0
    assert r == pytest.approx(expected_r)
    assert v == pytest.approx(expected_v)


def test_single_satellite_multiple_dates(
    single_satellite_multiple_dates_data, benchmark
):
    (
        (line1, line2),
        epochs,
        expected_rs,
        expected_vs,
    ) = single_satellite_multiple_dates_data
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
