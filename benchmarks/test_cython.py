import numpy as np
from numpy.testing import assert_allclose
import pytest

from cysgp4 import PyTle, Satellite, PyDateTime, propagate_many

from sgp4.api import jday


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
    jd_l, fr_l = [], []
    for epoch in epochs:
        jd, fr = jday(*datetime_components(epoch))
        jd_l.append(jd)
        fr_l.append(fr)

    return np.array(jd_l), np.array(fr_l)


def test_single_satellite_single_date(single_satellite_single_date_data, benchmark):
    (line1, line2), epoch, expected_r, expected_v = single_satellite_single_date_data

    sat = Satellite(PyTle("_", line1, line2), None, PyDateTime(epoch))
    benchmark(sat.eci_pos)
    r = sat.eci_pos().loc
    v = sat.eci_pos().vel

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
    jd, fr = jday_from_epochs(epochs)
    mjds = (jd - 2400000.5) + fr

    tles = np.array([PyTle("_", line1, line2)])
    result = benchmark(propagate_many, mjds, tles)
    r = result["eci_pos"]
    v = result["eci_vel"]

    assert_allclose(r, expected_rs, rtol=1e-6)  # Default rtol=1e-7 makes test fail
    assert_allclose(v, expected_vs, rtol=1e-6)  # Default rtol=1e-7 makes test fail


def test_single_satellite_multiple_dates_large(
    single_satellite_multiple_dates_data_large, benchmark
):
    (
        (line1, line2),
        epochs,
        expected_shape,
    ) = single_satellite_multiple_dates_data_large
    jd, fr = jday_from_epochs(epochs)
    mjds = (jd - 2400000.5) + fr

    tles = np.array([PyTle("_", line1, line2)])
    result = benchmark(propagate_many, mjds, tles)
    r = result["eci_pos"]
    v = result["eci_vel"]

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
    jd, fr = jday_from_epochs(epochs)
    mjds = (jd - 2400000.5) + fr

    tles = np.array([PyTle("_", line1, line2) for (line1, line2) in tles])[..., None]
    result = benchmark(propagate_many, mjds, tles)
    r = result["eci_pos"]
    v = result["eci_vel"]

    assert_allclose(r, expected_rs, rtol=1e-6)  # Default rtol=1e-7 makes test fail
    assert_allclose(v, expected_vs, rtol=1e-6)  # Default rtol=1e-7 makes test fail


def test_multiple_satellites_multiple_dates_large(
    multiple_satellites_multiple_dates_data_large, benchmark
):
    (tles, epochs, expected_shape) = multiple_satellites_multiple_dates_data_large
    jd, fr = jday_from_epochs(epochs)
    mjds = (jd - 2400000.5) + fr

    tles = np.array([PyTle("_", line1, line2) for (line1, line2) in tles])[..., None]
    result = benchmark(propagate_many, mjds, tles)
    r = result["eci_pos"]
    v = result["eci_vel"]

    assert r.shape == expected_shape
    assert v.shape == expected_shape
