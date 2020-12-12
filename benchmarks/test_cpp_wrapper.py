import numpy as np
from numpy.testing import assert_allclose
import pytest

from sgp4.model import WGS72
from sgp4.api import jday
from sgp4.wrapper import (
    Satrec as CPPWrapperSatrec,
    SatrecArray as CPPWrapperSatrecArray,
)


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
    jd, fr = jday(*datetime_components(epoch))

    satrec = CPPWrapperSatrec.twoline2rv(line1, line2, WGS72)

    def f():
        return satrec.sgp4(jd, fr)

    e, r, v = benchmark(f)

    assert e == 0
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

    satrec = CPPWrapperSatrec.twoline2rv(line1, line2, WGS72)
    e, r, v = benchmark(satrec.sgp4_array, jd, fr)

    assert_allclose(e, 0)
    assert_allclose(r, expected_rs)
    assert_allclose(v, expected_vs)


def test_single_satellite_multiple_dates_large(
    single_satellite_multiple_dates_data_large, benchmark
):
    (
        (line1, line2),
        epochs,
        expected_shape,
    ) = single_satellite_multiple_dates_data_large
    jd, fr = jday_from_epochs(epochs)

    satrec = CPPWrapperSatrec.twoline2rv(line1, line2, WGS72)
    e, r, v = benchmark(satrec.sgp4_array, jd, fr)

    assert_allclose(e, 0)
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
    satellites = [CPPWrapperSatrec.twoline2rv(*tle, WGS72) for tle in tles]

    satrec_array = CPPWrapperSatrecArray(satellites)
    e, r, v = benchmark(satrec_array.sgp4, jd, fr)

    assert_allclose(e, 0)
    assert_allclose(r, expected_rs)
    assert_allclose(v, expected_vs)


def test_multiple_satellites_multiple_dates_large(
    multiple_satellites_multiple_dates_data_large, benchmark
):
    (tles, epochs, expected_shape) = multiple_satellites_multiple_dates_data_large
    jd, fr = jday_from_epochs(epochs)
    satellites = [CPPWrapperSatrec.twoline2rv(*tle, WGS72) for tle in tles]

    satrec_array = CPPWrapperSatrecArray(satellites)
    e, r, v = benchmark(satrec_array.sgp4, jd, fr)

    assert_allclose(e, 0)
    assert r.shape == expected_shape
    assert v.shape == expected_shape
