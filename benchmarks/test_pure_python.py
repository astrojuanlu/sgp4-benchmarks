import pytest

from sgp4.model import Satrec as PurePythonSatrec, WGS72
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

    return jd_l, fr_l


def test_single_satellite_single_date(single_satellite_single_date_data, benchmark):
    (line1, line2), epoch, expected_r, expected_v = single_satellite_single_date_data
    jd, fr = jday(*datetime_components(epoch))

    satrec = PurePythonSatrec.twoline2rv(line1, line2, WGS72)
    e, r, v = benchmark(satrec.sgp4, jd, fr)

    assert e == 0
    assert r == pytest.approx(expected_r)
    assert v == pytest.approx(expected_v)
