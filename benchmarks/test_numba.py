import pytest

from sgp4.model import WGS72
from sgp4.api import jday
from sgp4.fast.model import Satrec as NumbaSatrec, twoline2rv as numba_twoline2rv


def datetime_components(epoch):
    return (
        epoch.year,
        epoch.month,
        epoch.day,
        epoch.hour,
        epoch.minute,
        epoch.second + epoch.microsecond * 1e-6,
    )


def test_single_satellite_single_date(single_satellite_single_date_data, benchmark):
    (line1, line2), epoch, expected_r, expected_v = single_satellite_single_date_data
    jd, fr = jday(*datetime_components(epoch))

    satrec = numba_twoline2rv(NumbaSatrec(), line1, line2, WGS72)

    e, r, v = benchmark(satrec.sgp4, jd, fr)

    assert e == 0
    assert r == pytest.approx(expected_r)
    assert v == pytest.approx(expected_v)
