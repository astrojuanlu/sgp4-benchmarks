import pytest

from sgp4_vec.io import twoline2rv as vec_twoline2rv
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


def test_single_satellite_single_date(single_satellite_single_date_data, benchmark):
    (line1, line2), epoch, expected_r, expected_v = single_satellite_single_date_data
    components = datetime_components(epoch)

    satellite = vec_twoline2rv(line1, line2, wgs72)
    r, v = benchmark(satellite.propagate, *components)

    assert satellite.error == 0
    assert r == pytest.approx(expected_r)
    assert v == pytest.approx(expected_v)
