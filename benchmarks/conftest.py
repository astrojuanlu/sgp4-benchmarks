import datetime as dt

import pytest


@pytest.fixture
def single_satellite_single_date_data():
    tle = (
        "1 41557U 16033B   20345.20030338  .00003290  00000-0  12071-3 0  9996",
        "2 41557  97.3998  74.3002 0013100 179.2679 265.4184 15.28602096252616",
    )
    epoch = dt.datetime(2020, 12, 11, 12, 0, 0)
    expected_r = (1902.017829907682, 5514.393345746716, 3616.812182638985)
    expected_v = (-0.20735506463872982, -4.140722397852882, 6.391434915103069)

    return tle, epoch, expected_r, expected_v
