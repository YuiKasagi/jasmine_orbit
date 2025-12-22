from datetime import datetime, timedelta, timezone

import pytest

from jasmine_orbit.OrbitTool import calc_obliquity, get_refdate, setdate


def test_setdate_uses_spring_reference():
    args = {'-a': False, '-s': True, '-p': '1', '-w': '2'}

    start_date, days_calc, end_date = setdate(args)

    expected_ref = datetime(2030, 3, 19, 15, 0, tzinfo=timezone.utc)
    expected_start = expected_ref + timedelta(days=1)

    assert start_date == expected_start
    assert days_calc == pytest.approx(2.0)
    assert end_date == expected_start + timedelta(days=2)


def test_setdate_uses_autumn_reference():
    args = {'-a': True, '-s': False, '-p': '0.5', '-w': '1.25'}

    start_date, days_calc, end_date = setdate(args)

    expected_ref = datetime(2030, 9, 22, 15, 0, tzinfo=timezone.utc)
    expected_start = expected_ref + timedelta(days=0.5)

    assert start_date == expected_start
    assert days_calc == pytest.approx(1.25)
    assert end_date == expected_start + timedelta(days=1.25)


def test_get_refdate_returns_utc_dates():
    spring, autumn = get_refdate()

    assert spring.tzinfo == timezone.utc
    assert autumn.tzinfo == timezone.utc
    assert spring.month == 3
    assert autumn.month == 9
    assert autumn > spring


def test_calc_obliquity_matches_known_value():
    epsilon = calc_obliquity(2451545.0)

    assert epsilon == pytest.approx(23.4392911, rel=1e-6)


def test_calc_obliquity_decreases_for_future_epochs():
    epsilon_j2000 = calc_obliquity(2451545.0)
    epsilon_future = calc_obliquity(2455197.5)

    assert epsilon_future < epsilon_j2000
