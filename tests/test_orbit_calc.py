from types import SimpleNamespace

import pytest

pytest.importorskip("numpy")
pytest.importorskip("sgp4")
pytest.importorskip("astropy")

from jasmine_orbit import OrbitCalc


def test_ltan_to_raan_uses_local_time(monkeypatch):
    class DummyAngle:
        def __init__(self, value):
            self.value = value

        def to(self, unit):
            return SimpleNamespace(value=self.value)

    def fake_get_sun(date):
        return SimpleNamespace(ra=DummyAngle(123.0))

    monkeypatch.setattr(OrbitCalc, "get_sun", fake_get_sun)

    raan = OrbitCalc.ltan_to_raan(lt=6.0, date=None)

    assert raan == pytest.approx((123.0 - 90.0) % 360)


def test_set_TLE_returns_physical_values():
    line1, line2, inclination_deg, mean_motion = OrbitCalc.set_TLE(altitude=600)

    assert line1.startswith("1 00000U 30001A")
    assert line1.endswith("9991")

    assert line2.startswith("2 00000 ")
    fields = line2.split()
    assert len(fields) == 8
    assert fields[1] == "00000"
    assert fields[4].isdigit()

    assert 0 < inclination_deg < 180
    assert mean_motion > 0
    assert pytest.approx(float(fields[7]), rel=1e-12) == mean_motion
