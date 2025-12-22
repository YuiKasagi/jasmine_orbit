from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("pandas")
pytest.importorskip("astropy")
quaternion = pytest.importorskip("quaternion")

from astropy.constants import R_earth

from jasmine_orbit import OrbitAttitude as OA
from jasmine_orbit.defaults import Config, DEFAULT_CONFIG


def test_horizon_angle_matches_manual_computation():
    expected = np.arcsin(
        R_earth.to("km").value / (R_earth.to("km").value + DEFAULT_CONFIG.ALTITUDE_KM)
    )

    assert OA.horizon_angle(DEFAULT_CONFIG) == pytest.approx(expected)


def test_horizon_angle_respects_custom_altitude():
    custom_config = Config(ALTITUDE_KM=800)
    expected = np.arcsin(
        R_earth.to("km").value / (R_earth.to("km").value + custom_config.ALTITUDE_KM)
    )

    assert OA.horizon_angle(custom_config) == pytest.approx(expected)


def test_angle_between_returns_expected_values():
    vectors_a = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    vectors_b = np.array([[0.0, 1.0, 0.0], [0.5, np.sqrt(3) / 2, 0.0]])

    result = OA.angle_between(vectors_a, vectors_b)

    assert np.allclose(result, [90.0, 60.0])


def test_prepare_orbit_uses_configuration(monkeypatch):
    args = {'-m': '5', '-p': '0', '-w': '1', '-s': True, '-a': False}
    start_date = datetime(2030, 1, 1)
    end_date = start_date + timedelta(days=1)
    captured = {}

    def fake_setdate(received_args):
        assert received_args is args
        return start_date, 1.0, end_date

    def fake_satellite_position(altitude, start, end, time_step):
        captured['altitude'] = altitude
        captured['start'] = start
        captured['end'] = end
        captured['time_step'] = time_step
        return ['orbit'], 97.6, 'L1', 'L2'

    monkeypatch.setattr(OA, "setdate", fake_setdate)
    monkeypatch.setattr(OA, "satellite_position", fake_satellite_position)

    custom_config = Config(ALTITUDE_KM=720)

    results = OA.prepare_orbit(args, config=custom_config)

    assert results == (['orbit'], 97.6, ('L1', 'L2'), start_date, 1.0, 720)
    assert captured == {
        'altitude': custom_config.ALTITUDE_KM,
        'start': start_date,
        'end': end_date,
        'time_step': timedelta(minutes=5.0),
    }


def test_first_index_after_handles_missing_value():
    series = np.array([0, 1, 2, 3, 4])

    assert OA._first_index_after(series, 1, lambda arr: arr > 2) == 3
    assert OA._first_index_after(series, 4, lambda arr: arr > 10) is None


def test_find_observation_interval_indices_finds_bounds():
    series = np.array([1, 2, 5, 3, 2, 6, 2])

    start, end, next_start = OA.find_observation_interval_indices(series, threshold=3)

    assert (start, end, next_start) == (3, 5, 6)


def test_midpoint_index_handles_missing_values():
    assert OA.midpoint_index(2, 6) == 4
    assert OA.midpoint_index(None, 6) is None


def test_compute_fraction_between_nodes_counts_matches():
    index_an = np.array([0, 5, 10])
    candidates = np.array([1, 2, 6, 7, 8])

    fractions = OA.compute_fraction_between_nodes(index_an, candidates)

    assert np.allclose(fractions, [0.4, 0.6])


def test_compute_thermal_fraction_per_orbit_handles_empty_ranges():
    index_an = np.array([0, 5, 10])
    thermal_indices = np.array([0, 1, 5, 8])

    fractions = OA.compute_thermal_fraction_per_orbit(index_an, thermal_indices)

    assert np.allclose(fractions, [0.4, 0.4])


def test_thermal_input_per_orbit_sums_matching_indices():
    index_an = np.array([0, 5, 10])
    thermal_indices = np.array([1, 5, 6, 8])
    thermal_input = np.arange(10, dtype=float)

    inputs = OA.thermal_input_per_orbit(index_an, thermal_indices, thermal_input)

    assert inputs == [thermal_input[1], thermal_input[5] + thermal_input[6] + thermal_input[8]]


def test_find_orbit_cycles_from_sat_tgt_detects_segments():
    sat_tgt = np.array([120, 80, 70, 95, 100, 85, 75, 110], dtype=float)

    cycles = OA.find_orbit_cycles_from_SatTgt(sat_tgt, obs_max_deg=90)

    assert cycles == [(1, 2, 5), (5, 6, 7)]


def test_detect_ascending_nodes_returns_times():
    def loc(lat_deg):
        return SimpleNamespace(lat=SimpleNamespace(deg=lat_deg))

    results = [
        ("t0", None, np.array([0.0, 0.0, -1.0]), loc(-5)),
        ("t1", None, np.array([0.0, 0.0, 2.0]), loc(5)),
        ("t2", None, np.array([0.0, 0.0, -1.0]), loc(-3)),
        ("t3", None, np.array([0.0, 0.0, 3.0]), loc(7)),
    ]

    ascending = OA.detect_ascending_nodes(results)

    assert ascending == ["t1", "t3"]


def test_orbattitude_handles_observation_and_non_observation(monkeypatch):
    n_points = 5
    times = [datetime(2030, 1, 1) + timedelta(minutes=i) for i in range(n_points)]
    to_sun = np.tile(np.array([1.0, 0.0, 0.0]), (n_points, 1))
    to_tgt = np.tile(np.array([0.0, 0.0, 1.0]), (n_points, 1))
    to_sat = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    sat_tgt = np.array([0.0, 0.0, 90.0, 90.0, 90.0])
    sun_tgt = np.full(n_points, 45.0)
    sat_proj = np.zeros(n_points)
    bary = to_sat.copy()
    pos = to_sat.copy()

    def fake_compute_vectors_angles_arr(results, skycoord_target):
        assert len(results) == n_points
        return (
            times,
            pos.copy(),
            to_sun.copy(),
            to_tgt.copy(),
            to_sat.copy(),
            sat_tgt.copy(),
            sun_tgt.copy(),
            sat_proj.copy(),
            bary.copy(),
            pos.copy(),
        )

    monkeypatch.setattr(OA, "compute_vectors_angles_arr", fake_compute_vectors_angles_arr)

    def fake_find_orbit_cycles(SatTgt, obs_max):
        assert obs_max == 45
        assert np.allclose(SatTgt, sat_tgt)
        return [(0, 1, 4)]

    monkeypatch.setattr(OA, "find_orbit_cycles_from_SatTgt", fake_find_orbit_cycles)

    original_angle_between = OA.angle_between

    def angle_between_rows(a, b):
        return original_angle_between(np.atleast_2d(a), np.atleast_2d(b))[0]

    monkeypatch.setattr(OA, "angle_between", angle_between_rows)

    config = Config(OBSERVATION_ANGLE_MAX_DEG=45)
    dummy_results = [(t, None, None, None) for t in times]

    (
        _,
        _,
        toSun_out,
        toTgt_out,
        _,
        SatTgt_out,
        _,
        SatX,
        SatY,
        SatZ,
        SatZSun,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = OA.orbattitude(dummy_results, object(), config=config)

    assert np.allclose(SatTgt_out, sat_tgt)
    assert np.allclose(toSun_out, to_sun)
    assert np.allclose(toTgt_out, to_tgt)

    # Observation segment should keep SatZ aligned with the target direction.
    for idx in (0, 1):
        assert np.allclose(SatZ[idx], to_tgt[idx])
        assert SatZSun[idx] == pytest.approx(90.0)

    # Non-observation segment should follow the rotation logic.
    def expected_non_observation(index):
        s_obs, e_obs, s_next = 0, 1, 4
        idx_obs = np.arange(s_obs, e_obs + 1)
        idx_non_obs = np.arange(e_obs + 1, s_next + 1)
        i_c = idx_obs[len(idx_obs) // 2]
        i_n = idx_non_obs[len(idx_non_obs) // 2]

        toSun_c = to_sun[i_c]
        SatZc = to_tgt[i_c]
        SatYc = OA.normalize(np.cross(SatZc, toSun_c))
        SatXc = np.cross(SatYc, SatZc)

        SatZn = OA.normalize(pos[i_n])
        qrot_pi = quaternion.from_rotation_vector(toSun_c * (-np.pi))
        Zno = OA.normalize(quaternion.as_rotation_matrix(qrot_pi).dot(SatZc))
        tiltAngle = -np.arccos(np.clip(np.dot(Zno, SatZn), -1.0, 1.0))
        rotdir = np.dot(toSun_c, SatZc)

        s = np.clip((index - e_obs) / max(s_next - e_obs, 1), 0.0, 1.0)
        phi = -2 * np.pi * s
        qrot_sun = quaternion.from_rotation_vector(toSun_c * phi)
        R_sun = quaternion.as_rotation_matrix(qrot_sun)

        X0 = OA.normalize(R_sun.dot(SatXc))
        Y0 = OA.normalize(R_sun.dot(SatYc))
        Z0 = OA.normalize(R_sun.dot(SatZc))

        psi = tiltAngle * s
        if rotdir < 0:
            psi = -psi
        qrot_tilt = quaternion.from_rotation_vector(Y0 * psi)
        R_tilt = quaternion.as_rotation_matrix(qrot_tilt)

        expected_x = OA.normalize(R_tilt.dot(X0))
        expected_y = Y0
        expected_z = OA.normalize(R_tilt.dot(Z0))
        return expected_x, expected_y, expected_z

    expected_x, expected_y, expected_z = expected_non_observation(4)
    assert np.allclose(SatX[4], expected_x)
    assert np.allclose(SatY[4], expected_y)
    assert np.allclose(SatZ[4], expected_z)


def test_load_target_coordinates_reads_matching_row(tmp_path):
    catalog_path = tmp_path / "targets.csv"
    catalog_path.write_text("name,ra,dec\nAlpha,10.0,20.0\nBeta,30.0,-10.0\n")

    config = Config(TARGET_CATALOG_PATH=catalog_path)

    coord, frame = OA.load_target_coordinates("Beta", config=config)

    assert coord.ra.deg == pytest.approx(30.0)
    assert coord.dec.deg == pytest.approx(-10.0)
    assert frame.iloc[0]["name"] == "Beta"


def test_load_target_coordinates_raises_on_missing_target(tmp_path):
    catalog_path = tmp_path / "targets.csv"
    catalog_path.write_text("name,ra,dec\nAlpha,10.0,20.0\n")

    config = Config(TARGET_CATALOG_PATH=catalog_path)

    with pytest.raises(ValueError):
        OA.load_target_coordinates("Missing", config=config)
