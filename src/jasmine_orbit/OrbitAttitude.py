from datetime import timedelta
from jasmine_orbit.OrbitCalc import satellite_position
from jasmine_orbit.OrbitTool import setdate
from jasmine_orbit.ViewOrbit import normalize, view_sat
from jasmine_orbit.defaults import Config, DEFAULT_CONFIG
import numpy as np
from astropy.time import Time
from astropy.coordinates import get_sun, SkyCoord, TEME, GCRS
import astropy.units as u
import astropy.constants as c
import quaternion
import pandas as pd
from pathlib import Path
from typing import Callable, List, Optional, Tuple


# ErfaWarning と AstropyWarning を非表示にする
import warnings
from astropy.utils.exceptions import ErfaWarning, AstropyWarning
warnings.filterwarnings('ignore', category=ErfaWarning)
warnings.filterwarnings('ignore', category=AstropyWarning)


Comparator = Callable[[np.ndarray], np.ndarray]


def prepare_orbit(args, config: Config = DEFAULT_CONFIG):
    """Prepare satellite orbit data based on command-line arguments.
    Args:
        args: Command-line arguments.
    
    Returns:
        Tuple containing orbit results, inclination in degrees, TLE lines, start date, calculation days, and altitude.
    """
    start_date, days_calc, end_date = setdate(args)
    altitude = config.ALTITUDE_KM
    print(f"衛星高度: {altitude} km")
    time_step = timedelta(minutes=float(args['-m']))
    results, inclination_deg, tle1, tle2 = satellite_position(
        altitude, start_date, end_date, time_step
    )
    return results, inclination_deg, (tle1, tle2), start_date, days_calc, altitude

def horizon_angle(config: Config = DEFAULT_CONFIG):
    altitude = config.ALTITUDE_KM
    R_earth = c.R_earth.to("km").value
    return np.arcsin(R_earth / (R_earth + altitude))

def load_target_coordinates(target_name: str, config: Config = DEFAULT_CONFIG):
    """Load target coordinates from a catalog CSV file.
    Args:
        target_name: Name of the target to search for.
        catalog_path: Path to the catalog CSV file.
        
    Returns:
        Tuple containing the SkyCoord of the target and the corresponding DataFrame row.
    """
    catalog = pd.read_csv(config.TARGET_CATALOG_PATH)
    target_data = catalog.loc[catalog["name"] == target_name]
    if target_data.empty:
        raise ValueError(f"Target '{target_name}' not found in {config.TARGET_CATALOG_PATH}")
    row = target_data.iloc[0]
    coord = SkyCoord(ra=row["ra"] * u.degree, dec=row["dec"] * u.degree, frame="icrs")
    return coord, target_data

def angle_between(a, b):
    """Calculate the angle in degrees between two vectors a and b."""
    dot = np.sum(a * b, axis=-1)
    dot = np.clip(dot, -1.0, 1.0)
    result = np.degrees(np.arccos(dot))
    return result.item() if result.ndim == 0 else result

def _first_index_after(series: np.ndarray, start: int, comparator: Comparator) -> Optional[int]:
    """Find the first index in 'series' after 'start' where 'comparator' returns True."""
    sub_series = series[start:]
    mask = comparator(sub_series)
    if not np.any(mask):
        return None
    return start + int(np.argmax(mask))


def find_observation_interval_indices(series: np.ndarray, threshold: float) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """Find the start, end, and next start indices of intervals where series values are below a threshold.
    Args:
        series: Numpy array of values.
        threshold: Threshold value to compare against.
    
    Returns:
        Tuple of (start index, end index, next start index) or (None, None, None) if not found.
    """
    first_above = _first_index_after(series, 0, lambda arr: arr > threshold)
    if first_above is None:
        return None, None, None
    start = _first_index_after(series, first_above, lambda arr: arr <= threshold)
    if start is None:
        return None, None, None
    end = _first_index_after(series, start, lambda arr: arr > threshold)
    next_start = _first_index_after(series, end, lambda arr: arr <= threshold) if end is not None else None
    return start, end, next_start


def midpoint_index(start_idx: Optional[int], end_idx: Optional[int]) -> Optional[int]:
    """Calculate the midpoint index between start_idx and end_idx.
    Args:
        start_idx: Start index.
        end_idx: End index.
    
    Returns:
        Midpoint index or None if either index is None.
    """
    if start_idx is None or end_idx is None:
        return None
    return (start_idx + end_idx) // 2


def compute_fraction_between_nodes(index_an: np.ndarray, candidate_indices: np.ndarray) -> float:
    """Compute the fraction of candidate indices between ascending node indices.
    Args:
        index_an: Numpy array of ascending node indices.
        candidate_indices: Numpy array of candidate indices.

    Returns:
        List of fractions for each orbit segment.
    """
    if len(index_an) < 2:
        return 0.0
    fractions = []
    for start, end in zip(index_an[:-1], index_an[1:]):
        if end <= start:
            fractions.append(0.0)
            continue
        orbit_range = np.arange(start, end)
        len_orb = len(orbit_range)
        if len_orb == 0:
            fractions.append(0.0)
            continue
        len_obs = len(np.intersect1d(orbit_range, candidate_indices))
        fractions.append(len_obs / len_orb)
    return fractions


def compute_thermal_fraction_per_orbit(index_an: np.ndarray, thermal_indices: np.ndarray) -> List[float]:
    """Compute the fraction of thermal unfeasible time per orbit segment.
    
    Args:
        index_an: Numpy array of ascending node indices.
        thermal_indices: Numpy array of indices where thermal conditions are unfeasible.
    Returns:
        List of fractions for each orbit segment.
    """
    fractions: List[float] = []
    for start, end in zip(index_an[:-1], index_an[1:]):
        if end <= start:
            fractions.append(0.0)
            continue
        orbit_range = np.arange(start, end)
        if len(orbit_range) == 0:
            fractions.append(0.0)
            continue
        len_orb = len(orbit_range) 
        if len_orb == 0:
            fractions.append(0.0)
            continue
        len_thermal = len(np.intersect1d(orbit_range, thermal_indices))
        fractions.append(len_thermal / len_orb)
    return fractions

import matplotlib.pyplot as plt
def thermal_input_per_orbit(index_an: np.ndarray, thermal_indices: np.array, thermal_input: np.ndarray) -> List[float]:
    input_per_orbit = []
    for start, end in zip(index_an[:-1], index_an[1:]):
        if end <= start:
            input_per_orbit.append(0.0)
            continue
        orbit_range = np.arange(start, end)
        if len(orbit_range) == 0:
            input_per_orbit.append(0.0)
            continue
        len_orb = len(orbit_range) 
        if len_orb == 0:
            input_per_orbit.append(0.0)
            continue
        input_ind_tmp = np.intersect1d(orbit_range, thermal_indices)
        input_sum = np.sum(thermal_input[input_ind_tmp])
        input_per_orbit.append(input_sum)
    return input_per_orbit

def compute_visibility(lat, lon, results, time_an, config: Config = DEFAULT_CONFIG):
    """Compute visibility fraction for a given latitude and longitude.
    Args:
        lat: Latitude in degrees.
        lon: Longitude in degrees.
        results: Satellite orbit results.
        time_an: List of ascending node times.
    
     Returns:
        Fraction of observation time.
    """
    skycoord_target = SkyCoord(lon=lon * u.degree, lat=lat * u.degree, frame="geocentricmeanecliptic")
    try:
        times, _, _, _, _, SatTgt, _, _, _, _, SatZSun, _, _, _, _, _, _ = orbattitude(results, skycoord_target)
        sun_min, sun_max = config.THERMAL_SUN_ANGLE_RANGE_DEG
        mask_obs = (SatTgt <= config.OBSERVATION_ANGLE_MAX_DEG) & (sun_min <= SatZSun) & (SatZSun <= sun_max)
        index_obs = np.where(mask_obs)[0]

        times_array = np.asarray(times)
        index_an = np.where(np.isin(times_array, np.asarray(time_an)))[0]
        fractions_arr = compute_fraction_between_nodes(index_an, index_obs)
        return np.nanmean(fractions_arr)
    except Exception:
        return 0.0

def compute_visibility_from_args(args):
    """Wrapper function to compute visibility from argument tuple."""
    lat, lon, results, time_an = args
    return compute_visibility(lat, lon, results, time_an)

def compute_vectors_angles_arr(results, skycoord_target):
    """Compute satellite vectors and angles for given orbit results and target coordinates.
    Args:
        results: Satellite orbit results.
        skycoord_target: SkyCoord of the target.
    Returns:
        Tuple containing times, satellite positions, various angles, and direction vectors.
    """
    times = [res[0] for res in results]
    r_arr  = np.array([res[1] for res in results], dtype=float)
    v_arr  = np.array([res[2] for res in results], dtype=float)

    astrotime = Time(times, scale='utc')

    # ---- TEME -> ICRS / GCRS を一括変換 ----
    sat_teme = SkyCoord(
        x=r_arr[:, 0] * u.km,
        y=r_arr[:, 1] * u.km,
        z=r_arr[:, 2] * u.km,
        v_x=v_arr[:, 0] * u.km / u.s,
        v_y=v_arr[:, 1] * u.km / u.s,
        v_z=v_arr[:, 2] * u.km / u.s,
        frame=TEME(obstime=astrotime),
        representation_type='cartesian',
        differential_type='cartesian',
    )

    sat_icrs = sat_teme.icrs
    sat_gcrs = sat_teme.transform_to(GCRS(obstime=astrotime))

    # 衛星位置系
    Sat = r_arr.copy()  # 衛星位置

    # 地心→衛星方向ベクトル（GCRS）
    pos_gcrs = sat_gcrs.cartesian.xyz.to_value(u.km).T
    v_gcrs   = sat_gcrs.velocity.d_xyz.to_value(u.km / u.s).T
    toSat    = normalize(pos_gcrs) # 地心から衛星方向

    # Barycenter -> Sat （ICRS, AU）
    BarytoSat = sat_icrs.cartesian.xyz.to(u.AU).value.T  # barycenter から衛星方向

    # 太陽方向
    sun_gcrs = get_sun(astrotime)
    toSun = normalize(sun_gcrs.cartesian.xyz.value.T)    # 太陽方向

    # ターゲット方向
    target_icrs = skycoord_target.icrs
    target_gcrs = target_icrs.transform_to(GCRS(obstime=astrotime))
    toTgt = normalize(target_gcrs.cartesian.xyz.value.T)  # ターゲット方向

    SatTgt = angle_between(toSat, toTgt) # 地心から衛星方向 と ターゲット方向 がなす角 (0-180 deg)
    SunTgt = angle_between(toSun, toTgt) # 太陽方向 と ターゲット方向 がなす角 (0-180 deg)

    n_orbit = normalize(np.cross(pos_gcrs, v_gcrs)) # 軌道面法線ベクトル
    dot_tgt_norm = np.sum(toTgt * n_orbit, axis=1)
    tgt_proj = toTgt - dot_tgt_norm[:, None] * n_orbit
    projected_to_tgt = normalize(tgt_proj)
    SatprojTgt = angle_between(projected_to_tgt, toSat) # ターゲットベクトルを軌道面に射影したベクトル と 衛星方向 がなす角

    return times, Sat, toSun, toTgt, toSat, SatTgt, SunTgt, SatprojTgt, BarytoSat, pos_gcrs

def init_angles(n_results):
    """Initialize arrays for satellite attitude angles.
    Args:
        n_results: Number of results.
    Returns:
        Tuple of initialized numpy arrays for various angles and directions.
    """
    SatZSun = np.empty(n_results)  # 衛星指向方向と太陽方向がなす角 (0-180 deg)
    SatXSun = np.empty(n_results)  # 衛星X方向と太陽方向がなす角 (0-180 deg)
    SatYSun = np.empty(n_results)  # 衛星Y方向と太陽方向がなす角 (0-180 deg)
    SatX = np.empty((n_results, 3))
    SatY = np.empty((n_results, 3))
    SatZ = np.empty((n_results, 3))
    toSatZn = np.empty(n_results)  # 地心から衛星方向 の天頂角, 衛星XYZ座標 (0-180 deg)
    toSatAz = np.empty(n_results)  # 地心から衛星方向 の方位角, 衛星XYZ座標 (-180-180 deg)
    return SatZSun, SatXSun, SatYSun, SatX, SatY, SatZ, toSatZn, toSatAz

def orbattitude_old(results, skycoord_target, config: Config = DEFAULT_CONFIG):
    """Calculate satellite attitude and related angles for given orbit results and target coordinates.
    (Based on RPR-SJ4B0509)
    Args:
        results: Satellite orbit results.
        skycoord_target: SkyCoord of the target.
    
    Returns:
        Tuple containing times, satellite positions, various angles, and direction vectors.
    """
    # 計算点における各種の角度
    n_results = len(results)
    SatZSun, SatXSun, SatYSun, SatX, SatY, SatZ, toSatZn, toSatAz = init_angles(n_results)

    times, Sat, toSun, toTgt, toSat, SatTgt, SunTgt, SatprojTgt, BarytoSat, pos_gcrs = compute_vectors_angles_arr(results, skycoord_target)

    for i, result in enumerate(results):
        Z = toSun[i]
        X = normalize(toTgt[i] - np.dot(toTgt[i], Z) * Z)

        if SatTgt[i] <= config.OBSERVATION_ANGLE_MAX_DEG:
            SatZ[i] = toTgt[i]
        else:
            if np.abs(np.dot(toTgt[i], X)) < np.finfo(float).eps:
                ZZ = toSun[i]
            else:
                if np.dot(toTgt[i], Z) < 0:
                    ZZ = normalize(-toTgt[i] + X)
                else:
                    ZZ = normalize(toTgt[i] - X)
            XX = normalize(toTgt[i] - np.dot(toTgt[i], ZZ) * ZZ)
            YY = np.cross(ZZ, XX)
            theta = np.arctan2(np.dot(YY, toSat[i]), np.dot(XX, toSat[i]))
            if theta < 0 and theta > -np.pi / 2:
                theta = -np.pi / 2
            if theta > 0 and theta < np.pi / 2:
                theta = np.pi / 2
            rot_quat = quaternion.from_rotation_vector(2 * (theta - np.pi / 2) * ZZ)
            SatZ[i] = quaternion.as_rotation_matrix(rot_quat).dot(toTgt[i])
        SatY[i] = -normalize(np.cross(toSun[i], SatZ[i]))
        SatX[i] = np.cross(SatY[i], SatZ[i])
        SatZSun[i] = angle_between(SatZ[i], toSun[i])
        SatXSun[i] = angle_between(SatX[i], toSun[i])
        SatYSun[i] = angle_between(SatY[i], toSun[i])
        toSatZn[i] = angle_between(toSat[i], SatZ[i])
        tx = toSat[i] @ SatX[i]    # toSat の SatX 方向成分
        ty = toSat[i] @ SatY[i]    # toSat の SatY 方向成分

        theta_rad = np.arctan2(ty, tx)
        theta_deg = np.degrees(theta_rad)
        toSatAz[i] = theta_deg

    return times, Sat, toSun, toTgt, toSat, SatTgt, SunTgt, SatX, SatY, SatZ,\
        SatZSun, SatXSun, SatYSun, SatprojTgt, toSatZn, toSatAz, BarytoSat

def orbattitude(results, skycoord_target, config: Config = DEFAULT_CONFIG):
    """Calculate satellite attitude and related angles for given orbit results and target coordinates.
    (Based on RPR-SJ512017B)

    Args:
        results: Satellite orbit results.
        skycoord_target: SkyCoord of the target.
    
    Returns:
        Tuple containing times, satellite positions, various angles, and direction vectors.
    """
    # 計算点における各種の角度
    n_results = len(results)
    SatZSun, SatXSun, SatYSun, SatX, SatY, SatZ, toSatZn, toSatAz = init_angles(n_results)

    times, Sat, toSun, toTgt, toSat, SatTgt, SunTgt, SatprojTgt, BarytoSat, pos_gcrs = compute_vectors_angles_arr(results, skycoord_target)

    is_obs = SatTgt <= config.OBSERVATION_ANGLE_MAX_DEG
    orbit_cycles = find_orbit_cycles_from_SatTgt(SatTgt, config.OBSERVATION_ANGLE_MAX_DEG)

    for (s_obs, e_obs, s_next) in orbit_cycles:
        # この周回のインデックス範囲（観測＋非観測）
        idx_seg = np.arange(s_obs, s_next + 1)

        # 観測インデックス / 非観測インデックス
        idx_obs_seg = np.arange(s_obs, e_obs + 1)
        idx_nonobs_seg = np.arange(e_obs + 1, s_next + 1)
        if len(idx_nonobs_seg) == 0:
            # ほぼ全周観測になってしまった場合のフォールバック
            idx_nonobs_seg = np.array([e_obs])

        # 観測中心 / 最終観測 / 非観測中心
        i_c = idx_obs_seg[len(idx_obs_seg)//2]   # 観測中心
        i_e = e_obs                              # 最終観測
        i_n = idx_nonobs_seg[len(idx_nonobs_seg)//2]  # 非観測中心

        # ---- main() の SatXc, SatYc, SatZc, SatZn, tiltAngle, rotdir を計算 ----
        Satc = pos_gcrs[i_c]
        Satn = pos_gcrs[i_n]

        toObj_c = toTgt[i_c]
        toSun_c = toSun[i_c]

        SatZc = toObj_c
        SatYc = normalize(np.cross(SatZc, toSun_c))
        SatXc = np.cross(SatYc, SatZc)

        SatZn = normalize(Satn)
        SatYn = normalize(np.cross(SatZn, toSun_c))
        SatXn = np.cross(SatYn, SatZn)

        rotdir = np.dot(toSun_c, toObj_c)

        qrot_pi = quaternion.from_rotation_vector(toSun_c * (-np.pi))
        Zno = normalize(quaternion.as_rotation_matrix(qrot_pi).dot(SatZc))
        tiltAngle = -np.arccos(np.clip(np.dot(Zno, SatZn), -1.0, 1.0))

        # ---- この周回の各時刻に対して姿勢を決定 ----
        for i in idx_seg:
            if is_obs[i]:
                # ===== 観測中：ターゲット指向 =====
                # （既存 orbitattitude の観測側ロジックをそのまま使ってよい）
                Z = toSun[i]
                X_tmp = toTgt[i] - np.dot(toTgt[i], Z) * Z
                X = normalize(X_tmp)

                if SatTgt[i] <= config.OBSERVATION_ANGLE_MAX_DEG:
                    SatZ[i] = toTgt[i]
                else:
                    # ここは観測中だけど SatTgt>lim というレアケースなので、
                    # 必要に応じて処理を書く。単純には SatZ=toTgt で良い。
                    SatZ[i] = toTgt[i]

                SatY[i] = -normalize(np.cross(toSun[i], SatZ[i]))
                SatX[i] = np.cross(SatY[i], SatZ[i])

            else:
                # ===== 非観測中：main() と同じ「太陽軸回転 + tilt」 =====

                s = (i - i_e) / max(s_next - i_e, 1)
                s = np.clip(s, 0.0, 1.0)

                # 太陽軸回転量 (0 → -π rad)
                phi_i = -2*np.pi * s
                qrot_sun = quaternion.from_rotation_vector(toSun_c * phi_i)
                R_sun = quaternion.as_rotation_matrix(qrot_sun)

                X0 = normalize(R_sun.dot(SatXc))
                Y0 = normalize(R_sun.dot(SatYc))
                Z0 = normalize(R_sun.dot(SatZc))

                # tilt 量 psi
                psi = tiltAngle * s
                if rotdir < 0:
                    psi = -psi

                qrot_tilt = quaternion.from_rotation_vector(Y0 * psi)
                R_tilt = quaternion.as_rotation_matrix(qrot_tilt)

                SatX[i] = normalize(R_tilt.dot(X0))
                SatZ[i] = normalize(R_tilt.dot(Z0))
                SatY[i] = Y0

            SatZSun[i] = angle_between(SatZ[i], toSun[i])
            SatXSun[i] = angle_between(SatX[i], toSun[i])
            SatYSun[i] = angle_between(SatY[i], toSun[i])
            toSatZn[i] = angle_between(toSat[i], SatZ[i])
            tx = toSat[i] @ SatX[i]    # toSat の SatX 方向成分
            ty = toSat[i] @ SatY[i]    # toSat の SatY 方向成分

            theta_rad = np.arctan2(ty, tx)
            theta_deg = np.degrees(theta_rad)
            toSatAz[i] = theta_deg

    return times, Sat, toSun, toTgt, toSat, SatTgt, SunTgt, SatX, SatY, SatZ,\
        SatZSun, SatXSun, SatYSun, SatprojTgt, toSatZn, toSatAz, BarytoSat

def find_orbit_cycles_from_SatTgt(SatTgt, obs_max_deg):
    """
    Find orbit cycles based on satellite-target angles.
    An orbit cycle is defined as the interval from one observation start to the next observation start,
    including the observation and non-observation periods in between.
    
    Args:
        SatTgt: Numpy array of satellite-target angles in degrees.
        obs_max_deg: Maximum angle in degrees for observation.

    Returns:
        List of tuples (start_obs_index, end_obs_index, next_start_obs_index) for each orbit cycle.
    """
    SatTgt = np.asarray(SatTgt)
    is_obs = SatTgt <= obs_max_deg
    n = len(SatTgt)

    start_obs_list = []
    end_obs_list = []

    for i in range(1, n):
        # 非観測→観測 に入った瞬間 = 観測開始
        if not is_obs[i-1] and is_obs[i]:
            start_obs_list.append(i)
        # 観測→非観測 に出た瞬間 = 観測終了
        if is_obs[i-1] and not is_obs[i]:
            end_obs_list.append(i-1)

    # 端の処理（データの最初/最後が観測中で始まる/終わる場合）
    if is_obs[0]:
        # 先頭がすでに観測中なら 0 を開始点として追加
        start_obs_list.insert(0, 0)
    if is_obs[-1]:
        # 最後まで観測中なら、最後の点を終了点として追加
        end_obs_list.append(n-1)

    # start/end の数が合っていない場合は、短い方に揃える
    m = min(len(start_obs_list), len(end_obs_list))
    start_obs_list = start_obs_list[:m]
    end_obs_list   = end_obs_list[:m]

    orbit_cycles = []
    # 連続する start_obs 同士の区間を 1 周回とみなす
    for k in range(m - 1):
        s_obs = start_obs_list[k]
        e_obs = end_obs_list[k]
        s_next = start_obs_list[k+1]
        if s_obs < e_obs < s_next:
            orbit_cycles.append((s_obs, e_obs, s_next))

    # 最後の周回は「次の観測開始」が無いので、必要なら末尾までを 1 周とみなす
    # （ここではお好みで。例として末尾までを 1周として追加）
    if m >= 1:
        s_obs = start_obs_list[-1]
        e_obs = end_obs_list[-1]
        if e_obs > s_obs:
            orbit_cycles.append((s_obs, e_obs, n-1))

    return orbit_cycles


def detect_ascending_nodes(results):
    """Detect ascending node times from satellite orbit results.
    Args:
        results: Satellite orbit results.

    Returns:
        List of times when ascending nodes occur.
    """
    ascending_passes = []

    for i in range(1, len(results)):
        t0, r0, v0, loc0 = results[i-1]
        t1, r1, v1, loc1 = results[i]

        lat0 = loc0.lat.deg
        lat1 = loc1.lat.deg

        # 赤道通過（南→北）＝緯度が負→正、かつ z方向速度 > 0
        if lat0 < 0 and lat1 >= 0:
            vz = v1[2]  # z方向速度（km/s）
            if vz > 0:
                ascending_passes.append(t1)
    return ascending_passes