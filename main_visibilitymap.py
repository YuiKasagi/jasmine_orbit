#!/usr/bin/env python3

"""
Satellite orbit and attitude, original: H. Kataza, edit by Y. Kasagi and Codex CLI (2025)

    usage:
        main_target.py [-h|--help] (-s|-a) -p <day_offset> -w <days> [-o] [-t <target_name>] [-m <minutes>] [-c <coordinate>]

    options:
        -h --help       show this help message and exit
        -s              春分点を基準
        -a              秋分点を基準
        -p <day_offset> 基準日からの計算開始日(この日を含む)
        -w <days>       計算期間(日)
        -o              グラフ出力(True or False)
        -m <minutes>   time step in minutes [default: 1]
        -c <coordinate>   座標系 'C' (赤道座標) or 'E' (黄道座標) [default: C]
"""

from docopt import docopt
import time
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord, GeocentricTrueEcliptic
import astropy.units as u
import healpy as hp
from jasmine_orbit.OrbitAttitude import (
    prepare_orbit, 
    horizon_angle,
    detect_ascending_nodes, 
    orbattitude,
    thermal_input_per_orbit,
    compute_fraction_between_nodes
)
from jasmine_orbit.GraphOrbit import map_visibility, get_true_segments

import seaborn as sns
sns.set_context('talk')

# load config
from config.settings_example import CONFIG

from joblib import Parallel, delayed
def _one_pix(results, skycoord, times_array, alpha,
             sun_min, sun_max, obs_lim, az_lim, zn_min, th_thermal_input, dt):

    times, Sat, toSun, toTgt, toSat, SatTgt, SunTgt, SatX, SatY, SatZ, \
        SatZSun, SatXSun, SatYSun, SatprojTgt, toSatZn, toSatAz, BarytoSat = \
        orbattitude(results, skycoord, config=CONFIG)
    
    mask_tgt = (SatTgt <= obs_lim) 
    obs_start_idx = get_true_segments(mask_tgt)
    time_obs_start = [times[idx[0]] for idx in obs_start_idx]
    index_obs_start = np.where(np.isin(times_array, np.asarray(time_obs_start)))[0]

    mask_obs = (SatTgt <= obs_lim) & (sun_min <= SatZSun) & (SatZSun <= sun_max) #True -> observable
    obs_idx = np.flatnonzero(mask_obs)
    frac_obs = compute_fraction_between_nodes(index_obs_start, obs_idx)
    frac_obs = np.array(frac_obs)

    mask_thermal = (np.abs(toSatAz) <= az_lim) & (toSatZn >= zn_min) #True -> Earth's IR light enters into the radiator
    thermal_idx = np.flatnonzero(mask_thermal)

    thermal_input = np.cos(np.deg2rad(toSatAz)) * np.cos(np.pi - (alpha + np.deg2rad(toSatZn)))
    sum_thermal_input = thermal_input_per_orbit(index_obs_start, thermal_idx, thermal_input, dt)

    visible = (frac_obs > 0) & (np.asarray(sum_thermal_input) < th_thermal_input)
    return int(np.sum(visible))

def main_visibility_map(args, nside=8, th_thermal_input=7.):
    """Main function to plot the number of visible orbits.
    Args:
        args: Command-line arguments.
        nside: Resolution of healpix
        th_thermal_input: threshold for sum of the thrmal input
    """
    dt = float(args['-m'])
    coord_use = args['-c']

    # Get orbit data
    results, inclination_deg, (tle1, tle2), start_date, days_calc, altitude = prepare_orbit(args, config=CONFIG)

    # パラメータ設定
    npix = hp.nside2npix(nside)

    m = np.zeros(npix)  # 初期化

    # ピクセルセンターの角度 (θ, φ) を取得 (θ は天頂からの角度、φ は方位角)
    theta, phi = hp.pix2ang(nside, np.arange(npix))

    # HEALPix(theta,phi) -> ICRS (RA,Dec)
    if coord_use == 'C':
        ra = phi * u.rad
        dec = (0.5 * np.pi - theta) * u.rad
        skycoord_targets = SkyCoord(ra=ra, dec=dec, frame="icrs")
    elif coord_use == 'E':
        lon = phi * u.rad                          # 黄経 λ
        lat = (0.5*np.pi - theta) * u.rad          # 黄緯 β
        skycoord_targets = SkyCoord(lon=lon, lat=lat, frame=GeocentricTrueEcliptic())

    sun_min, sun_max = CONFIG.THERMAL_SUN_ANGLE_RANGE_DEG
    obs_lim = CONFIG.OBSERVATION_ANGLE_MAX_DEG
    az_lim  = CONFIG.THERMAL_Az_MAX_DEG
    zn_min  = CONFIG.THERMAL_Zn_MAX_DEG

    alpha = horizon_angle(config=CONFIG)

    # index_an を1回だけ確定（timesが共通前提）
    times0, *_ = orbattitude(results, skycoord_targets[0], config=CONFIG)
    times_array = np.asarray(times0)

    with tqdm_joblib(tqdm(total=len(skycoord_targets))) as progress_bar:
        m = np.asarray(
        Parallel(n_jobs=-2, backend="loky", batch_size="auto")(
            delayed(_one_pix)(results, skycoord, times_array, alpha,
                            sun_min, sun_max, obs_lim, az_lim, zn_min, th_thermal_input, dt)
            for skycoord in skycoord_targets
        ),
        dtype=np.int32
        )

    if args['-o']:
        outfile_map = f"{CONFIG.OUTPUT_DIR}/figs/map/{(times_array[-1] - times_array[0]).days + 1}days_from{times_array[0].strftime('%Y-%m-%d')}_visibilitymap_{coord_use}.png"
        outfile_map_data = f"{CONFIG.OUTPUT_DIR}/data/map/{(times_array[-1] - times_array[0]).days + 1}days_from{times_array[0].strftime('%Y-%m-%d')}_visibilitymap_{coord_use}.fits"
        hp.write_map(outfile_map_data, m, nest=False, overwrite=True)
    else:
        outfile_map = None

    # 系外惑星ターゲット
    df_target = pd.read_csv(CONFIG.TARGET_CATALOG_PATH)
    target_plot = ["LTT 1445 A", "GJ 357", "TRAPPIST-1", "GJ 486", "GJ 3929"]
    df_target_plot = df_target[df_target["name"].isin(target_plot)]

    # map
    map_visibility(m, times_array, coord_map=coord_use, coord_plot=coord_use, df_target=df_target_plot, outfile=outfile_map)

import contextlib
from tqdm import tqdm
import joblib
# https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution/58936697#58936697
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

if __name__ == '__main__':
    start = time.time()

    args = docopt(__doc__)
    main_visibility_map(args)

    end = time.time()
    print("Elapsed time: {:.2f} seconds".format(end - start))