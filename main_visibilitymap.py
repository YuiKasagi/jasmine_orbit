#!/usr/bin/env python3

"""
Satellite orbit and attitude, original: H. Kataza, edit by Y. Kasagi and Codex CLI (2025)

    usage:
        main_target.py [-h|--help] (-s|-a) -p <day_offset> -w <days> [-o] [-t <target_name>] [-m <minutes>]

    options:
        -h --help       show this help message and exit
        -s              春分点を基準
        -a              秋分点を基準
        -p <day_offset> 基準日からの計算開始日(この日を含む)
        -w <days>       計算期間(日)
        -o              グラフ出力(True or False)
        -t <target_name>    target name
        -m <minutes>   time step in minutes [default: 1]
"""

from docopt import docopt
import time
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
import healpy as hp
from jasmine_orbit.OrbitAttitude import (
    prepare_orbit, 
    horizon_angle,
    detect_ascending_nodes, 
    orbattitude,
    thermal_input_per_orbit
)
from jasmine_orbit.GraphOrbit import map_visibility

import seaborn as sns
sns.set_context('talk')

# load config
from config.settings_example import CONFIG

from joblib import Parallel, delayed
def _one_pix(results, skycoord, index_an, alpha,
             sun_min, sun_max, obs_lim, az_lim, zn_min, th_thermal_input, dt):

    times, Sat, toSun, toTgt, toSat, SatTgt, SunTgt, SatX, SatY, SatZ, \
        SatZSun, SatXSun, SatYSun, SatprojTgt, toSatZn, toSatAz, BarytoSat = \
        orbattitude(results, skycoord, config=CONFIG)

    mask_obs = (SatTgt <= obs_lim) & (sun_min <= SatZSun) & (SatZSun <= sun_max)
    mask_thermal = (np.abs(toSatAz) <= az_lim) & (toSatZn >= zn_min)
    mask_visible = mask_obs & mask_thermal
    visible_idx = np.flatnonzero(mask_visible)

    thermal_input = np.cos(np.deg2rad(toSatAz)) * np.cos(np.pi - (alpha + np.deg2rad(toSatZn)))
    sum_thermal_input = thermal_input_per_orbit(index_an, visible_idx, thermal_input, dt)

    visible = (np.asarray(sum_thermal_input) < th_thermal_input)
    return int(np.sum(visible))

def main_target(args, nside=8, th_thermal_input=8.):
    """Main function to plot the number of visible orbits.
    Args:
        args: Command-line arguments.
        nside: Resolution of healpix
        th_thermal_input: threshold for sum of the thrmal input
    """
    dt = float(args['-m'])

    # Get orbit data
    results, inclination_deg, (tle1, tle2), start_date, days_calc, altitude = prepare_orbit(args, config=CONFIG)

    time_an = detect_ascending_nodes(results)
    time_an_arr = np.asarray(time_an)

    # パラメータ設定
    npix = hp.nside2npix(nside)

    m = np.zeros(npix)  # 初期化

    # ピクセルセンターの角度 (θ, φ) を取得 (θ は天頂からの角度、φ は方位角)
    theta, phi = hp.pix2ang(nside, np.arange(npix))

    # HEALPix(theta,phi) -> ICRS (RA,Dec)
    ra = phi * u.rad
    dec = (0.5 * np.pi - theta) * u.rad
    skycoord_targets = SkyCoord(ra=ra, dec=dec, frame="icrs")

    sun_min, sun_max = CONFIG.THERMAL_SUN_ANGLE_RANGE_DEG
    obs_lim = CONFIG.OBSERVATION_ANGLE_MAX_DEG
    az_lim  = CONFIG.THERMAL_Az_MAX_DEG
    zn_min  = CONFIG.THERMAL_Zn_MAX_DEG

    alpha = horizon_angle(config=CONFIG)

    # index_an を1回だけ確定（timesが共通前提）
    times0, *_ = orbattitude(results, skycoord_targets[0], config=CONFIG)
    times_array = np.asarray(times0)
    index_an = np.where(np.isin(times_array, time_an_arr))[0]

    m = np.asarray(
    Parallel(n_jobs=-2, backend="loky", batch_size="auto")(
        delayed(_one_pix)(results, skycoord, index_an, alpha,
                          sun_min, sun_max, obs_lim, az_lim, zn_min, th_thermal_input, dt)
        for skycoord in skycoord_targets
    ),
    dtype=np.int32
    )

    # map
    map_visibility(m, times_array, outfile=None)

if __name__ == '__main__':
    start = time.time()

    args = docopt(__doc__)
    main_target(args)

    end = time.time()
    print("Elapsed time: {:.2f} seconds".format(end - start))