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

import sys
sys.path.insert(0, "/Users/yuikasagi/Jasmine/analysis/jasmine_orbit/src")

from docopt import docopt
import time
import numpy as np
from astropy.coordinates import SkyCoord, Galactic, CartesianRepresentation, ICRS
import astropy.units as u
from jasmine_orbit.OrbitAttitude import (
    prepare_orbit, 
    horizon_angle,
    detect_ascending_nodes, 
    load_target_coordinates, 
    orbattitude,
    compute_fraction_between_nodes,
    compute_thermal_fraction_per_orbit,
    thermal_input_per_orbit
)
from jasmine_orbit.GraphOrbit import plot_orbit_3d, plot_frac_themalfeasibility

import seaborn as sns
sns.set_context('talk')

# load config
from .config.settings_example import CONFIG

def main_target(args):
    """Main function to process satellite orbit and attitude data for a specific target.
    Args:
        args: Command-line arguments.
    """
    # Get orbit data
    results, inclination_deg, (tle1, tle2), start_date, days_calc, altitude = prepare_orbit(args, config=CONFIG)

    time_an = detect_ascending_nodes(results)

    # Set target
    target_name = args['-t']
    print(target_name)
    if target_name in ['GC']:
        skycoord_target = SkyCoord(l=0 * u.degree, b=0 * u.degree, frame=Galactic)
    else:
        try:
            skycoord_target, target_data = load_target_coordinates(target_name, config=CONFIG)
        except ValueError as exc:
            print(exc)
            return
        print(target_data)

    times, Sat, toSun, toTgt, toSat, SatTgt, SunTgt, SatX, SatY, SatZ,\
        SatZSun, SatXSun, SatYSun, SatprojTgt, toSatZn, toSatAz, BarytoSat = orbattitude(results, skycoord_target, config=CONFIG)

    # Jadge observation and thermal feasibility
    sun_min, sun_max = CONFIG.THERMAL_SUN_ANGLE_RANGE_DEG
    mask_obs = (SatTgt <= CONFIG.OBSERVATION_ANGLE_MAX_DEG) & (sun_min <= SatZSun) & (SatZSun <= sun_max)
    index_obs = np.where(mask_obs)[0]

    mask_thermal = (np.abs(toSatAz) <= CONFIG.THERMAL_Az_MAX_DEG) & (toSatZn >= CONFIG.THERMAL_Zn_MAX_DEG) 
    thermal_indices = np.where(mask_thermal)[0]

    # define thermal input with depending on phi(Az), theta(Zn)
    alpha = horizon_angle(config=CONFIG)
    thermal_input = np.cos(np.deg2rad(toSatAz)) * np.cos(np.pi - (alpha+np.deg2rad(toSatZn)))

    times_array = np.asarray(times)
    index_an = np.where(np.isin(times_array, np.asarray(time_an)))[0]

    # Calculate fractions
    frac_obs = compute_fraction_between_nodes(index_an, index_obs)
    frac_obs_thermal = compute_thermal_fraction_per_orbit(index_an, thermal_indices)
    sum_thermal_input = thermal_input_per_orbit(index_an, thermal_indices, thermal_input)

    BarytoSat_ecliptic = SkyCoord(CartesianRepresentation(BarytoSat.T * u.AU), frame=ICRS()).transform_to('barycentricmeanecliptic').cartesian.xyz.T.value

    if args['-o']:
        outfile_angle_fig = f"{CONFIG.OUTPUT_DIR}/figs/{target_name}_{times[0].strftime('%Y-%m-%d')}_angles.png"  
        outfile_angle_data = f"{CONFIG.OUTPUT_DIR}/orbit/{target_name}_{times[0].strftime('%Y-%m-%d')}_angles.npz"  
        outfile_orbit3d_fig = f"{CONFIG.OUTPUT_DIR}/figs/{target_name}_{times[0].strftime('%Y-%m-%d')}_orbit3d.png"
        outfile_orbit3d_data = f"{CONFIG.OUTPUT_DIR}/orbit/{target_name}_{times[0].strftime('%Y-%m-%d')}_orbit3d.npz"
        outfile_frac_thermal = f"{CONFIG.OUTPUT_DIR}/figs/{target_name}_{times[0].strftime('%Y-%m-%d')}_frac_thermal.png"

        np.savez(outfile_angle_data, SatTgt=SatTgt, SatZSun=SatZSun, SatAz=toSatAz, times_array=times_array)
        np.savez(outfile_orbit3d_data, BarytoSat_ecliptic=BarytoSat_ecliptic, toTgt=toTgt, mask_obs=mask_obs, mask_thermal=mask_thermal)
    else:
        outfile_angle_fig = None
        outfile_orbit3d_fig = None
        outfile_frac_thermal = None
    
    #plot_angle_target(times, SatTgt, SatprojTgt, SatZSun, index_obs, index_an, frac_obs, target_name, outfile_angle_fig)
    #plot_orbit_3d(BarytoSat_ecliptic, toTgt, mask_obs, mask_thermal, outfile=outfile_orbit3d_fig)
    #plot_frac_themalfeasibility(times, index_an, frac_obs, frac_obs_thermal, target_name, outfile=outfile_frac_thermal)
    plot_frac_themalfeasibility(times, index_an, frac_obs, sum_thermal_input, target_name, outfile=outfile_frac_thermal)


if __name__ == '__main__':
    start = time.time()

    args = docopt(__doc__)
    main_target(args)

    end = time.time()
    print("Elapsed time: {:.2f} seconds".format(end - start))