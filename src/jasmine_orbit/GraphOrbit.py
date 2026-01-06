import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from jasmine_orbit.OrbitCalc import ltan_str
import matplotlib.dates as mdates
from matplotlib.legend_handler import HandlerTuple
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import healpy as hp
from astropy import units as u
from astropy.coordinates import SkyCoord

from cartopy.feature.nightshade import Nightshade # failed to use with cfeature (?)

def get_true_segments(mask: np.ndarray):
    """
    真(True)が連続しているインデックス区間の
    (start_idx, end_idx) のリストを返す。
    """
    mask = np.asarray(mask, dtype=bool)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []

    # 差分が1より大きいところで区切る
    split_points = np.where(np.diff(idx) > 1)[0] + 1
    groups = np.split(idx, split_points)

    return [(g[0], g[-1]) for g in groups]

def plot_angle(times, SatGc, SunGc, SatZSun, SatXSun, SatYSun,
               altitude, inclination_deg, start_date, days_calc,
               tle1, tle2, time1, time2, time3, meanSunGc, outfile):
    #  ,times2, SatZtoGc,SatYtoSun,SatZtoEarth,thetaYL,thetaL):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(times, SatGc, label='Satellite - GC')
    ax.plot(times, SunGc, label='Sun - GC')
    ax.plot(times, SatZSun, label='Satellite-Z - Sun')
    ax.plot(times, SatYSun, label='Satellite-Y - Sun')
    ax.plot(times, SatXSun, label='Satellite-X - Sun')

    # 横軸のフォーマットを指定（mm/dd HH:MM:SS.S）
    ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 15, 30, 45]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    fig.autofmt_xdate()

    # グラフのラベルなど
    ax.set_xlabel('Time (mm/dd HH:MM)')
    ax.set_ylabel('Angle')
    ax.set_title('Time vs Angle')
    ax.set_ylim(0, 180)
    ax.legend()
    plt.subplots_adjust(bottom=0.27)
    fig.text(0.1, 0.1, 'Orbit start {} obs-end {} next-orbit {} mean Sun-Gc angle {:.2f}'.
             format(time1, time2, time3, meanSunGc))
    fig.text(0.1, 0.07,
             "Altitude: {} km,  Inclination: {:.2f} degree,  Local time of ascending node {},  "
             .format(altitude, inclination_deg, ltan_str) +
             "Start date {},  Duration {} day"
             .format(start_date.strftime("%Y/%m/%d  %H:%M:%S"), days_calc))
    fig.text(0.2, 0.04, "TLE:")
    fig.text(0.24, 0.04, tle1)
    fig.text(0.24, 0.01, tle2)

    # グラフを表示
    if outfile:
        plt.savefig(outfile)
    plt.show()


def plot_oribt(results, altitude, inclination_deg, start_date, days_calc,
               tle1, tle2,  method, outfile):
    # 結果の表示
    # for result in results:
    #     print(f"日付: {result[0]}, 位置 (x, y, z): {result[1]}, "
    #           f"速度 (vx, vy, vz): {result[2]}, "
    #           f"緯度 {result[3].lat.deg} 経度 {result[3].lon.deg}")
    lat = np.array([result[3].lat.deg for result in results])
    lon = np.array([result[3].lon.deg for result in results])
    # 投影する
    fig = plt.figure(figsize=(12, 6))
    if method == 'Mercator':  # メルカトル図法
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mercator(central_longitude=30))
    elif method == 'Mollweide':  # モルワイデ図法
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mollweide(central_longitude=30))
    # タイトル等
    ax.set_title('Dawn Dusk Orbit')
    fig.text(0.1, 0.07,
             "Altitude: {} km,  Inclination: {:.2f} degree,  Local time of ascending node {},  "
             .format(altitude, inclination_deg, ltan_str) +
             "Start date {},  Duration {} day"
             .format(start_date.strftime("%Y/%m/%d  %H:%M:%S"), days_calc))
    fig.text(0.2, 0.04, "TLE:")
    fig.text(0.24, 0.04, tle1)
    fig.text(0.24, 0.01, tle2)

    # 地球全面を表示
    ax.set_global()

    # 大陸の形を描画
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    # 点をプロットし、それらを線で結ぶ
    # ax.plot(lon, lat, color='blue', marker='o', transform=ccrs.PlateCarree())
    ax.plot(lon, lat, linestyle='None', marker='.', transform=ccrs.PlateCarree())

    # 
    #julian_date = Time(start_date, format='isot').jd # days
    #obliquity = calc_obliquity(julian_date) # deg

    #beta_angle = np.pi - (obliquity + inclination_deg)


    # 大陸の塗りつぶし
    # ax.add_feature(cfeature.LAND, facecolor='lightgray')

    # グリッド線を描画（経緯線）
    # ax.gridlines(draw_labels=True)

    # 夜に影をつける
    ax.add_feature(Nightshade(start_date, alpha=0.2))

    # グラフを表示
    if outfile:
        plt.savefig(outfile)
    plt.show()

def plot_angle_target(times, SatTgt, SatprojTgt, SatZSun, index_obs, index_an, frac_obs, target_name, outfile):
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(times, SatTgt, '.', color="grey", label="angle between Sat-GEOC-Tgt [deg]")
    ax.plot(np.array(times)[index_obs], SatTgt[index_obs], '.', color="tab:blue")#, label="observable")

    #ax.plot(times, SatprojTgt, 'x', color="grey", label="angle between Sat-GEOC-projTgt [deg]")
    #ax.plot(np.array(times)[index_obs], SatprojTgt[index_obs], 'x', color="tab:blue")#, label="observable")

    if len(index_obs) == 0:
        index_obs_tmp = np.where(SatTgt <= 90)[0]
        ax.plot(np.array(times)[index_obs_tmp], SatZSun[index_obs_tmp], '+', color="grey", label="angle between SatZ-GEOC-Sun [deg]")
    else:
        ax.plot(np.array(times)[index_obs], SatZSun[index_obs], '+', color="tab:blue", label="angle between SatZ-GEOC-Sun [deg]")

    ax.vlines(np.array(times)[index_an], 0, 180, color='k', ls="dashed", lw=1)

    ax.legend()
    ax.set_title("%s, visibile fraction = %.2f per orb."%(target_name, frac_obs))
    ax.set(xlabel='Time (mm/dd HH:MM)', ylabel='Angle')
    ax.set(ylim=(0,180))
    # 横軸のフォーマットを指定（mm/dd HH:MM:SS.S）
    ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 15, 30, 45]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    fig.autofmt_xdate()

    if outfile:
        plt.savefig(outfile, bbox_inches="tight")
    plt.show()

def plot_orbit_3d(pos_sat, pos_tar, mask_obs, mask_thermal, outfile, view_angle=[40, 90, 0]):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection='3d')

    ax.scatter(
    pos_sat[~mask_obs, 0], pos_sat[~mask_obs, 1], pos_sat[~mask_obs, 2], s=1, marker='.', color='gray', alpha=0.3)
    ax.scatter(
    pos_sat[mask_obs, 0], pos_sat[mask_obs, 1], pos_sat[mask_obs, 2], s=3, marker='.', alpha=0.5)
    ax.scatter(
    pos_sat[mask_thermal, 0], pos_sat[mask_thermal, 1], pos_sat[mask_thermal, 2], s=5, marker='x', color='r', alpha=0.5)

    st = ax.scatter(pos_sat[0, 0], pos_sat[0, 1], pos_sat[0, 2], color='r', s=30)
    ed = ax.scatter(pos_sat[-1, 0], pos_sat[-1, 1], pos_sat[-1, 2], color='blue', s=30)

    #ax2 = fig.add_subplot(projection='3d')  # quiver 用（見た目の z）
    #ax2.set_position(ax.get_position())
    #hide_axis3d(ax2)  # ← w_xaxis を使わない
    #ax2.quiver([0], [0], [0], pos_tar[:,0], pos_tar[:,1], pos_tar[:,2], normalize=True, length=0.5)


    ax.set_xlabel('Ecliptic X (au)', fontsize=14)
    ax.set_ylabel('Ecliptic Y (au)', fontsize=14)

    hg = ax.scatter([], [], s=250, marker='.', color='gray')
    hb = ax.scatter([], [], s=250, marker='.', color='C0')
    hx = ax.scatter([], [], s=150, marker='x', color='r')

    ax.legend(
    [hb, hx, st, ed], ['observable', 'Earth IR', 'start', 'end'], fontsize=14,
    handler_map={tuple: HandlerTuple(ndivide=None, pad=0.5)})

    elev, azim, roll = view_angle
    ax.view_init(elev=elev, azim=azim, roll=roll)
    #ax2.view_init(elev=elev, azim=azim, roll=roll)

    #ax.set(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1))
    #ax2.set(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1), zlim=(-1.1, 1.1))

    fig.tight_layout()
    if outfile:
        fig.savefig(outfile)
    plt.show()

def hide_axis3d(ax):
    """3D Axes の枠・面・目盛・グリッドを消してデータだけ見せる（公開APIのみ）"""
    # 背景を透明に（重ねても下の軸が見える）
    ax.patch.set_alpha(0.0)

    # 軸パネル（pane）を非表示
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            a.pane.set_visible(False)   # 3.6+
        except Exception:
            pass

    # 目盛/ラベルを消す
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")

    # グリッドを消す
    ax.grid(False)

    # 枠線（spine 相当）は 3D にはないので、box を消すなら axis_off
    # ただしデータも見えるようにする
    ax.set_axis_off()   # これが一番手っ取り早い（データは表示されます）

def plot_frac_themalfeasibility(times, index_an, frac_obs, frac_obs_thermal, target_name, outfile, th_thermal_input=7.):
    times_an = np.array(times)[index_an[:-1]]
    orbit_num = range(len(times_an))
    time_span = (times[-1] - times[0]).days

    mask_obs_orb = np.array(frac_obs)>0

    fig, ax1 = plt.subplots(figsize=(10,5))
    ax1.plot(orbit_num, frac_obs_thermal, '-')
    ax1.axhline(th_thermal_input, orbit_num[0], orbit_num[-1], ls="dashed", lw=1, color="tab:red", zorder=0)

    for start, end in get_true_segments(~mask_obs_orb):
        ax1.axvspan(start, end, color="grey", alpha=0.2)
    
    if time_span == 89.: # 1シーズン分の図を作るとき、始め・中・終わりに星印
        idx = np.where(mask_obs_orb)[0]
        if len(idx) > 0:
            first = idx[0]
            last  = idx[-1]
            middle = idx[len(idx) // 2]
        else:
            first = middle = last = None 
        for idx_tmp in [first, middle, last]:
            frac_tmp = frac_obs_thermal[idx_tmp]
            if frac_tmp>th_thermal_input:
                plot_args = {"marker": "*", "mfc": "yellow", "mec": "tab:blue", "ms": 20}
            else:
                plot_args = {"marker": "*", "mfc": "tab:red", "mec": "red", "ms": 20}
            ax1.plot(orbit_num[idx_tmp], frac_tmp, **plot_args)

    ax1.set_title(f"{target_name}, {times[0].strftime('%Y-%m-%d')} → {times[-1].strftime('%Y-%m-%d')} ({(times[-1] - times[0]).days + 1} days)")
    ax1.set_xlabel("Orbit number from {}".format(times[0].strftime("%Y-%m-%d")))
    #ax1.set_ylabel("Fraction of thermal unfeasible time\nper orbit")
    ax1.set_ylabel("Thermal input during a single orbit")
    ax1.set(xlim=(orbit_num[0], orbit_num[-1]))
    #ax1.set(ylim=(0,1.))

    ax2 = ax1.twiny()
    ax2.plot(times_an, frac_obs_thermal, '-', alpha=0.)

    ax2.set(xlim=(times_an[0], times_an[-1]))
    ax2.set(xlabel=f"Date ({str(times[0].tzinfo)})")
    plt.setp(ax2.get_xticklabels(), rotation=30, ha="left")
    if outfile:
        plt.savefig(outfile, bbox_inches="tight")
    plt.show()

def plot_visibility_mollweide(lon_arr, lat_arr, frac_obs_map, outfile):
    lon_rad = np.radians(lon_arr)
    lat_rad = np.radians(lat_arr)
    lon_grid, lat_grid = np.meshgrid(lon_rad, lat_rad)

    masked_map = np.ma.masked_where(frac_obs_map == 0, frac_obs_map)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='mollweide')
    im = ax.pcolormesh(
        lon_grid,
        lat_grid,
        masked_map,
        shading='auto',
        cmap='viridis',
        vmin=0.62,
        vmax=0.69,
    )
    im.cmap.set_bad('lightgray')

    cb = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05)
    cb.set_label('frac_obs')

    ax.grid(True)
    ax.set_title("Fraction of Observation in Ecliptic Coordinates (Mollweide)")

    x_labels = ["210°", "240°", "270°", "300°", "330°", "0°", "30°", "60°", "90°", "120°", "150°"]
    x_ticks = np.radians(np.arange(30, 360, 30)) - np.pi
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, color='grey')
    if outfile:
        plt.savefig(outfile, bbox_inches="tight")
    plt.show()

def map_visibility(m, times, coord_map='C', coord_plot='C', df_target=None, outfile=None):
    plt.figure(figsize=(15,10))
    hp.mollview(m, cmap=cm.magma, 
                title=f"{times[0].strftime('%Y-%m-%d')} → {times[-1].strftime('%Y-%m-%d')} ({(times[-1] - times[0]).days + 1} days)", 
                coord=[coord_map, coord_plot], notext=True, hold=True)#, rot=(180, 0, 0))
    hp.graticule()

    # 系外惑星ターゲット
    if df_target is not None:
        ra_target = df_target["ra"]
        dec_target = df_target["dec"]
        c_target = SkyCoord(ra=ra_target.values*u.degree, dec=dec_target.values*u.degree, frame='icrs')
        if coord_plot == 'C':
            theta_target = np.pi/2 - c_target.dec.rad  # Decからthetaに変換（赤緯を天頂角に）
            phi_target = c_target.ra.rad  # RAはそのままphiとして使う
        elif coord_plot == 'E':
            c_target_ecl = c_target.transform_to('geocentrictrueecliptic')
            theta_target = np.pi/2 - c_target_ecl.lat.rad  # 黄緯からthetaに変換（黄緯を天頂角に）
            phi_target = c_target_ecl.lon.rad  # 黄経はそのままphiとして使う

        norm = Normalize(vmin=0, vmax=40)

        hp.projscatter(theta_target, phi_target, lonlat=False, c=df_target["JASMINE S/N"], cmap=cm.seismic, norm=norm, s=50, edgecolor='k',coord=[coord_plot])#, rot=(0,0,180))

    # 銀河中心の位置 (銀河中心の位置は銀河座標で l = 0, b = 0)
    l_gal_center = 0.0
    b_gal_center = 0.0

    hp.projscatter(l_gal_center, b_gal_center, lonlat=True, coord=['G', coord_plot], s=100, c='red', marker='*', label='Galactic Center')
    hp.projscatter(l_gal_center+180, -b_gal_center, lonlat=True, coord=['G', coord_plot], s=100, c='blue', marker='*')

    plt.legend(loc="lower left")

    plt.gca().invert_xaxis()
    if outfile:
        plt.savefig(outfile, bbox_inches="tight")
    plt.show()