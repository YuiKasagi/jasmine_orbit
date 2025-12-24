#from astropy.utils.iers import conf
#conf.auto_max_age = None

import warnings
from erfa import ErfaWarning
from astropy.utils.exceptions import AstropyWarning

# ErfaWarning と AstropyWarning を非表示にする
warnings.filterwarnings("ignore", category=ErfaWarning)
warnings.filterwarnings("ignore", category=AstropyWarning)

import numpy as np
from sgp4.earth_gravity import wgs84
from sgp4.api import Satrec, jday
from astropy.time import Time
from astropy.coordinates import TEME, CartesianRepresentation, ITRS, get_sun
from astropy import units

# 衛星の基準時(epoch)と昇交点地方時の設定
epoch, tle_epoch = Time('2030-01-01 12:00:00', format='iso', scale='utc'), "30001.50000000"
ltan, ltan_str = 6.0, "6:00"


def satellite_position(altitude, start_date, end_date, time_step):
    """
    軌道高度から Dawn-Dusk orbit の TLEを求め、指定された日時間の衛星軌道を SGP4 により
    計算する。SGP4の計算結果と軌道傾斜角、TLEを返す
    """
    # TLEを計算
    tle_line1, tle_line2, inclination_deg, mean_motion = set_TLE(altitude)
    print(f"軌道傾斜角: {inclination_deg:.4f} 度")
    print(f"平均運動: {mean_motion:.8f} 回/日")
    print('TLE :\n', tle_line1, '\n', tle_line2)

    # SGP4オブジェクトの作成
    satellite = Satrec.twoline2rv(tle_line1, tle_line2)

    current_time = start_date
    results = []
    while current_time < end_date:
        #print(current_time)
        # ユリウス日を計算
        year, month = current_time.year, current_time.month
        day, hour = current_time.day, current_time.hour
        minute, second = current_time.minute, current_time.second
        jd, fr = jday(year, month, day, hour, minute, second)

        # SGP4モデルを使用して位置と速度を計算:地心直交座標系 (ECI: Earth-Centered Inertial frame)
        e, r, v = satellite.sgp4(jd, fr)

        if e == 0:  # 正常に計算できた場合
            # ECI 座標 (TEME 座標系) を GCRS 座標系に変換
            time = Time(f"{year}-{month}-{day} {hour}:{minute}:{second}",
                        scale='utc')
            r_teme = CartesianRepresentation(r[0] * units.km, r[1] * units.km, r[2] * units.km)
            teme = TEME(r_teme, obstime=time)
            itrs = teme.transform_to(ITRS(obstime=time))
            # ITRS (ECEF) 座標を経度・緯度に変換
            location = itrs.earth_location
#           lon = location.lon.deg  # 経度
#           lat = location.lat.deg  # 緯度
            results.append((current_time, r, v, location))
        else:
            print('Error')
            exit()
        current_time += time_step
    return results, inclination_deg, tle_line1, tle_line2


def set_TLE(altitude):
    # 定数
    mu = wgs84.mu  # 地球の標準重力パラメータ [km^3/s^2]
    Re = wgs84.radiusearthkm  # 地球の半径 [km]
    J2 = wgs84.j2  # 地球のJ2項
    # 1年の日数(太陽年)  なぜか、これを定義しているモジュールは見つからない。
    days_per_year = 365.24219040
    # 軌道半径
    r = Re + altitude
    # 軌道傾斜角をJ2項を考慮して計算
    i_cos = (-(2 * np.pi / (days_per_year * 86400))
             / (1.5 * J2 * (Re / r)**2 * np.sqrt(mu / r**3)))
    # 軌道傾斜角を求める（範囲内の値をチェック）
    if -1 <= i_cos <= 1:
        inclination_rad = np.arccos(i_cos)  # ラジアン単位の軌道傾斜角
    else:
        print("i_cos out of bounds, defaulting to 97.8 degrees")
        inclination_rad = np.radians(97.8)  # エラー時はデフォルトで97.8度
    # 傾斜角を度に変換
    inclination_deg = np.degrees(inclination_rad)
    # ケプラーの第3法則を使用して軌道周期を計算
    T = 2 * np.pi * np.sqrt(r**3 / mu)  # [秒]
    # 平均運動を計算（1日あたりの回数）
    mean_motion = 86400 / T  # [回/日]
    # 離心率（円軌道を仮定）
    eccentricity = 0.0000001  # 非常に小さな離心率
    # 離心率のフォーマット修正（7桁の正の値、小数点以下のみ)
    eccentricity_str = f"{eccentricity:.7f}".split(".")[1].zfill(7)
    # 昇交点赤経: Right Ascension of the Ascending Node
    raan = ltan_to_raan(ltan, epoch)

    # TLE作成
    #  各行の先頭に行番号
    #  1行目 : カタログ番号 ここでは00000  秘密区分U(unclassified,公開TLE)
    #          国際識別番号 打ち上げの YY,打上番号(3桁),識別符号 で 30001A
    #          エポック、 平均運動の1次微分値:ここではゼロ 平均運動の1次微分値:ここではゼロ
    #          抗力項:ここではゼロ 軌道モデル 0 最後は999が固定、チェックサム1(計算せず)
    #  2行目 : カタログ番号 軌道傾斜角 昇交点赤経 離心率 離心率
    #          近地点引数 ここではゼロ 平均近点角 ここでは90度 これらは円軌道なら多分無意味
    #          平均運動
    tle_line1 = (f"1 00000U 30001A   {tle_epoch}"
                 "  .00000000  00000-0  00000-0 0  9991")
    tle_line2 = (f"2 00000 {inclination_deg:08.4f} {raan:08.4f} "
                 f"{eccentricity_str} 000.0000 090.0000 "
                 f"{mean_motion:17.14f}")
    return tle_line1, tle_line2, inclination_deg, mean_motion


def ltan_to_raan(lt, date):
    # 太陽の視赤経
    solar_ra = get_sun(date).ra.to(units.deg).value
    # LTANをRAANに変換
    raan = (solar_ra - lt * 15) % 360
    return raan
