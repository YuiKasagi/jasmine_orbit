import numpy as np
import pyvista as pv
import sys
from astropy import constants as AC
import quaternion
import math


# -------------------------------------------------------------------------------------
def view_sat(sunlight, Orb, Sat, SatX, SatZ, outfile):  # 衛星を表示
    ORG = np.array([0, 0, 0])
    scale = 1/300  # 描画上のスケール
    # 太陽を方向性の光源として作成し、方向を設定
    sun = pv.Light(light_type='scene light',
                   position=AC.au.value * scale * sunlight,
                   focal_point=ORG, color='white')
    # 地球
    globe = pv.Sphere(radius=AC.R_earth.value * scale, center=ORG)
    # 衛星軌道
    orbit = pv.PolyData(Orb * 1000 * scale)
    orbit.lines = np.hstack([[len(Orb)], np.arange(len(Orb))])
    # JASMINEオブジェクト
    JAS, JASc, JASo = JASMINE()
    satpos = pv.PolyData(Sat * 1000 * scale)

    # 表示を実際に行う
    plotter = pv.Plotter()
    plotter.enable_parallel_projection()  # 平行投影
    plotter.remove_all_lights()
    plotter.add_light(sun)
    plotter.add_mesh(globe, color=(0.20, 0.70, 1.00))
    plotter.add_mesh(orbit, color=(0.20, 0.70, 1.00), line_width=1, point_size=1)
    plotter.add_mesh(satpos, color=(0.20, 0.70, 1.00))
    for i in range(0, len(Sat), 7):  # 飛ばしで表示
        J = rotate_move_block(JAS, SatX[i], SatZ[i], Sat[i]*1000*scale)
        plot_blck(plotter, J, JASc, JASo)

    if outfile:
        plotter.show(auto_close=False)
        plotter.screenshot(outfile)
        plotter.close()
    else:
        plotter.show()


# -----------------------------------------------------------------------------------------
def rotate_block(block, R, theta, P):
    """
    定義済みの図形を点Pを中心に、回転軸Rの周りにtheta 回転した図形を返す
    """
    # 入力ブロックに合わせて出力ブロックを用意
    r_block = pv.MultiBlock([None] * block.n_blocks)
    for i in range(block.n_blocks):
        mesh1 = block[i].translate(-P)   # 回転のために原点に移動
        mesh2 = mesh1.rotate_vector(R, theta)
        r_block[i] = mesh2.translate(P)

    return r_block


# -----------------------------------------------------------------------------------------
def rotate_move_block(block, U, W, P):
    """
    XYZ空間において定義された図形を新しい座標系に合わせるように原点まわりで回転
    X, Z軸を U, W に合うように回転。
    回転したあと、原点が P に来るように平行移動する。
    """
    NX = np.array([1, 0, 0])
    NZ = np.array([0, 0, 1])

    # 入力ブロックに合わせて出力ブロックを用意
    r_block = pv.MultiBlock([None] * block.n_blocks)
    # XYZ空間で定義されたblockを UVW空間にあうよう 回転させたblockを返す Vはどうせ使わない
    # まず、Z軸方向を W 方向に回転する。
    rot = np.cross(NZ, W)
    rotlen = np.linalg.norm(rot)
    if not np.array_equal(NZ, W):  # もともと一致していた場合を除く
        if rotlen > np.finfo(float).eps or rotlen < -np.finfo(float).eps:
            rotax = rot / rotlen
        else:  # 反転のとき
            rotax = NX
        theta = math.acos(np.dot(NZ, W))
        # X軸の回転を行う
        rot_quat = quaternion.from_rotation_vector(theta * rotax)
        NXr = quaternion.as_rotation_matrix(rot_quat).dot(NX)
    else:  # もともと一致なので回転不要
        NXr = NX
        theta = 0
        rot_quat = np.quaternion(1, 0, 0, 0)

    # 次に、回転された NX ベクトルを Wの回りに角度 omega 回して Uに重ねる
    rot2 = np.cross(NXr, U)
    rot2len = np.linalg.norm(rot2)
    if not np.array_equal(NXr, U):  # もともと一致していた場合を除く
        if rot2len > np.finfo(float).eps or rotlen < -np.finfo(float).eps:
            rot2ax = rot2 / rot2len
        else:  # 反転のとき
            rot2ax = W
        omega = math.acos(np.dot(NXr, U))
        rot2_quat = quaternion.from_rotation_vector(omega * rot2ax)
    else:
        omega = 0
        rot2_quat = np.quaternion(1, 0, 0, 0)
    # 以上、rotaxの回りのtheta回転とrot2ax回りのomega回転がすべき回転。
    # 合成
    rotall_quat = rot2_quat * rot_quat

    # 回転軸と角度を求める
    rotation_axis = quaternion.as_rotation_vector(rotall_quat)
    angle = math.degrees(np.linalg.norm(rotation_axis))
    rotation = rotation_axis / angle if angle > 0 else np.array([1, 0, 0])

    omega = math.degrees(omega)
    for i in range(block.n_blocks):
        mesh = block[i].rotate_vector(rotation, angle)
        r_block[i] = mesh.translate(P)

    return r_block


# -----------------------------------------------------------------------------------------
def plot_blck(plotter, block, cmap, opacity):  # メッシュブロックの色と不透明度つきプロット
    for i in range(block.n_blocks):
        plotter.add_mesh(block[i], color=cmap[i], opacity=opacity[i])


# -----------------------------------------------------------------------------------------
def JASMINE():
    """
    JASMINEのモデルブロックを返す
    """
    NZ = np.array([0, 0, 1])
    # バスの大きさは 950x950x950
    BusW, BusD, BusH = (950, 950, 950)
    # SAPは 中心から 995はなれたところから (1285+45)x840 2枚
    SapOff, SapW, SapH, SapD = (995, 1330, 840, 10)
    # ミッション部大きさは H 1053 W D はバスと同じ
    TboxH, TboxW, TboxD = (1053, 950, 950)
    # バッフルは Dia 556 H900 中心から120 太陽より
    BufD, BufH, BufOff = (556, 900, 120)
    # サンシールド H 2840(下から) 幅 1120   1615-885=730 のところ
    SldH, SldW, SldOff, SldD = (2840, 1120, 730, 10)
    # オブジェクトの作成
    Bus = pv.Cube(center=(0, 0, -BusH / 2 - TboxH / 2),
                  x_length=BusD, y_length=BusW, z_length=BusH)
    Busc, Buso = ([255. / 255, 215. / 255, 0.], 1.0)
    SapA = pv.Cube(center=(0, SapW + SapOff, -BusH / 2 - TboxH / 2),
                   x_length=SapD, y_length=SapW * 2, z_length=SapH)
    SapB = pv.Cube(center=(0, -SapW - SapOff, -BusH / 2 - TboxH / 2),
                   x_length=SapD, y_length=SapW * 2, z_length=SapH)
    Sapc, Sapo = ([0., 0., 139 / 225.], 1.0)
    Tbox = pv.Cube(center=(0, 0, 0),
                   x_length=TboxD, y_length=TboxW, z_length=TboxH)
    Tboxc, Tboxo = ([0., 250. / 255, 154. / 255], 1.0)
    Buf = pv.Cylinder(center=(BufOff, 0, BufH / 2 + TboxH / 2),
                      direction=NZ, radius=BufD / 2, height=BufH, capping=False)
    Bufc, Bufo = ([100. / 255, 100. / 255, 100. / 255], 1.0)
    Sld = pv.Cube(center=(SldOff, 0, SldH / 2 - TboxH / 2),
                  x_length=SldD, y_length=SldW, z_length=SldH)
    Sldc, Sldo = ([199. / 255, 21. / 255, 133. / 255], 1.0)
    block = pv.MultiBlock([Bus, SapA, SapB, Tbox, Buf, Sld])
    cmap = [Busc, Sapc, Sapc, Tboxc, Bufc, Sldc]
    opac = [Buso, Sapo, Sapo, Tboxo, Bufo, Sldo]
    return block, cmap, opac


# -------------------------------------------------------------------------------------
def normalize_old(v):  # ベクトルの規格化
    n = np.linalg.norm(v)
    if n < np.finfo(float).eps:
        print('Vector', v, ' length too small')
        sys.exit(1)

    return v / n

def normalize(v, eps=1e-12):
    # v: (...,3)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)