import sys
import os
import warnings
# 忽略 PyQt5 的 DeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
from scipy.interpolate import bisplrep, bisplev, UnivariateSpline
import vtk
import pyvista as pv
import matplotlib.colors as mcolors
from pyvistaqt import BackgroundPlotter
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget,
                             QPushButton, QLabel, QFileDialog, QHBoxLayout, QTextEdit,
                             QCheckBox, QScrollArea, QFrame, QComboBox, QColorDialog,
                             QSlider, QDoubleSpinBox, QShortcut)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPalette, QKeySequence

# occwl imports
from occwl.compound import Compound
from occwl.entity_mapper import EntityMapper
from occwl.face import Face
from occwl.solid import Solid

# OCC imports for triangulation and OBB
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.BRep import BRep_Tool, BRep_Builder
from OCC.Core.Bnd import Bnd_OBB
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib_Add, brepbndlib_AddOBB
from OCC.Core.Poly import Poly_Triangulation, Poly_Triangle, Poly_Array1OfTriangle
from OCC.Core.TColgp import TColgp_Array1OfPnt, TColgp_Array2OfPnt
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax3, gp_Ax1, gp_Ax2
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeShapeOnMesh, BRepBuilderAPI_MakeFace
from OCC.Core.TopoDS import TopoDS_Compound, TopoDS_Face
from OCC.Core.GeomAPI import GeomAPI_PointsToBSplineSurface, GeomAPI_PointsToBSpline
from OCC.Core.GeomAbs import GeomAbs_C2
from OCC.Core.GC import GC_MakeCylindricalSurface, GC_MakeConicalSurface
from OCC.Core.Geom import Geom_SphericalSurface, Geom_SurfaceOfRevolution
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_SurfaceOfRevolution

# 预定义的高亮颜色列表 (用于区分不同行的 face 对)
HIGHLIGHT_COLORS = [
    "#FF0000",  # 红色
    "#00FF00",  # 绿色
    "#0000FF",  # 蓝色
    "#FFFF00",  # 黄色
    "#FF00FF",  # 洋红
    "#00FFFF",  # 青色
    "#FFA500",  # 橙色
    "#800080",  # 紫色
    "#008000",  # 深绿
    "#FFC0CB",  # 粉红
    "#A52A2A",  # 棕色
    "#FFD700",  # 金色
    "#4B0082",  # 靛蓝
    "#EE82EE",  # 紫罗兰
    "#DC143C",  # 深红
    "#00CED1",  # 深青
    "#FF6347",  # 番茄红
    "#32CD32",  # 酸橙绿
    "#BA55D3",  # 中紫
    "#20B2AA",  # 浅海绿
]

# 点云显示数量限制：0=无限制，>0 时从 npz 中均匀采样至该数量（避免巨量点导致卡顿）
POINT_CLOUD_MAX_POINTS = 0

# 概览模式：每个面对最多显示的点数（用于查看朝向，控制性能）
OVERVIEW_POINTS_PER_PAIR = 500

# 点云 Albedo 映射三端颜色（起始 / 中间 / 结束）
ALBEDO_COLOR_LOW = np.array([0.0, 0.0, 0.6])    # 蓝
ALBEDO_COLOR_MID = np.array([0.0, 0.9, 0.3])    # 绿
ALBEDO_COLOR_HIGH = np.array([1.0, 0.2, 0.2])   # 红

# 等值面 B-spline 拟合：网格分辨率、平滑参数
ISOSURFACE_FIT_N_GRID = 50
ISOSURFACE_FIT_SMOOTHING = None  # None 则自动设为 len(points)*0.1

# 曲面表示：非周期 = B-spline (tck + 局部坐标系)，周期 = 参数化曲面 (圆柱/圆锥/球) 或 周期样条 (periodic_bspline / periodic_revolution)
# surface 为 dict: type in ("bspline", "cylinder", "cone", "sphere", "periodic_bspline", "periodic_revolution")


def extend_bspline_surface(surface, ext_v):
    """
    仅对 type=="bspline" 的等值曲面做 UV 延展：扩展量为相对曲面自身 UV 范围的比例。
    ext_v 为比例系数，每侧扩展量 = (Umax-Umin)*ext_v 或 (Vmax-Vmin)*ext_v。
    例如 u:[0,2], v:[0,1]，ext_v=0.5 时 dU=2*0.5=1, dV=1*0.5=0.5，延展后 u:[-1,3], v:[-0.5,1.5]。
    周期面不延展，返回原 surface 的浅拷贝。
    """
    if surface is None:
        return None
    stype = surface.get("type")
    if stype != "bspline":
        return dict(surface)
    u0, u1 = surface["u_bounds"]
    v0, v1 = surface["v_bounds"]
    r = float(ext_v)
    du = (u1 - u0) * r
    dv = (v1 - v0) * r
    out = dict(surface)
    out["u_bounds"] = (u0 - du, u1 + du)
    out["v_bounds"] = (v0 - dv, v1 + dv)
    return out


def sample_surface_to_mesh(surface, n_grid=50):
    """
    将曲面表示采样为三角网格，用于显示与 STEP 导出。
    surface: 由各 fit_* 返回的曲面 dict（B-spline 或参数化）。
    n_grid: 采样分辨率（u/v 方向网格数）。

    Returns:
        grid_points: (N, 3) 顶点
        triangles: (K, 3) 三角面片索引
    若 surface 为 None 或类型未知，返回 (None, None)。
    """
    if surface is None:
        return None, None
    stype = surface.get("type")
    if stype == "bspline":
        tck = surface["tck"]
        centroid = surface["centroid"]
        eigenvectors = surface["eigenvectors"]
        u_bounds = surface["u_bounds"]
        v_bounds = surface["v_bounds"]
        u_margin = (u_bounds[1] - u_bounds[0]) * 0.02 or 1e-6
        v_margin = (v_bounds[1] - v_bounds[0]) * 0.02 or 1e-6
        u_min, u_max = u_bounds[0] + u_margin, u_bounds[1] - u_margin
        v_min, v_max = v_bounds[0] + v_margin, v_bounds[1] - v_margin
        u_grid = np.linspace(u_min, u_max, n_grid)
        v_grid = np.linspace(v_min, v_max, n_grid)
        w_grid = bisplev(u_grid, v_grid, tck)
        grid_points = []
        for i, u in enumerate(u_grid):
            for j, v in enumerate(v_grid):
                local_pt = np.array([u, v, w_grid[i, j]])
                world_pt = local_pt @ eigenvectors.T + centroid
                grid_points.append(world_pt)
        grid_points = np.array(grid_points)
        triangles = []
        for i in range(n_grid - 1):
            for j in range(n_grid - 1):
                idx = i * n_grid + j
                triangles.append([idx, idx + 1, idx + n_grid])
                triangles.append([idx + 1, idx + n_grid + 1, idx + n_grid])
        triangles = np.array(triangles)
        return grid_points, triangles

    if stype == "cylinder":
        origin = surface["origin"]
        axis_dir = surface["axis_dir"]
        x_dir, y_dir = surface["x_dir"], surface["y_dir"]
        radius = surface["radius"]
        v_min, v_max = surface["v_bounds"]
        n_u, n_v = n_grid, max(10, n_grid // 2)
        u_grid = np.linspace(0, 2 * np.pi, n_u, endpoint=False)
        v_grid = np.linspace(v_min, v_max, n_v)
        grid_points = []
        for v in v_grid:
            for u in u_grid:
                pt = origin + v * axis_dir + radius * (np.cos(u) * x_dir + np.sin(u) * y_dir)
                grid_points.append(pt)
        grid_points = np.array(grid_points)
        triangles = []
        for i in range(n_v - 1):
            for j in range(n_u):
                jn = (j + 1) % n_u
                idx = i * n_u + j
                idx_next = i * n_u + jn
                idx_up = (i + 1) * n_u + j
                idx_up_next = (i + 1) * n_u + jn
                triangles.append([idx, idx_next, idx_up])
                triangles.append([idx_next, idx_up_next, idx_up])
        return grid_points, np.array(triangles)

    if stype == "cone":
        apex = surface["apex"]
        axis_dir = surface["axis_dir"]
        x_dir, y_dir = surface["x_dir"], surface["y_dir"]
        semi_angle = surface["semi_angle"]
        v_min, v_max = surface["v_bounds"]
        n_u, n_v = n_grid, max(10, n_grid // 2)
        u_grid = np.linspace(0, 2 * np.pi, n_u, endpoint=False)
        v_grid = np.linspace(v_min, v_max, n_v)
        grid_points = []
        for v in v_grid:
            r_at_v = max(abs(v) * np.tan(semi_angle), 1e-9)
            for u in u_grid:
                pt = apex + v * axis_dir + r_at_v * (np.cos(u) * x_dir + np.sin(u) * y_dir)
                grid_points.append(pt)
        grid_points = np.array(grid_points)
        triangles = []
        for i in range(n_v - 1):
            for j in range(n_u):
                jn = (j + 1) % n_u
                idx = i * n_u + j
                idx_next = i * n_u + jn
                idx_up = (i + 1) * n_u + j
                idx_up_next = (i + 1) * n_u + jn
                triangles.append([idx, idx_next, idx_up])
                triangles.append([idx_next, idx_up_next, idx_up])
        return grid_points, np.array(triangles)

    if stype == "sphere":
        center = surface["center"]
        radius = surface["radius"]
        theta_min, theta_max = surface["theta_bounds"]
        phi_min, phi_max = surface["phi_bounds"]
        n_theta, n_phi = n_grid, max(10, n_grid // 2)
        theta_grid = np.linspace(theta_min, theta_max, n_theta)
        phi_grid = np.linspace(phi_min, phi_max, n_phi)
        grid_points = []
        for phi_val in phi_grid:
            for theta_val in theta_grid:
                st, ct = np.sin(theta_val), np.cos(theta_val)
                sp, cp = np.sin(phi_val), np.cos(phi_val)
                pt = center + radius * np.array([ct * sp, st * sp, cp])
                grid_points.append(pt)
        grid_points = np.array(grid_points)
        triangles = []
        for i in range(n_phi - 1):
            for j in range(n_theta - 1):
                idx = i * n_theta + j
                triangles.append([idx, idx + 1, idx + n_theta])
                triangles.append([idx + 1, idx + n_theta + 1, idx + n_theta])
        return grid_points, np.array(triangles)

    if stype == "periodic_bspline":
        origin = surface["origin"]
        axis_dir = surface["axis_dir"]
        x_dir, y_dir = surface["x_dir"], surface["y_dir"]
        tck = surface["tck"]
        v_min, v_max = surface["v_bounds"]
        n_u, n_v = n_grid, max(20, n_grid)
        u_grid = np.linspace(0, 2 * np.pi, n_u, endpoint=False)
        v_grid = np.linspace(v_min, v_max, n_v)
        r_grid = bisplev(u_grid, v_grid, tck)
        grid_points = []
        for i, u in enumerate(u_grid):
            for j, v in enumerate(v_grid):
                r = float(np.maximum(r_grid[i, j], 1e-9))
                pt = origin + v * axis_dir + r * (np.cos(u) * x_dir + np.sin(u) * y_dir)
                grid_points.append(pt)
        grid_points = np.array(grid_points)
        triangles = []
        for i in range(n_v - 1):
            for j in range(n_u):
                jn = (j + 1) % n_u
                idx = i * n_u + j
                idx_next = i * n_u + jn
                idx_up = (i + 1) * n_u + j
                idx_up_next = (i + 1) * n_u + jn
                triangles.append([idx, idx_next, idx_up])
                triangles.append([idx_next, idx_up_next, idx_up])
        return grid_points, np.array(triangles)

    if stype == "periodic_revolution":
        origin = surface["origin"]
        axis_dir = surface["axis_dir"]
        x_dir, y_dir = surface["x_dir"], surface["y_dir"]
        r_spline = surface["r_spline"]
        v_min, v_max = surface["v_bounds"]
        n_u, n_v = n_grid, max(20, n_grid)
        u_grid = np.linspace(0, 2 * np.pi, n_u, endpoint=False)
        v_grid = np.linspace(v_min, v_max, n_v)
        grid_points = []
        for v in v_grid:
            r = float(np.maximum(r_spline(v), 1e-9))
            for u in u_grid:
                pt = origin + v * axis_dir + r * (np.cos(u) * x_dir + np.sin(u) * y_dir)
                grid_points.append(pt)
        grid_points = np.array(grid_points)
        triangles = []
        for i in range(n_v - 1):
            for j in range(n_u):
                jn = (j + 1) % n_u
                idx = i * n_u + j
                idx_next = i * n_u + jn
                idx_up = (i + 1) * n_u + j
                idx_up_next = (i + 1) * n_u + jn
                triangles.append([idx, idx_next, idx_up])
                triangles.append([idx_next, idx_up_next, idx_up])
        return grid_points, np.array(triangles)

    return None, None


# Albedo 映射数据源选项
ALBEDO_DATA_SOURCES = [
    ("offset_pred", "Offset 预测值"),
    ("offset_gt", "Offset 真实值"),
    ("validity_pred", "Validity 预测值"),
    ("validity_gt", "Validity 真实值"),
]


def fit_nurbs_surface_from_points(points, n_grid=50, smoothing=None):
    """
    用 B-spline 拟合点云，得到 B-spline 曲面表示（非周期）。
    fit_ref.py 思路：PCA 主平面参数化 → bisplrep 拟合 w=f(u,v)，曲面以 (tck, 局部坐标系) 表示。

    Args:
        points: (M, 3) 待拟合的 3D 点
        n_grid: 仅用于兼容接口，实际采样由 sample_surface_to_mesh(surface, n_grid) 完成
        smoothing: 平滑参数，None 则自动

    Returns:
        surface: dict type="bspline"（tck, centroid, eigenvectors, u_bounds, v_bounds），或 None
        info: dict 或 None
    """
    if len(points) < 16:
        return None, None
    centroid = points.mean(axis=0)
    pts_centered = points - centroid
    cov = np.cov(pts_centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    local_coords = pts_centered @ eigenvectors
    u_coords, v_coords, w_coords = local_coords[:, 0], local_coords[:, 1], local_coords[:, 2]
    if smoothing is None:
        smoothing = len(points) * 0.1
    try:
        tck = bisplrep(u_coords, v_coords, w_coords, s=smoothing, kx=3, ky=3)
    except Exception:
        try:
            tck = bisplrep(u_coords, v_coords, w_coords, s=len(points) * 1.0, kx=3, ky=3)
        except Exception:
            return None, None
    u_min, u_max = float(np.min(u_coords)), float(np.max(u_coords))
    v_min, v_max = float(np.min(v_coords)), float(np.max(v_coords))
    surface = {
        "type": "bspline",
        "tck": tck,
        "centroid": centroid,
        "eigenvectors": eigenvectors,
        "u_bounds": (u_min, u_max),
        "v_bounds": (v_min, v_max),
    }
    info = {"n_input_points": len(points), "type": "non-periodic"}
    return surface, info


def _get_face_surface_type(face):
    """
    获取 occwl Face 的曲面类型。
    返回 "cylinder" | "cone" | "periodic" | None（非周期或无法识别）。
    周期面：Cylinder, Cone, Sphere, Torus, SurfaceOfRevolution。
    """
    try:
        shape = face.topods_shape()
        adapt = BRepAdaptor_Surface(shape)
        stype = adapt.GetType()
        if stype == GeomAbs_Cylinder:
            return "cylinder"
        if stype == GeomAbs_Cone:
            return "cone"
        if stype == GeomAbs_Sphere:
            return "sphere"
        if stype in (GeomAbs_Torus, GeomAbs_SurfaceOfRevolution):
            return "periodic"
        return None
    except Exception:
        return None


def _get_cylinder_params_from_face(face):
    """从圆柱面 Face 取 (origin, axis_direction, radius)，numpy 数组。失败返回 None。"""
    try:
        shape = face.topods_shape()
        adapt = BRepAdaptor_Surface(shape)
        if adapt.GetType() != GeomAbs_Cylinder:
            return None
        try:
            cyl = adapt.Cylinder()
        except Exception:
            cyl = adapt.Surface().Cylinder()
        pos = cyl.Position()
        loc = pos.Location()
        ax = pos.Direction()
        origin = np.array([loc.X(), loc.Y(), loc.Z()], dtype=np.float64)
        axis_dir = np.array([ax.X(), ax.Y(), ax.Z()], dtype=np.float64)
        axis_dir = axis_dir / (np.linalg.norm(axis_dir) + 1e-12)
        radius = cyl.Radius()
        return origin, axis_dir, radius
    except Exception:
        return None


def _get_cone_params_from_face(face):
    """从圆锥面 Face 取 (apex, axis_direction, semi_angle)，numpy 数组。失败返回 None。"""
    try:
        shape = face.topods_shape()
        adapt = BRepAdaptor_Surface(shape)
        if adapt.GetType() != GeomAbs_Cone:
            return None
        try:
            cone = adapt.Cone()
        except Exception:
            cone = adapt.Surface().Cone()
        pos = cone.Position()
        apex = pos.Location()
        ax = pos.Direction()
        apex_pt = np.array([apex.X(), apex.Y(), apex.Z()], dtype=np.float64)
        axis_dir = np.array([ax.X(), ax.Y(), ax.Z()], dtype=np.float64)
        axis_dir = axis_dir / (np.linalg.norm(axis_dir) + 1e-12)
        semi_angle = cone.SemiAngle()
        return apex_pt, axis_dir, semi_angle
    except Exception:
        return None


def _get_sphere_params_from_face(face):
    """从球面 Face 取 (center, radius)，numpy 数组。失败返回 None。"""
    try:
        shape = face.topods_shape()
        adapt = BRepAdaptor_Surface(shape)
        if adapt.GetType() != GeomAbs_Sphere:
            return None
        try:
            sph = adapt.Sphere()
        except Exception:
            sph = adapt.Surface().Sphere()
        pos = sph.Position()
        center = pos.Location()
        center_pt = np.array([center.X(), center.Y(), center.Z()], dtype=np.float64)
        radius = sph.Radius()
        return center_pt, radius
    except Exception:
        return None


def _get_revolution_axis_from_face(face):
    """从周期面（旋转面等）取 (origin, axis_direction)。失败返回 None。"""
    try:
        shape = face.topods_shape()
        adapt = BRepAdaptor_Surface(shape)
        stype = adapt.GetType()
        if stype == GeomAbs_Cylinder:
            r = _get_cylinder_params_from_face(face)
            return (r[0], r[1]) if r else None
        if stype == GeomAbs_Cone:
            r = _get_cone_params_from_face(face)
            return (r[0], r[1]) if r else None
        if stype == GeomAbs_SurfaceOfRevolution:
            try:
                rev = adapt.Surface().Surface()
            except Exception:
                rev = adapt.Surface()
            ax1 = rev.Axis()
            loc = ax1.Location()
            direc = ax1.Direction()
            origin = np.array([loc.X(), loc.Y(), loc.Z()], dtype=np.float64)
            axis_dir = np.array([direc.X(), direc.Y(), direc.Z()], dtype=np.float64)
            axis_dir = axis_dir / (np.linalg.norm(axis_dir) + 1e-12)
            return origin, axis_dir
        if stype in (GeomAbs_Sphere, GeomAbs_Torus):
            centroid = np.array([0.0, 0.0, 0.0])
            axis_dir = np.array([0.0, 0.0, 1.0])
            return centroid, axis_dir
        return None
    except Exception:
        return None


def _make_perp_frame(axis_dir):
    """给定单位向量 axis_dir (z)，返回 (x_dir, y_dir) 单位正交，与 OCC 右手系一致。"""
    z = np.asarray(axis_dir, dtype=np.float64)
    z = z / (np.linalg.norm(z) + 1e-12)
    if abs(z[2]) < 0.9:
        x = np.cross(z, np.array([0, 0, 1.0]))
    else:
        x = np.cross(z, np.array([1.0, 0, 0]))
    x = x / (np.linalg.norm(x) + 1e-12)
    y = np.cross(z, x)
    y = y / (np.linalg.norm(y) + 1e-12)
    return x, y


def fit_cylinder_surface_from_points(points, origin, axis_dir, radius, n_grid=50):
    """
    用圆柱面（参数化曲面）拟合点云，中轴与朝向与给定 origin/axis_dir 一致。
    返回参数化曲面表示 (surface, info)，采样由 sample_surface_to_mesh(surface, n_grid) 完成。
    """
    if len(points) < 8:
        return None, None
    points = np.asarray(points, dtype=np.float64)
    axis_dir = np.asarray(axis_dir, dtype=np.float64)
    axis_dir = axis_dir / (np.linalg.norm(axis_dir) + 1e-12)
    x_dir, y_dir = _make_perp_frame(axis_dir)
    v_vals = (points - origin) @ axis_dir
    v_min, v_max = float(np.min(v_vals)), float(np.max(v_vals))
    if v_max <= v_min:
        v_max = v_min + 1e-6
    r_vals = np.linalg.norm((points - origin) - np.outer((points - origin) @ axis_dir, axis_dir), axis=1)
    r_median = float(np.median(r_vals)) if len(r_vals) else radius
    r_median = max(r_median, 1e-9)
    surface = {
        "type": "cylinder",
        "origin": origin,
        "axis_dir": axis_dir,
        "x_dir": x_dir,
        "y_dir": y_dir,
        "radius": r_median,
        "v_bounds": (v_min, v_max),
    }
    info = {"n_input_points": len(points), "type": "cylinder"}
    return surface, info


def fit_sphere_surface_from_points(points, center, radius, n_grid=50):
    """
    用球面（参数化曲面）拟合点云，球心与半径与给定 center/radius 一致；半径用点云到球心距离中值微调。
    参数化：theta (经度), phi (余纬)，仅覆盖点云角范围。采样由 sample_surface_to_mesh(surface, n_grid) 完成。
    """
    if len(points) < 8:
        return None, None
    points = np.asarray(points, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    to_pts = points - center
    dist = np.linalg.norm(to_pts, axis=1)
    r_median = float(np.median(dist)) if len(dist) else radius
    r_median = max(r_median, 1e-9)
    z = to_pts[:, 2]
    xy = np.linalg.norm(to_pts[:, :2], axis=1)
    theta = np.arctan2(to_pts[:, 1], to_pts[:, 0])
    phi = np.arctan2(xy, z)
    theta_min, theta_max = float(np.min(theta)), float(np.max(theta))
    phi_min, phi_max = float(np.min(phi)), float(np.max(phi))
    if theta_max <= theta_min:
        theta_max = theta_min + 2 * np.pi
    if phi_max <= phi_min:
        phi_max = phi_min + 1e-6
    surface = {
        "type": "sphere",
        "center": center,
        "radius": r_median,
        "theta_bounds": (theta_min, theta_max),
        "phi_bounds": (phi_min, phi_max),
    }
    info = {"n_input_points": len(points), "type": "sphere"}
    return surface, info


def fit_cone_surface_from_points(points, apex, axis_dir, semi_angle, n_grid=50):
    """
    用圆锥面（参数化曲面）拟合点云，中轴与朝向与给定 apex/axis_dir 一致。
    返回参数化曲面表示 (surface, info)，采样由 sample_surface_to_mesh(surface, n_grid) 完成。
    """
    if len(points) < 8:
        return None, None
    points = np.asarray(points, dtype=np.float64)
    axis_dir = np.asarray(axis_dir, dtype=np.float64)
    axis_dir = axis_dir / (np.linalg.norm(axis_dir) + 1e-12)
    x_dir, y_dir = _make_perp_frame(axis_dir)
    to_pts = points - apex
    v_vals = to_pts @ axis_dir
    v_min, v_max = float(np.min(v_vals)), float(np.max(v_vals))
    if v_max <= v_min:
        v_max = v_min + 1e-6
    surface = {
        "type": "cone",
        "apex": apex,
        "axis_dir": axis_dir,
        "x_dir": x_dir,
        "y_dir": y_dir,
        "semi_angle": semi_angle,
        "v_bounds": (v_min, v_max),
    }
    info = {"n_input_points": len(points), "type": "cone"}
    return surface, info


def fit_periodic_revolution_from_points(points, origin, axis_dir, n_grid=50):
    """
    通用周期面（旋转面）参数化拟合：以给定轴为旋转轴，半径 r(v) 用 1D B-spline 从点云估计。
    返回参数化曲面表示 (surface, info)，S(u,v)=origin+v*axis_dir+r_spline(v)*(cos(u)*x+sin(u)*y)。
    采样由 sample_surface_to_mesh(surface, n_grid) 完成。
    """
    if len(points) < 16:
        return None, None
    points = np.asarray(points, dtype=np.float64)
    axis_dir = np.asarray(axis_dir, dtype=np.float64)
    axis_dir = axis_dir / (np.linalg.norm(axis_dir) + 1e-12)
    x_dir, y_dir = _make_perp_frame(axis_dir)
    v_vals = (points - origin) @ axis_dir
    r_vals = np.linalg.norm((points - origin) - np.outer((points - origin) @ axis_dir, axis_dir), axis=1)
    v_min, v_max = float(np.min(v_vals)), float(np.max(v_vals))
    if v_max <= v_min:
        v_max = v_min + 1e-6
    n_v = max(20, min(n_grid, len(points) // 2))
    v_edges = np.linspace(v_min, v_max, n_v + 1)
    v_centers = (v_edges[:-1] + v_edges[1:]) / 2
    r_at_v = []
    for i in range(n_v):
        mask = (v_vals >= v_edges[i]) & (v_vals < v_edges[i + 1])
        if np.any(mask):
            r_at_v.append(float(np.median(r_vals[mask])))
        else:
            r_at_v.append(0.0 if not r_at_v else r_at_v[-1])
    r_at_v = np.array(r_at_v)
    r_at_v = np.maximum(r_at_v, 1e-9)
    try:
        r_spline = UnivariateSpline(v_centers, r_at_v, k=min(3, n_v - 1), s=0)
    except Exception:
        def r_spline(v):
            return float(np.maximum(np.interp(np.asarray(v, dtype=float), v_centers, r_at_v), 1e-9))
    surface = {
        "type": "periodic_revolution",
        "origin": origin,
        "axis_dir": axis_dir,
        "x_dir": x_dir,
        "y_dir": y_dir,
        "r_spline": r_spline,
        "v_bounds": (v_min, v_max),
    }
    info = {"n_input_points": len(points), "type": "periodic_revolution"}
    return surface, info


def fit_periodic_spline_surface_from_points(points, origin, axis_dir, n_grid=50, smoothing=None):
    """
    非圆柱/圆锥/球的周期面（如 Torus、一般旋转面）：用 scipy 的 bisplrep 做周期性样条曲面拟合。
    参数化 (u,v)：u 为绕轴角度 [0, 2*pi)，v 为沿轴坐标，拟合 r = r(u,v)。通过复制 u=0/2*pi 两侧数据实现 u 方向周期。
    参考 fit_ref.py：bisplrep 拟合 → 曲面 S(u,v) = origin + v*axis_dir + r(u,v)*(cos(u)*x_dir + sin(u)*y_dir)。
    返回 (surface, info)，surface["type"] == "periodic_bspline"。
    """
    print("通用周期面拟合")

    if len(points) < 16:
        return None, None
    points = np.asarray(points, dtype=np.float64)
    axis_dir = np.asarray(axis_dir, dtype=np.float64)
    axis_dir = axis_dir / (np.linalg.norm(axis_dir) + 1e-12)
    x_dir, y_dir = _make_perp_frame(axis_dir)
    to_origin = points - origin
    v_vals = to_origin @ axis_dir
    radial = to_origin - np.outer(v_vals, axis_dir)
    r_vals = np.linalg.norm(radial, axis=1)
    r_vals = np.maximum(r_vals, 1e-9)
    u_vals = np.arctan2(radial @ y_dir, radial @ x_dir)
    u_vals = np.where(u_vals < 0, u_vals + 2 * np.pi, u_vals)
    v_min, v_max = float(np.min(v_vals)), float(np.max(v_vals))
    if v_max <= v_min:
        v_max = v_min + 1e-6
    u_ext = np.concatenate([u_vals, u_vals + 2 * np.pi, u_vals - 2 * np.pi])
    v_ext = np.concatenate([v_vals, v_vals, v_vals])
    r_ext = np.concatenate([r_vals, r_vals, r_vals])
    if smoothing is None:
        smoothing = len(points) * 0.1
    try:
        tck = bisplrep(u_ext, v_ext, r_ext, s=smoothing, kx=3, ky=3)
    except Exception:
        try:
            tck = bisplrep(u_ext, v_ext, r_ext, s=len(points) * 1.0, kx=3, ky=3)
        except Exception:
            return None, None
    surface = {
        "type": "periodic_bspline",
        "origin": origin,
        "axis_dir": axis_dir,
        "x_dir": x_dir,
        "y_dir": y_dir,
        "tck": tck,
        "u_bounds": (0.0, 2 * np.pi),
        "v_bounds": (v_min, v_max),
    }
    info = {"n_input_points": len(points), "type": "periodic_bspline"}
    return surface, info


def fit_isosurface_by_face_type(points, left_face, right_face, n_grid=50):
    """
    根据左右面类型选择拟合方式：双圆柱->圆柱参数化，双圆锥->圆锥参数化，双球面->球面参数化，
    双周期(非圆柱圆锥球)->通用旋转面参数化，否则->非周期 B-spline 曲面。
    left_face, right_face 为 occwl Face 或 None（无模型时）。
    返回 (surface, info)：surface 为 B-spline 或参数化曲面 dict，采样由 sample_surface_to_mesh(surface, n_grid) 得到网格。
    """
    if left_face is None or right_face is None:
        return fit_nurbs_surface_from_points(points, n_grid=n_grid)
    t_left = _get_face_surface_type(left_face)
    t_right = _get_face_surface_type(right_face)
    if t_left != t_right:
        return fit_nurbs_surface_from_points(points, n_grid=n_grid)
    if t_left == "cylinder":
        params_left = _get_cylinder_params_from_face(left_face)
        params_right = _get_cylinder_params_from_face(right_face)
        if params_left is not None:
            origin, axis_dir, radius = params_left
            return fit_cylinder_surface_from_points(points, origin, axis_dir, radius, n_grid)
        if params_right is not None:
            origin, axis_dir, radius = params_right
            return fit_cylinder_surface_from_points(points, origin, axis_dir, radius, n_grid)
    if t_left == "cone":
        params_left = _get_cone_params_from_face(left_face)
        params_right = _get_cone_params_from_face(right_face)
        if params_left is not None:
            apex, axis_dir, semi_angle = params_left
            return fit_cone_surface_from_points(points, apex, axis_dir, semi_angle, n_grid)
        if params_right is not None:
            apex, axis_dir, semi_angle = params_right
            return fit_cone_surface_from_points(points, apex, axis_dir, semi_angle, n_grid)
    if t_left == "sphere":
        params_left = _get_sphere_params_from_face(left_face)
        params_right = _get_sphere_params_from_face(right_face)
        if params_left is not None:
            center, radius = params_left
            return fit_sphere_surface_from_points(points, center, radius, n_grid)
        if params_right is not None:
            center, radius = params_right
            return fit_sphere_surface_from_points(points, center, radius, n_grid)
    if t_left == "periodic":
        axis_left = _get_revolution_axis_from_face(left_face)
        axis_right = _get_revolution_axis_from_face(right_face)
        origin, axis_dir = (axis_left or axis_right or (np.zeros(3), np.array([0, 0, 1.0])))
        if axis_left is None and axis_right is None:
            return fit_nurbs_surface_from_points(points, n_grid=n_grid)
        return fit_periodic_spline_surface_from_points(points, origin, axis_dir, n_grid=n_grid)
    return fit_nurbs_surface_from_points(points, n_grid=n_grid)


def polydata_to_occ_shape(grid_points, triangles):
    """
    将 (vertices, triangles) 转为 OCC TopoDS_Shape，用于 STEP 导出。
    grid_points: (N, 3) 顶点坐标，triangles: (K, 3) 三角面片索引（0-based）。
    """
    n_pts = len(grid_points)
    n_tri = len(triangles)
    nodes = TColgp_Array1OfPnt(1, n_pts)
    for i in range(n_pts):
        nodes.SetValue(i + 1, gp_Pnt(float(grid_points[i, 0]), float(grid_points[i, 1]), float(grid_points[i, 2])))
    tris = Poly_Array1OfTriangle(1, n_tri)
    for i in range(n_tri):
        # Poly_Triangle 使用 1-based 索引
        t = Poly_Triangle(int(triangles[i, 0]) + 1, int(triangles[i, 1]) + 1, int(triangles[i, 2]) + 1)
        tris.SetValue(i + 1, t)
    poly = Poly_Triangulation(nodes, tris)
    builder = BRepBuilderAPI_MakeShapeOnMesh(poly)
    builder.Build()
    if not builder.IsDone():
        return None
    return builder.Shape()


# 采样网格分辨率，用于将 B-spline/周期样条转为 OCC 点阵再拟合成 OCC 曲面
OCC_SURFACE_APPROX_GRID = 40


def _surface_to_points_2d(surface, n_approx):
    """将 surface 采样为网格点，返回 (points_2d, u_min, u_max, v_min, v_max)。用于转为 OCC TColgp_Array2OfPnt。"""
    grid_points, _ = sample_surface_to_mesh(surface, n_grid=n_approx)
    if grid_points is None:
        return None, None, None, None, None
    stype = surface.get("type")
    if stype == "bspline":
        nu = nv = n_approx
        pts = grid_points.reshape(nu, nv, 3)
        u_bounds = surface["u_bounds"]
        v_bounds = surface["v_bounds"]
        u_margin = (u_bounds[1] - u_bounds[0]) * 0.02 or 1e-6
        v_margin = (v_bounds[1] - v_bounds[0]) * 0.02 or 1e-6
        u_min = u_bounds[0] + u_margin
        u_max = u_bounds[1] - u_margin
        v_min = v_bounds[0] + v_margin
        v_max = v_bounds[1] - v_margin
        return pts, u_min, u_max, v_min, v_max
    if stype in ("cylinder", "cone", "periodic_bspline", "periodic_revolution"):
        v_min, v_max = surface["v_bounds"]
        n_u_act = n_approx
        n_v_act = len(grid_points) // n_u_act
        if n_v_act < 2:
            return None, None, None, None, None
        pts = grid_points.reshape(n_v_act, n_u_act, 3)
        return pts, 0.0, 2 * np.pi, v_min, v_max
    if stype == "sphere":
        theta_min, theta_max = surface["theta_bounds"]
        phi_min, phi_max = surface["phi_bounds"]
        n_theta = n_approx
        n_phi = len(grid_points) // n_theta
        if n_phi < 2:
            return None, None, None, None, None
        pts = grid_points.reshape(n_phi, n_theta, 3)
        return pts, theta_min, theta_max, phi_min, phi_max
    return None, None, None, None, None


def surface_to_occ_face(surface, n_approx=OCC_SURFACE_APPROX_GRID):
    """
    将内部曲面表示（B-spline 或参数化）转为 OCC TopoDS_Face（B-spline 或解析曲面），用于 STEP 导出。
    解析曲面（圆柱、圆锥、球）直接构造 OCC 几何；B-spline/周期样条先采样再 GeomAPI_PointsToBSplineSurface 近似。
    失败返回 None。
    """
    if surface is None:
        return None
    stype = surface.get("type")
    try:
        if stype == "cylinder":
            origin = np.asarray(surface["origin"], dtype=np.float64)
            axis_dir = np.asarray(surface["axis_dir"], dtype=np.float64)
            axis_dir = axis_dir / (np.linalg.norm(axis_dir) + 1e-12)
            x_dir = np.asarray(surface["x_dir"], dtype=np.float64)
            radius = float(surface["radius"])
            v_min, v_max = surface["v_bounds"]
            ax2 = gp_Ax2(gp_Pnt(origin[0], origin[1], origin[2]), gp_Dir(axis_dir[0], axis_dir[1], axis_dir[2]))
            result = GC_MakeCylindricalSurface(ax2, radius)
            if not result.IsDone():
                return None
            geom = result.Value()
            face_builder = BRepBuilderAPI_MakeFace(geom, 0.0, 2 * np.pi, v_min, v_max, 1e-6)
            if not face_builder.IsDone():
                return None
            return face_builder.Face()

        if stype == "cone":
            apex = np.asarray(surface["apex"], dtype=np.float64)
            axis_dir = np.asarray(surface["axis_dir"], dtype=np.float64)
            axis_dir = axis_dir / (np.linalg.norm(axis_dir) + 1e-12)
            semi_angle = float(surface["semi_angle"])
            v_min, v_max = surface["v_bounds"]
            ax2 = gp_Ax2(gp_Pnt(apex[0], apex[1], apex[2]), gp_Dir(axis_dir[0], axis_dir[1], axis_dir[2]))
            result = GC_MakeConicalSurface(ax2, semi_angle, 1e-9)
            if not result.IsDone():
                return None
            geom = result.Value()
            face_builder = BRepBuilderAPI_MakeFace(geom, 0.0, 2 * np.pi, v_min, v_max, 1e-6)
            if not face_builder.IsDone():
                return None
            return face_builder.Face()

        if stype == "sphere":
            center = np.asarray(surface["center"], dtype=np.float64)
            radius = float(surface["radius"])
            theta_min, theta_max = surface["theta_bounds"]
            phi_min, phi_max = surface["phi_bounds"]
            ax3 = gp_Ax3(gp_Pnt(center[0], center[1], center[2]), gp_Dir(0, 0, 1))
            geom = Geom_SphericalSurface(ax3, radius)
            face_builder = BRepBuilderAPI_MakeFace(geom, theta_min, theta_max, phi_min, phi_max, 1e-6)
            if not face_builder.IsDone():
                return None
            return face_builder.Face()

        if stype == "periodic_revolution":
            origin = np.asarray(surface["origin"], dtype=np.float64)
            axis_dir = np.asarray(surface["axis_dir"], dtype=np.float64)
            axis_dir = axis_dir / (np.linalg.norm(axis_dir) + 1e-12)
            x_dir = np.asarray(surface["x_dir"], dtype=np.float64)
            r_spline = surface["r_spline"]
            v_min, v_max = surface["v_bounds"]
            n_curve = max(20, n_approx)
            v_vals = np.linspace(v_min, v_max, n_curve)
            curve_pts = [origin + v * axis_dir + max(float(r_spline(v)), 1e-9) * x_dir for v in v_vals]
            pts_arr = TColgp_Array1OfPnt(1, n_curve)
            for i, pt in enumerate(curve_pts):
                pts_arr.SetValue(i + 1, gp_Pnt(float(pt[0]), float(pt[1]), float(pt[2])))
            curve_builder = GeomAPI_PointsToBSpline(pts_arr, 3, 8)
            if not curve_builder.IsDone():
                return None
            meridian = curve_builder.Curve()
            ax1 = gp_Ax1(gp_Pnt(origin[0], origin[1], origin[2]), gp_Dir(axis_dir[0], axis_dir[1], axis_dir[2]))
            geom = Geom_SurfaceOfRevolution(meridian, ax1)
            cv_min = meridian.FirstParameter()
            cv_max = meridian.LastParameter()
            face_builder = BRepBuilderAPI_MakeFace(geom, 0.0, 2 * np.pi, cv_min, cv_max, 1e-6)
            if not face_builder.IsDone():
                return None
            return face_builder.Face()

        if stype in ("bspline", "periodic_bspline"):
            # 对 B-spline/周期样条曲面：用规则网格采样 + GeomAPI_PointsToBSplineSurface 近似成 OCC 的 Geom_BSplineSurface，
            # 然后用自然 UV 范围创建 Face（不再用 Python 侧 PCA 的 u/v 范围去裁剪，避免参数域不一致导致的导出问题）。
            pts_2d, _, _, _, _ = _surface_to_points_2d(surface, n_approx)
            if pts_2d is None:
                return None
            nu, nv = pts_2d.shape[0], pts_2d.shape[1]
            arr = TColgp_Array2OfPnt(1, nu, 1, nv)
            for i in range(nu):
                for j in range(nv):
                    p = pts_2d[i, j]
                    arr.SetValue(i + 1, j + 1, gp_Pnt(float(p[0]), float(p[1]), float(p[2])))
            approx = GeomAPI_PointsToBSplineSurface(arr, 3, 8, GeomAbs_C2, 1.0e-3)
            if not approx.IsDone():
                return None
            geom = approx.Surface()
            # 使用自然参数域构造 Face，避免与 Python 侧 PCA u/v 范围不一致
            face_builder = BRepBuilderAPI_MakeFace(geom, 1.0e-6)
            if not face_builder.IsDone():
                return None
            return face_builder.Face()
    except Exception:
        return None
    return None


def compute_obb_for_faces(step_path, left_tag, right_tag):
    """根据左右面 idx 计算二者合并的 Oriented Bounding Box (OBB) 参数。
    与 renderer_0.py 中 compute_obb_for_faces 保持一致。"""
    if not os.path.exists(step_path):
        raise FileNotFoundError(f"STEP 文件不存在: {step_path}")
    loaded = Compound.load_from_step(step_path)
    solids = list(loaded.solids()) if hasattr(loaded, "solids") else []
    if not solids and isinstance(loaded, Solid):
        solids = [loaded]
    if not solids:
        try:
            solids = list(loaded)
        except Exception:
            pass
    if not solids:
        raise ValueError("未找到 Solid")
    solid = solids[0]
    mapper = EntityMapper(solid)
    index_to_face = {}
    for face in solid.faces():
        index_to_face[mapper.face_index(face)] = face
    if left_tag not in index_to_face:
        raise ValueError(f"左面 tag {left_tag} 未找到")
    if right_tag not in index_to_face:
        raise ValueError(f"右面 tag {right_tag} 未找到")
    left_shape = index_to_face[left_tag].topods_shape()
    right_shape = index_to_face[right_tag].topods_shape()

    obb = Bnd_OBB()
    brepbndlib_AddOBB(left_shape, obb)
    brepbndlib_AddOBB(right_shape, obb)

    center = obb.Center()
    xh, yh, zh = obb.XHSize(), obb.YHSize(), obb.ZHSize()
    x_dir, y_dir, z_dir = obb.XDirection(), obb.YDirection(), obb.ZDirection()
    obb_origin = np.array([
        center.X() - xh * x_dir.X() - yh * y_dir.X() - zh * z_dir.X(),
        center.Y() - xh * x_dir.Y() - yh * y_dir.Y() - zh * z_dir.Y(),
        center.Z() - xh * x_dir.Z() - yh * y_dir.Z() - zh * z_dir.Z(),
    ], dtype=np.float64)
    obb_x_vec = np.array([x_dir.X(), x_dir.Y(), x_dir.Z()], dtype=np.float64) * (2.0 * xh)
    obb_y_vec = np.array([y_dir.X(), y_dir.Y(), y_dir.Z()], dtype=np.float64) * (2.0 * yh)
    obb_z_vec = np.array([z_dir.X(), z_dir.Y(), z_dir.Z()], dtype=np.float64) * (2.0 * zh)
    return obb_origin, obb_x_vec, obb_y_vec, obb_z_vec


def get_obb_center(obb_origin, obb_x_vec, obb_y_vec, obb_z_vec):
    """从 OBB 参数计算几何中心"""
    return obb_origin + 0.5 * obb_x_vec + 0.5 * obb_y_vec + 0.5 * obb_z_vec


def compute_solid_center(shape):
    """计算 shape 的包围盒几何中心（用于镜头 target）"""
    bbox = Bnd_Box()
    brepbndlib_Add(shape, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    return np.array([
        (xmin + xmax) / 2.0,
        (ymin + ymax) / 2.0,
        (zmin + zmax) / 2.0,
    ], dtype=np.float64)


def _lines_to_tube_polydata(lines_pv, radius, n_sides=6):
    """将折线转为世界空间固定粗细的 tube mesh（用于 albedo unlit 渲染）"""
    if lines_pv is None or lines_pv.n_points == 0:
        return None
    tube = vtk.vtkTubeFilter()
    tube.SetInputData(lines_pv)
    tube.SetRadius(radius)
    tube.SetNumberOfSides(n_sides)
    tube.CappingOff()
    tube.Update()
    return pv.wrap(tube.GetOutput())


def _merge_deduplicate_edges(edge_polydatas):
    """合并多个 edge polydata，去重共用边（相同端点视为同一条边）"""
    seen = set()
    all_points = []
    all_lines = []

    def _add_segment(a, b):
        key = tuple(sorted([
            (round(float(a[0]), 8), round(float(a[1]), 8), round(float(a[2]), 8)),
            (round(float(b[0]), 8), round(float(b[1]), 8), round(float(b[2]), 8))
        ]))
        if key in seen:
            return
        seen.add(key)
        i0, i1 = len(all_points), len(all_points) + 1
        all_points.append(np.asarray(a).tolist())
        all_points.append(np.asarray(b).tolist())
        all_lines.append([2, i0, i1])

    for pd in edge_polydatas:
        if pd is None or pd.n_points < 2:
            continue
        pts = np.asarray(pd.points)
        for i in range(pd.n_cells):
            cell = pd.get_cell(i)
            if cell.n_points < 2:
                continue
            inds = list(cell.point_ids)
            for j in range(len(inds) - 1):
                _add_segment(pts[inds[j]], pts[inds[j + 1]])
    if not all_points or not all_lines:
        return None
    pts_arr = np.array(all_points)
    lines_arr = np.hstack(all_lines)
    return pv.PolyData(pts_arr, lines=lines_arr)


def _create_obb_wireframe_polydata(obb_origin, obb_x_vec, obb_y_vec, obb_z_vec):
    """创建 OBB 的 12 条棱边为折线 Polydata"""
    o = np.asarray(obb_origin)
    x, y, z = np.asarray(obb_x_vec), np.asarray(obb_y_vec), np.asarray(obb_z_vec)
    # 8 个顶点 (0,0,0) 到 (1,1,1)
    v000 = o
    v100 = o + x
    v010 = o + y
    v001 = o + z
    v110 = o + x + y
    v101 = o + x + z
    v011 = o + y + z
    v111 = o + x + y + z
    pts = np.array([v000, v100, v010, v001, v110, v101, v011, v111])
    # 12 条棱
    edges = [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 4), (2, 6), (3, 5), (3, 6), (4, 7), (5, 7), (6, 7)]
    lines = np.array([[2, e[0], e[1]] for e in edges]).flatten()
    return pv.PolyData(pts, lines=lines)


def obb_normalized_to_world(points_01, obb_origin, obb_x_vec, obb_y_vec, obb_z_vec):
    """将 OBB 归一化空间 [0,1]³ 中的点变换回世界坐标。
    与 renderer_0.py 中 obb_normalized_to_world 保持一致。
    points_01: (N, 3) float, 每行 [u,v,w] 且 u,v,w in [0,1]"""
    pts = np.asarray(points_01, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    return obb_origin + (
        pts[:, 0:1] * obb_x_vec + pts[:, 1:2] * obb_y_vec + pts[:, 2:3] * obb_z_vec
    )


def triangulate_face(face, deflection=0.001):
    """
    Triangulate an occwl Face for visualization using PyVista.
    """
    # Ensure triangulation exists
    BRepMesh_IncrementalMesh(face.topods_shape(), deflection)
    
    loc = TopLoc_Location()
    triangulation = BRep_Tool.Triangulation(face.topods_shape(), loc)
    
    if triangulation is None:
        return None
        
    # Get nodes (vertices)
    nodes = []
    for i in range(1, triangulation.NbNodes() + 1):
        pnt = triangulation.Node(i).Transformed(loc.Transformation())
        nodes.append([pnt.X(), pnt.Y(), pnt.Z()])
    nodes = np.array(nodes)
    
    # Get triangles
    triangles = []
    for i in range(1, triangulation.NbTriangles() + 1):
        tri = triangulation.Triangle(i)
        n1, n2, n3 = tri.Get()
        triangles.append([3, n1 - 1, n2 - 1, n3 - 1])
    
    if not triangles:
        return None
        
    triangles = np.hstack(triangles)
    
    # Create PyVista mesh
    mesh = pv.PolyData(nodes, triangles)
    
    # 计算顶点法线以实现平滑着色
    mesh = mesh.compute_normals(cell_normals=False, point_normals=True)
    
    return mesh


def parse_fallback_txt_samples(txt_path):
    """
    解析 fallback 文件夹中的 txt 原始数据文件。
    格式参考 101_1_49.txt：
    - 忽略 ModelPath、LEFT_POINTS、RIGHT_POINTS
    - SAMPLES_DATA 段每行: x y z o v (归一化坐标, offset真值, validity真值)
    返回: (query_points, offset_gt, validity_gt) 或 None
    """
    if not os.path.exists(txt_path):
        return None
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        return None
    in_samples = False
    query_points = []
    offset_gt = []
    validity_gt = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if "SAMPLES_DATA" in line and "Format" in line:
            in_samples = True
            continue
        if in_samples:
            if "LEFT_POINTS" in line or "RIGHT_POINTS" in line or "ACTUAL_COUNTS" in line:
                break
            parts = line.split()
            if len(parts) >= 5:
                try:
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    o = float(parts[3])
                    v = float(parts[4])
                    query_points.append([x, y, z])
                    offset_gt.append(o)
                    validity_gt.append(v)
                except ValueError:
                    continue
    if not query_points:
        return None
    # txt 中 validity=-1 表示有效，npz 中 validity=1 表示有效，需反转
    validity_gt = np.array(validity_gt)
    validity_gt = -validity_gt
    return (
        np.array(query_points, dtype=np.float64),
        np.array(offset_gt),
        validity_gt,
    )


def parse_highlight_file(filepath):
    """
    解析高亮配置文件。
    格式:
      每行: 一个或多个 face tag (用空格/逗号/制表符分隔)
    
    返回: list_of_face_tag_lists
    例如: [[1, 2], [3, 4], [5]]
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    if not lines:
        raise ValueError("文件为空")
    
    # 每行都是 face tag 列表
    face_tag_groups = []
    for line in lines:
        line = line.strip()
        if not line:
            continue  # 跳过空行
        
        # 支持空格、逗号、制表符分隔
        parts = line.replace(',', ' ').replace('\t', ' ').split()
        tags = []
        for part in parts:
            try:
                tags.append(int(part))
            except ValueError:
                print(f"警告: 无法解析 tag '{part}'，已跳过")
        
        if tags:
            face_tag_groups.append(tags)
    
    if not face_tag_groups:
        raise ValueError("配置文件中未找到有效的 face tag")
    
    return face_tag_groups


class FaceHighlighterWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CAD Face Highlighter (EntityMapper ID)")
        self.setGeometry(100, 100, 1400, 900)
        
        # Data storage
        self.current_solid = None
        self.entity_mapper = None
        self.face_id_to_actor = {}  # face_id -> actor
        self.actor_to_face_id = {}  # actor -> face_id
        self.face_id_to_edge_actor = {}  # face_id -> edge_actor (边界线)
        self.highlight_groups = []  # [(color, [face_ids]), ...]
        self.group_checkboxes = []  # [QCheckBox, ...]
        self.flip_checkboxes = []   # 每行一个“左右反转”checkbox，勾选时交换面对顺序并取反 offset
        
        # 配置文件与面对 (用于 npz 查找)
        self.config_dir = None
        self.config_name = None
        self.face_tag_groups = []  # 从配置文件解析的面对列表
        
        # 点云数据与渲染
        self.point_cloud_actors = []  # 点云相关 actor，用于清除
        self.point_cloud_data = None  # dict: query_points_ws, offset_pred, offset_gt, validity_pred, validity_gt
        self.current_face_pair = None  # (left_id, right_id) 当前加载的面对
        
        # Albedo 映射（三端：起始 / 中间 / 结束）
        self.albedo_color_low = ALBEDO_COLOR_LOW.copy()
        self.albedo_color_mid = ALBEDO_COLOR_MID.copy()
        self.albedo_color_high = ALBEDO_COLOR_HIGH.copy()
        
        # 镜头 target 管理
        self.model_center = None  # 模型几何中心，首次配置加载时设置
        self._prev_solo_value = 0  # 用于检测渲染模式切换（镜头 target 等）
        self.overview_point_cloud_actors = []  # 概览模式下各面对的点云 actor
        
        # Solo 模式(滑条=1)下左右面自定义颜色与透明度
        self.solo_left_color = np.array([0.2, 0.6, 1.0])   # 左面默认蓝
        self.solo_left_opacity = 1.0
        self.solo_right_color = np.array([1.0, 0.4, 0.2])  # 右面默认橙
        self.solo_right_opacity = 1.0
        
        # 点云小球半径缩放 (1.0 = 默认，滑条控制)
        self.point_cloud_radius_scale = 1.0
        self._point_cloud_obb_extent = None  # 当前点云 OBB 尺寸，用于半径重算
        self.point_cloud_opacity = 1.0  # 点云小球透明度 0-1
        
        # 等值面拟合（当前为筛选显示）：offset 数据源、等值、容差、颜色透明度
        self.isosurface_offset_source = "offset_pred"  # "offset_pred" | "offset_gt"
        self.isosurface_offset_value = 0.5
        self.isosurface_tolerance = 0.02
        self.isosurface_color = np.array([0.2, 0.8, 0.4])  # 绿
        self.isosurface_opacity = 1.0
        self.isosurface_filter_active = False  # 筛选显示开关
        self.isosurface_new_behavior = False  # True=新行为(offset!=0.5 时双区间两次操作)，False=旧行为(始终单区间一次操作)
        self.isosurface_ext_v = 0.0  # B-spline 延展比例：每侧扩展量 = (UV范围)*ext_v，仅对 type=bspline 生效
        self.isosurface_surface_actors = []  # 单次拟合的曲面 actor 列表（新行为下可有两个）
        self.batch_isosurface_actors = []  # 批量拟合的曲面 actors
        self.batch_isosurface_meshes = []  # 批量拟合: (surface, grid_points, triangles)，渲染用 mesh，导出用 surface 转 OCC 曲面
        self.batch_fit_done = False  # 批量拟合是否已完成（导出/延展按钮前置）
        
        # 模型边线：世界空间粗细(相对模型尺寸)、颜色，albedo unlit
        self.model_extent = 1.0  # 模型包围盒对角线，用于线粗参考
        self.edge_line_radius_scale = 0.0003  # 边线 tube 半径 = model_extent * this
        self.edge_line_color = np.array([0.0, 0.0, 0.0])  # 黑
        
        # OBB 框线（solo mode）：世界空间粗细、颜色，albedo unlit
        self.obb_line_radius_scale = 0.005
        self.obb_line_color = np.array([0.8, 0.2, 0.2])  # 红
        
        # Model 模式默认颜色与透明度（HIGHLIGHT/STF/Solo 覆盖时不用）
        # 8FA3AF
        self.default_model_color = np.array([0.56, 0.64, 0.69])  # 8FA3AF
        self.default_model_opacity = 0.2  # 非透明面列表中的面用此透明度
        
        # 透明面列表（model mode）：在此列表中的面以透明显示；STF 模式用于维护该列表
        self.transparent_face_ids = set()
        self.transparent_face_opacity = 0.3  # 0=全透明, 1=不透明，透明列表中的面用此
        self.stf_mode = False  # Select Transparent Face 模式
        self.stf_color_normal = np.array([0.7, 0.7, 0.7])    # STF 下非透明列表面的颜色
        self.stf_color_highlight = np.array([0.2, 0.95, 0.5]) # STF 下透明列表面的高亮色
        
        self.init_ui()
        
    def init_ui(self):
        # Main layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()
        main_widget.setLayout(layout)
        
        # Toolbar
        toolbar_layout = QHBoxLayout()
        
        self.btn_load = QPushButton("加载配置文件 (TXT)")
        self.btn_load.clicked.connect(self.load_config_file)
        toolbar_layout.addWidget(self.btn_load)
        
        self.btn_load_step = QPushButton("单独加载 STEP 文件")
        self.btn_load_step.clicked.connect(self.load_step_file_only)
        toolbar_layout.addWidget(self.btn_load_step)
        
        self.btn_clear = QPushButton("清除高亮")
        self.btn_clear.clicked.connect(self.clear_highlights)
        toolbar_layout.addWidget(self.btn_clear)
        
        toolbar_layout.addStretch()
        layout.addLayout(toolbar_layout)
        
        # Info label
        self.lbl_info = QLabel("请加载配置文件 (TXT) 或 STEP 文件")
        self.lbl_info.setStyleSheet("font-size: 14px; font-weight: bold; padding: 5px;")
        layout.addWidget(self.lbl_info)
        
        # Content area: Plotter + Legend
        content_layout = QHBoxLayout()
        
        # PyVista Plotter
        self.plotter = BackgroundPlotter(show=False)
        content_layout.addWidget(self.plotter, stretch=4)
        
        # Legend / Info panel（两列布局，减少纵向高度）
        info_panel = QWidget()
        info_main = QHBoxLayout()
        info_panel.setLayout(info_main)
        info_panel.setMaximumWidth(420)
        
        col_left = QWidget()
        col_left_layout = QVBoxLayout()
        col_left_layout.setAlignment(Qt.AlignTop)
        col_left.setLayout(col_left_layout)
        
        col_right = QWidget()
        col_right_layout = QVBoxLayout()
        col_right_layout.setAlignment(Qt.AlignTop)
        col_right.setLayout(col_right_layout)
        
        # 辅助：往左列添加
        def add_left(*items):
            for w in items:
                if isinstance(w, QHBoxLayout) or isinstance(w, QVBoxLayout):
                    col_left_layout.addLayout(w)
                else:
                    col_left_layout.addWidget(w)
        def add_right(*items):
            for w in items:
                if isinstance(w, QHBoxLayout) or isinstance(w, QVBoxLayout):
                    col_right_layout.addLayout(w)
                else:
                    col_right_layout.addWidget(w)
        
        # ===== 左列 =====
        self.lbl_legend_title = QLabel("高亮图例 (点击切换):")
        self.lbl_legend_title.setStyleSheet("font-size: 14px; font-weight: bold;")
        add_left(self.lbl_legend_title)
        
        select_btn_layout = QHBoxLayout()
        self.btn_select_all = QPushButton("全选")
        self.btn_select_all.clicked.connect(self.select_all_groups)
        select_btn_layout.addWidget(self.btn_select_all)
        self.btn_deselect_all = QPushButton("取消全选")
        self.btn_deselect_all.clicked.connect(self.deselect_all_groups)
        select_btn_layout.addWidget(self.btn_deselect_all)
        self.btn_flip_current_pair = QPushButton("取反当前面对")
        self.btn_flip_current_pair.setToolTip("对当前选中的面对做左右反转并 offset→1-offset（Ctrl+Shift+F）")
        self.btn_flip_current_pair.clicked.connect(self._flip_current_pair)
        select_btn_layout.addWidget(self.btn_flip_current_pair)
        add_left(select_btn_layout)
        self._shortcut_flip_pair = QShortcut(QKeySequence("Ctrl+Shift+F"), self)
        self._shortcut_flip_pair.activated.connect(self._flip_current_pair)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_area.setMaximumHeight(120)
        self.scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout()
        self.scroll_layout.setAlignment(Qt.AlignTop)
        self.scroll_widget.setLayout(self.scroll_layout)
        self.scroll_area.setWidget(self.scroll_widget)
        add_left(self.scroll_area)
        
        add_left(QLabel(""))
        lbl_solo = QLabel("渲染模式:")
        lbl_solo.setStyleSheet("font-size: 12px; font-weight: bold;")
        add_left(lbl_solo)
        solo_layout = QHBoxLayout()
        self.combo_render_mode = QComboBox()
        self.combo_render_mode.addItem("Model (全部)", 0)
        self.combo_render_mode.addItem("Solo (仅当前面)", 1)
        self.combo_render_mode.addItem("概览 (全部面对点云)", 2)
        self.combo_render_mode.setCurrentIndex(0)
        self.combo_render_mode.currentIndexChanged.connect(self._on_render_mode_changed)
        solo_layout.addWidget(self.combo_render_mode)
        add_left(solo_layout)
        self.lbl_solo_hint = QLabel("Model=全部面 | Solo=仅当前面 | 概览=全部面对点云+选中OBB")
        self.lbl_solo_hint.setStyleSheet("font-size: 10px; color: #666;")
        add_left(self.lbl_solo_hint)
        
        add_left(QLabel(""))
        lbl_default = QLabel("Model 默认:")
        lbl_default.setStyleSheet("font-size: 11px; font-weight: bold;")
        add_left(lbl_default)
        default_row = QHBoxLayout()
        self.btn_default_model_color = QPushButton("颜色")
        self._update_default_model_color_button()
        self.btn_default_model_color.clicked.connect(self._pick_default_model_color)
        default_row.addWidget(self.btn_default_model_color)
        default_row.addWidget(QLabel("透明:"))
        self.slider_default_model_opacity = QSlider(Qt.Horizontal)
        self.slider_default_model_opacity.setMinimum(0)
        self.slider_default_model_opacity.setMaximum(100)
        self.slider_default_model_opacity.setValue(100)
        self.slider_default_model_opacity.valueChanged.connect(self._on_default_model_opacity_changed)
        default_row.addWidget(self.slider_default_model_opacity)
        self.lbl_default_model_opacity = QLabel("1.00")
        default_row.addWidget(self.lbl_default_model_opacity)
        add_left(default_row)
        
        add_left(QLabel(""))
        lbl_transp = QLabel("透明面:")
        lbl_transp.setStyleSheet("font-size: 11px; font-weight: bold;")
        add_left(lbl_transp)
        self.btn_stf = QPushButton("选择透明面 (STF)")
        self.btn_stf.setCheckable(True)
        self.btn_stf.setChecked(False)
        self.btn_stf.clicked.connect(self._on_stf_toggled)
        self.btn_stf.setEnabled(True)
        add_left(self.btn_stf)
        stf_color_layout = QHBoxLayout()
        stf_color_layout.addWidget(QLabel("普通:"))
        self.btn_stf_normal_color = QPushButton()
        self.btn_stf_normal_color.setFixedWidth(50)
        self.btn_stf_normal_color.setStyleSheet("background-color: rgb(179,179,179);")
        self.btn_stf_normal_color.clicked.connect(lambda: self._pick_stf_color(is_highlight=False))
        stf_color_layout.addWidget(self.btn_stf_normal_color)
        stf_color_layout.addWidget(QLabel("高亮:"))
        self.btn_stf_highlight_color = QPushButton()
        self.btn_stf_highlight_color.setFixedWidth(50)
        self.btn_stf_highlight_color.setStyleSheet("background-color: rgb(51,242,128);")
        self.btn_stf_highlight_color.clicked.connect(lambda: self._pick_stf_color(is_highlight=True))
        stf_color_layout.addWidget(self.btn_stf_highlight_color)
        add_left(stf_color_layout)
        transp_opacity_layout = QHBoxLayout()
        transp_opacity_layout.addWidget(QLabel("透明:"))
        self.slider_transparent_opacity = QSlider(Qt.Horizontal)
        self.slider_transparent_opacity.setMinimum(0)
        self.slider_transparent_opacity.setMaximum(100)
        self.slider_transparent_opacity.setValue(30)
        self.slider_transparent_opacity.valueChanged.connect(self._on_transparent_opacity_changed)
        transp_opacity_layout.addWidget(self.slider_transparent_opacity)
        self.lbl_transparent_opacity = QLabel("0.30")
        transp_opacity_layout.addWidget(self.lbl_transparent_opacity)
        add_left(transp_opacity_layout)
        
        solo_color_layout = QVBoxLayout()
        lbl_solo_colors = QLabel("Solo 下面颜色:")
        lbl_solo_colors.setStyleSheet("font-size: 11px; font-weight: bold;")
        solo_color_layout.addWidget(lbl_solo_colors)
        left_row = QHBoxLayout()
        left_row.addWidget(QLabel("左:"))
        self.btn_solo_left_color = QPushButton("颜色")
        self.btn_solo_left_color.setStyleSheet("background-color: rgb(51,153,255); color: white;")
        self.btn_solo_left_color.clicked.connect(lambda: self._pick_solo_color(is_left=True))
        left_row.addWidget(self.btn_solo_left_color)
        left_row.addWidget(QLabel("透明度:"))
        self.spin_solo_left_opacity = QDoubleSpinBox()
        self.spin_solo_left_opacity.setRange(0.0, 1.0)
        self.spin_solo_left_opacity.setSingleStep(0.1)
        self.spin_solo_left_opacity.setValue(1.0)
        self.spin_solo_left_opacity.valueChanged.connect(lambda v: self._on_solo_opacity_changed(True, v))
        left_row.addWidget(self.spin_solo_left_opacity)
        solo_color_layout.addLayout(left_row)
        right_row = QHBoxLayout()
        right_row.addWidget(QLabel("右:"))
        self.btn_solo_right_color = QPushButton("颜色")
        self.btn_solo_right_color.setStyleSheet("background-color: rgb(255,102,51); color: white;")
        self.btn_solo_right_color.clicked.connect(lambda: self._pick_solo_color(is_left=False))
        right_row.addWidget(self.btn_solo_right_color)
        right_row.addWidget(QLabel("透明度:"))
        self.spin_solo_right_opacity = QDoubleSpinBox()
        self.spin_solo_right_opacity.setRange(0.0, 1.0)
        self.spin_solo_right_opacity.setSingleStep(0.1)
        self.spin_solo_right_opacity.setValue(1.0)
        self.spin_solo_right_opacity.valueChanged.connect(lambda v: self._on_solo_opacity_changed(False, v))
        right_row.addWidget(self.spin_solo_right_opacity)
        solo_color_layout.addLayout(right_row)
        add_left(solo_color_layout)
        
        # ===== 右列 =====
        add_right(QLabel(""))
        lbl_albedo = QLabel("点云 Albedo:")
        lbl_albedo.setStyleSheet("font-size: 12px; font-weight: bold;")
        add_right(lbl_albedo)
        
        albedo_combo_layout = QHBoxLayout()
        albedo_combo_layout.addWidget(QLabel("数据源:"))
        self.combo_albedo_source = QComboBox()
        for key, label in ALBEDO_DATA_SOURCES:
            self.combo_albedo_source.addItem(label, key)
        self.combo_albedo_source.currentIndexChanged.connect(self._on_albedo_source_changed)
        albedo_combo_layout.addWidget(self.combo_albedo_source)
        add_right(albedo_combo_layout)
        
        # 颜色条：起始色 | [渐变色条] | 中间色 | [渐变色条] | 结束色（三端）
        color_bar_layout = QHBoxLayout()
        self.btn_color_low = QPushButton("起始")
        self._update_albedo_button_style("low")
        self.btn_color_low.clicked.connect(lambda: self._pick_albedo_color(endpoint="low"))
        self.btn_color_mid = QPushButton("中间")
        self._update_albedo_button_style("mid")
        self.btn_color_mid.clicked.connect(lambda: self._pick_albedo_color(endpoint="mid"))
        self.btn_color_high = QPushButton("结束")
        self._update_albedo_button_style("high")
        self.btn_color_high.clicked.connect(lambda: self._pick_albedo_color(endpoint="high"))
        color_bar_layout.addWidget(self.btn_color_low)
        self.color_bar_preview = QFrame()
        self.color_bar_preview.setFixedHeight(24)
        color_bar_layout.addWidget(self.color_bar_preview, stretch=1)
        color_bar_layout.addWidget(self.btn_color_mid)
        color_bar_layout.addWidget(self.btn_color_high)
        self._update_color_bar_preview()
        add_right(color_bar_layout)
        
        add_right(QLabel(""))
        lbl_edges = QLabel("模型边线:")
        lbl_edges.setStyleSheet("font-size: 11px; font-weight: bold;")
        add_right(lbl_edges)
        self.cb_show_edges = QCheckBox("显示边线")
        self.cb_show_edges.setChecked(True)
        self.cb_show_edges.toggled.connect(self._on_show_edges_changed)
        add_right(self.cb_show_edges)
        edge_row = QHBoxLayout()
        self.btn_edge_color = QPushButton("颜色")
        self.btn_edge_color.setStyleSheet("background-color: black; color: white;")
        self.btn_edge_color.clicked.connect(self._pick_edge_color)
        edge_row.addWidget(self.btn_edge_color)
        edge_row.addWidget(QLabel("粗细:"))
        self.spin_edge_radius = QDoubleSpinBox()
        self.spin_edge_radius.setRange(0.00001, 0.01)
        self.spin_edge_radius.setSingleStep(0.0001)
        self.spin_edge_radius.setValue(0.0003)
        self.spin_edge_radius.setDecimals(5)
        self.spin_edge_radius.valueChanged.connect(self._on_edge_radius_changed)
        edge_row.addWidget(self.spin_edge_radius)
        add_right(edge_row)
        
        add_right(QLabel(""))
        lbl_obb = QLabel("OBB 框线:")
        lbl_obb.setStyleSheet("font-size: 11px; font-weight: bold;")
        add_right(lbl_obb)
        self.cb_show_obb = QCheckBox("显示 OBB 框线")
        self.cb_show_obb.setChecked(True)
        self.cb_show_obb.toggled.connect(self._on_show_obb_changed)
        add_right(self.cb_show_obb)
        obb_row = QHBoxLayout()
        self.btn_obb_color = QPushButton("颜色")
        self.btn_obb_color.setStyleSheet("background-color: rgb(204,51,51); color: white;")
        self.btn_obb_color.clicked.connect(self._pick_obb_color)
        obb_row.addWidget(self.btn_obb_color)
        obb_row.addWidget(QLabel("粗细:"))
        self.spin_obb_radius = QDoubleSpinBox()
        self.spin_obb_radius.setRange(0.00001, 0.01)
        self.spin_obb_radius.setSingleStep(0.0001)
        self.spin_obb_radius.setValue(0.005)
        self.spin_obb_radius.setDecimals(5)
        self.spin_obb_radius.valueChanged.connect(self._on_obb_radius_changed)
        obb_row.addWidget(self.spin_obb_radius)
        add_right(obb_row)
        
        add_right(QLabel(""))
        lbl_radius = QLabel("点云半径:")
        lbl_radius.setStyleSheet("font-size: 11px; font-weight: bold;")
        add_right(lbl_radius)
        radius_layout = QHBoxLayout()
        radius_layout.addWidget(QLabel("0.1×"))
        self.slider_point_radius = QSlider(Qt.Horizontal)
        self.slider_point_radius.setMinimum(10)
        self.slider_point_radius.setMaximum(300)
        self.slider_point_radius.setValue(100)
        self.slider_point_radius.setTickPosition(QSlider.TicksBelow)
        self.slider_point_radius.setTickInterval(50)
        self.slider_point_radius.valueChanged.connect(self._on_point_radius_changed)
        radius_layout.addWidget(self.slider_point_radius)
        radius_layout.addWidget(QLabel("3×"))
        self.lbl_radius_value = QLabel("1.0×")
        radius_layout.addWidget(self.lbl_radius_value)
        add_right(radius_layout)
        point_opacity_layout = QHBoxLayout()
        point_opacity_layout.addWidget(QLabel("透明:"))
        self.slider_point_opacity = QSlider(Qt.Horizontal)
        self.slider_point_opacity.setMinimum(0)
        self.slider_point_opacity.setMaximum(100)
        self.slider_point_opacity.setValue(100)
        self.slider_point_opacity.valueChanged.connect(self._on_point_cloud_opacity_changed)
        point_opacity_layout.addWidget(self.slider_point_opacity)
        self.lbl_point_opacity = QLabel("1.00")
        point_opacity_layout.addWidget(self.lbl_point_opacity)
        add_right(point_opacity_layout)
        
        add_right(QLabel(""))
        lbl_isosurf = QLabel("等值面拟合:")
        lbl_isosurf.setStyleSheet("font-size: 12px; font-weight: bold;")
        add_right(lbl_isosurf)
        iso_source_layout = QHBoxLayout()
        iso_source_layout.addWidget(QLabel("Offset:"))
        self.combo_isosurface_source = QComboBox()
        self.combo_isosurface_source.addItem("预测值", "offset_pred")
        self.combo_isosurface_source.addItem("真值", "offset_gt")
        self.combo_isosurface_source.currentIndexChanged.connect(self._on_isosurface_source_changed)
        iso_source_layout.addWidget(self.combo_isosurface_source)
        add_right(iso_source_layout)
        iso_offset_layout = QHBoxLayout()
        iso_offset_layout.addWidget(QLabel("等值:"))
        self.spin_isosurface_offset = QDoubleSpinBox()
        self.spin_isosurface_offset.setRange(-10.0, 10.0)
        self.spin_isosurface_offset.setValue(0.5)
        self.spin_isosurface_offset.setSingleStep(0.01)
        self.spin_isosurface_offset.setDecimals(4)
        self.spin_isosurface_offset.valueChanged.connect(self._on_isosurface_offset_changed)
        iso_offset_layout.addWidget(self.spin_isosurface_offset)
        add_right(iso_offset_layout)
        iso_tol_layout = QHBoxLayout()
        iso_tol_layout.addWidget(QLabel("Tol:"))
        self.spin_isosurface_tolerance = QDoubleSpinBox()
        self.spin_isosurface_tolerance.setRange(0.0, 10.0)
        self.spin_isosurface_tolerance.setValue(0.02)
        self.spin_isosurface_tolerance.setSingleStep(0.01)
        self.spin_isosurface_tolerance.setDecimals(4)
        self.spin_isosurface_tolerance.valueChanged.connect(self._on_isosurface_tolerance_changed)
        iso_tol_layout.addWidget(self.spin_isosurface_tolerance)
        add_right(iso_tol_layout)
        iso_color_layout = QHBoxLayout()
        self.btn_isosurface_color = QPushButton("颜色")
        self._update_isosurface_color_button()
        self.btn_isosurface_color.clicked.connect(self._pick_isosurface_color)
        iso_color_layout.addWidget(self.btn_isosurface_color)
        iso_color_layout.addWidget(QLabel("透明:"))
        self.slider_isosurface_opacity = QSlider(Qt.Horizontal)
        self.slider_isosurface_opacity.setMinimum(0)
        self.slider_isosurface_opacity.setMaximum(100)
        self.slider_isosurface_opacity.setValue(100)
        self.slider_isosurface_opacity.valueChanged.connect(self._on_isosurface_opacity_changed)
        iso_color_layout.addWidget(self.slider_isosurface_opacity)
        self.lbl_isosurface_opacity = QLabel("1.00")
        iso_color_layout.addWidget(self.lbl_isosurface_opacity)
        add_right(iso_color_layout)
        iso_btn_layout = QHBoxLayout()
        self.chk_isosurface_new_behavior = QCheckBox("新行为(双区间)")
        self.chk_isosurface_new_behavior.setChecked(False)
        self.chk_isosurface_new_behavior.stateChanged.connect(self._on_isosurface_new_behavior_changed)
        iso_btn_layout.addWidget(self.chk_isosurface_new_behavior)
        self.btn_isosurface_filter = QPushButton("筛选")
        self.btn_isosurface_filter.setCheckable(True)
        self.btn_isosurface_filter.setChecked(False)
        self.btn_isosurface_filter.clicked.connect(self._on_isosurface_filter_clicked)
        iso_btn_layout.addWidget(self.btn_isosurface_filter)
        self.btn_isosurface_fit = QPushButton("拟合")
        self.btn_isosurface_fit.clicked.connect(self._on_isosurface_fit_clicked)
        iso_btn_layout.addWidget(self.btn_isosurface_fit)
        add_right(iso_btn_layout)
        self.btn_batch_fit = QPushButton("批量拟合")
        self.btn_batch_fit.clicked.connect(self._batch_fit_all_isosurfaces)
        add_right(self.btn_batch_fit)
        self.btn_export_midsurf = QPushButton("导出 midsurf.step")
        self.btn_export_midsurf.setEnabled(False)
        self.btn_export_midsurf.clicked.connect(self._export_midsurf_step)
        add_right(self.btn_export_midsurf)
        # B-spline 延展：extV 滑条 (0~1) + 延展按钮（仅批量拟合后可按）
        iso_ext_layout = QHBoxLayout()
        iso_ext_layout.addWidget(QLabel("extV:"))
        self.slider_isosurface_ext_v = QSlider(Qt.Horizontal)
        self.slider_isosurface_ext_v.setMinimum(0)
        self.slider_isosurface_ext_v.setMaximum(50)
        self.slider_isosurface_ext_v.setValue(0)
        self.slider_isosurface_ext_v.valueChanged.connect(self._on_isosurface_ext_v_changed)
        iso_ext_layout.addWidget(self.slider_isosurface_ext_v)
        self.lbl_isosurface_ext_v = QLabel("0.00")
        iso_ext_layout.addWidget(self.lbl_isosurface_ext_v)
        self.btn_isosurface_extend = QPushButton("延展")
        self.btn_isosurface_extend.setEnabled(False)
        self.btn_isosurface_extend.clicked.connect(self._on_isosurface_extend_clicked)
        iso_ext_layout.addWidget(self.btn_isosurface_extend)
        add_right(iso_ext_layout)
        info_main.addWidget(col_left)
        info_main.addWidget(col_right)
        
        info_scroll = QScrollArea()
        info_scroll.setWidgetResizable(True)
        info_scroll.setFrameShape(QFrame.NoFrame)
        info_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        info_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        info_scroll.setWidget(info_panel)
        content_layout.addWidget(info_scroll, stretch=1)
        layout.addLayout(content_layout)
        
        # 启用点击事件显示 face id
        self.plotter.iren.add_observer("LeftButtonPressEvent", self.on_left_click)

    def load_config_file(self):
        """加载 TXT 配置文件"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "打开配置文件", "", "Text Files (*.txt);;All Files (*)"
        )
        if not filename:
            return
        
        try:
            face_tag_groups = parse_highlight_file(filename)
            config_dir = os.path.dirname(filename)
            config_name = os.path.splitext(os.path.basename(filename))[0]
            step_file_path = os.path.join(config_dir, f"{config_name}.step")
            
            # 保存配置文件信息，供点击面对时查找 npz
            self.config_dir = config_dir
            self.config_name = config_name
            self.face_tag_groups = face_tag_groups
            self._remove_batch_isosurfaces()
            
            if not os.path.exists(step_file_path):
                self.lbl_info.setText(
                    f"错误: 未找到同名 STEP 文件: {step_file_path}"
                )
                return
            
            self.lbl_info.setText(f"正在加载 {os.path.basename(step_file_path)}...")
            QApplication.processEvents()
            
            # 加载 STEP 文件
            self.load_step_file(step_file_path)
            
            # 应用高亮
            self.apply_highlights(face_tag_groups)
            
            # 首次通过配置文件加载：设置模型几何中心为镜头 target
            self._set_camera_target(self.model_center)
            
            self.lbl_info.setText(
                f"已加载: {os.path.basename(step_file_path)} | "
                f"面数: {len(list(self.current_solid.faces()))} | "
                f"高亮组数: {len(face_tag_groups)} | 点击面可加载点云"
            )
            
        except Exception as e:
            self.lbl_info.setText(f"加载配置文件出错: {str(e)}")
            import traceback
            traceback.print_exc()

    def load_step_file_only(self):
        """单独加载 STEP 文件（不应用高亮）"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "打开 STEP 文件", "", "STEP Files (*.stp *.step)"
        )
        if not filename:
            return
        
        self.config_dir = None
        self.config_name = None
        self.face_tag_groups = []
        self.load_step_file(filename)
        self.clear_highlights()
        self.lbl_info.setText(
            f"已加载: {os.path.basename(filename)} | "
            f"面数: {len(list(self.current_solid.faces()))}"
        )

    def load_step_file(self, filepath):
        """加载 STEP 文件"""
        # Load solids using occwl
        loaded_obj = Compound.load_from_step(filepath)
        
        # Handle return type
        if isinstance(loaded_obj, Compound):
            solids = [loaded_obj]
        else:
            solids = list(loaded_obj)
        
        if not solids:
            raise ValueError("文件中未找到 solid")
        
        # 获取第一个有效的 solid
        self.current_solid = None
        for s in solids:
            if isinstance(s, Solid):
                self.current_solid = s
                break
            elif isinstance(s, Compound):
                sub_solids = list(s.solids())
                if sub_solids:
                    self.current_solid = sub_solids[0]
                    break
        
        if self.current_solid is None:
            raise ValueError("未找到有效的 Solid")
        
        # 初始化 EntityMapper (与 cad_picker_2 保持一致)
        self.entity_mapper = EntityMapper(self.current_solid)
        
        # 计算并存储模型几何中心（用于镜头 target 管理）
        try:
            self.model_center = compute_solid_center(self.current_solid.topods_shape())
        except Exception as e:
            print(f"计算模型中心失败: {e}")
            self.model_center = None
        
        # 可视化
        self.visualize_solid()

    def _set_camera_target(self, center):
        """设置镜头焦点到给定中心点"""
        if center is None:
            return
        c = np.asarray(center)
        self.plotter.camera.focal_point = tuple(c.tolist())
        self.plotter.render()

    def visualize_solid(self):
        """可视化 solid 的所有面"""
        self.plotter.clear()
        self.point_cloud_actors = []
        self.point_cloud_data = None
        self.current_face_pair = None
        self.overview_point_cloud_actors = []
        self.plotter.enable_lightkit()
        self.face_id_to_actor = {}
        self.actor_to_face_id = {}
        self.face_id_to_edge_actor = {}
        self.model_edge_actor = None
        self.solo_edge_actor = None
        self.obb_box_actor = None
        self.index_to_face = {}
        self.transparent_face_ids.clear()
        
        faces = list(self.current_solid.faces())
        print(f"可视化 {len(faces)} 个面...")
        
        edge_polydatas = []
        for face in faces:
            face_id = self.entity_mapper.face_index(face)
            self.index_to_face[face_id] = face
            mesh = triangulate_face(face)
            if mesh:
                actor = self.plotter.add_mesh(
                    mesh,
                    color="lightgrey",
                    show_edges=False,
                    pickable=True,
                    diffuse=0.8,
                    specular=0.6,
                    specular_power=30.0,
                    ambient=0.15,
                    smooth_shading=True
                )
                self.face_id_to_actor[face_id] = actor
                self.actor_to_face_id[actor] = face_id
                try:
                    edges = mesh.extract_feature_edges(
                        boundary_edges=True,
                        non_manifold_edges=False,
                        feature_edges=False,
                        manifold_edges=False
                    )
                    if edges.n_points > 0 and edges.n_cells > 0:
                        edge_polydatas.append(edges)
                except Exception as e:
                    print(f"警告: 提取面 {face_id} 的边界线失败: {e}")
        
        try:
            bbox = Bnd_Box()
            brepbndlib_Add(self.current_solid.topods_shape(), bbox)
            xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
            self.model_extent = np.sqrt((xmax-xmin)**2 + (ymax-ymin)**2 + (zmax-zmin)**2)
            if self.model_extent <= 0:
                self.model_extent = 1.0
        except Exception:
            self.model_extent = 1.0
        
        merged_edges = _merge_deduplicate_edges(edge_polydatas)
        if merged_edges is not None:
            r = self.model_extent * self.edge_line_radius_scale
            tube_mesh = _lines_to_tube_polydata(merged_edges, r)
            if tube_mesh is not None:
                self.model_edge_actor = self.plotter.add_mesh(
                    tube_mesh,
                    color=tuple(self.edge_line_color.tolist()),
                    lighting=False,
                    pickable=False
                )
        
        self._apply_solo_mode()
        self.plotter.reset_camera()

    def _on_render_mode_changed(self, index):
        """渲染模式(0=Model/1=Solo/2=概览)变化时更新显示与镜头 target"""
        prev = self._prev_solo_value
        self._prev_solo_value = index
        value = index

        # 概览模式：进入时创建全部面对点云，离开时清除
        if value == 2:
            self._clear_point_cloud_actors_only()
            self._create_overview_point_clouds()
        elif prev == 2:
            self._clear_overview_point_clouds()

        # 镜头 target 管理
        if prev == 0 and value == 1:
            if self.current_face_pair and self.config_dir and self.config_name:
                step_path = os.path.join(self.config_dir, f"{self.config_name}.step")
                try:
                    left_id, right_id = self.current_face_pair
                    obb_origin, obb_x, obb_y, obb_z = compute_obb_for_faces(step_path, left_id, right_id)
                    obb_center = get_obb_center(obb_origin, obb_x, obb_y, obb_z)
                    self._set_camera_target(obb_center)
                except Exception as e:
                    print(f"设置 OBB target 失败: {e}")
        elif prev == 1 and value == 0:
            self._set_camera_target(self.model_center)
        if value == 1:
            self.stf_mode = False
            if hasattr(self, "btn_stf"):
                self.btn_stf.setChecked(False)
                self.btn_stf.setEnabled(False)
        elif value == 0 and hasattr(self, "btn_stf"):
            self.btn_stf.setEnabled(True)
        self._apply_solo_mode()

    def _apply_solo_mode(self):
        """
        根据渲染模式(0=Model / 1=Solo / 2=概览)和当前选中的面对，控制面的显示/隐藏及颜色。
        - 0: 显示所有面，model 边线，OBB 随当前面对
        - 1: 仅显示当前左右面，solo 边线，OBB 框
        - 2: 显示所有面 + 全部面对点云，model 边线，仅当前面对 OBB
        """
        solo = self.combo_render_mode.currentIndex() if hasattr(self, "combo_render_mode") else 0
        show_ids = None
        if solo == 1 and self.current_face_pair:
            left_id, right_id = self.current_face_pair
            show_ids = {left_id, right_id}

        def _get_normal_color(fid):
            for i, (color, tags) in enumerate(self.highlight_groups):
                if fid in tags:
                    cb = self.group_checkboxes[i] if i < len(self.group_checkboxes) else None
                    if cb and cb.isChecked():
                        return color
            return None  # 使用 default_model_color

        stf_mode = getattr(self, "stf_mode", False)
        transp_opacity = getattr(self, "transparent_face_opacity", 0.3)
        default_opacity = getattr(self, "default_model_opacity", 1.0)
        default_color = getattr(self, "default_model_color", np.array([0.83, 0.83, 0.83]))
        for face_id, actor in self.face_id_to_actor.items():
            visible = (show_ids is None) or (face_id in show_ids)
            actor.SetVisibility(visible)
            if solo == 1 and show_ids and face_id in show_ids:
                left_id, right_id = self.current_face_pair
                if face_id == left_id:
                    c, op = self.solo_left_color, self.solo_left_opacity
                else:
                    c, op = self.solo_right_color, self.solo_right_opacity
                actor.prop.color = (c[0], c[1], c[2])
                actor.prop.opacity = op
            else:
                if show_ids is None and stf_mode:
                    if face_id in self.transparent_face_ids:
                        c = self.stf_color_highlight
                    else:
                        c = self.stf_color_normal
                    actor.prop.color = tuple(c.tolist())
                    actor.prop.opacity = 1.0
                else:
                    h_color = _get_normal_color(face_id)
                    if h_color is not None:
                        actor.prop.color = h_color
                    else:
                        actor.prop.color = tuple(default_color.tolist())
                    if show_ids is None:
                        if face_id in self.transparent_face_ids:
                            actor.prop.opacity = transp_opacity
                        else:
                            actor.prop.opacity = default_opacity
                    else:
                        actor.prop.opacity = 1.0

        show_edges = self.cb_show_edges.isChecked() if hasattr(self, "cb_show_edges") else True
        if self.model_edge_actor is not None:
            self.model_edge_actor.SetVisibility(show_edges and (show_ids is None))
        if self.solo_edge_actor is not None:
            self.solo_edge_actor.SetVisibility(show_edges and (show_ids is not None))

        if show_ids is not None:
            self._ensure_solo_edges_and_obb()
        else:
            if self.solo_edge_actor is not None:
                try:
                    self.plotter.remove_actor(self.solo_edge_actor)
                except Exception:
                    pass
                self.solo_edge_actor = None
            if not self.current_face_pair and self.obb_box_actor is not None:
                try:
                    self.plotter.remove_actor(self.obb_box_actor)
                except Exception:
                    pass
                self.obb_box_actor = None
        show_obb = self.cb_show_obb.isChecked() if hasattr(self, "cb_show_obb") else True
        if self.current_face_pair:
            self._ensure_obb_box()
            if self.obb_box_actor is not None:
                self.obb_box_actor.SetVisibility(show_obb)

        self.plotter.render()

    def _ensure_obb_box(self):
        """确保当前面对的 OBB 框已创建，model/solo 模式均可用，由复选框控制显示"""
        if not self.current_face_pair or not self.config_dir or not self.config_name:
            return
        left_id, right_id = self.current_face_pair
        pair_key = (left_id, right_id)
        need_recreate = getattr(self, "_obb_pair_key", None) != pair_key
        if need_recreate and self.obb_box_actor is not None:
            try:
                self.plotter.remove_actor(self.obb_box_actor)
            except Exception:
                pass
            self.obb_box_actor = None
        if self.obb_box_actor is None:
            step_path = os.path.join(self.config_dir, f"{self.config_name}.step")
            try:
                obb_o, obb_x, obb_y, obb_z = compute_obb_for_faces(step_path, left_id, right_id)
                obb_extent = np.linalg.norm(obb_x) + np.linalg.norm(obb_y) + np.linalg.norm(obb_z)
                r = obb_extent * self.obb_line_radius_scale
                obb_lines = _create_obb_wireframe_polydata(obb_o, obb_x, obb_y, obb_z)
                obb_tube = _lines_to_tube_polydata(obb_lines, r)
                if obb_tube is not None:
                    self.obb_box_actor = self.plotter.add_mesh(
                        obb_tube,
                        color=tuple(self.obb_line_color.tolist()),
                        lighting=False,
                        pickable=False
                    )
            except Exception as e:
                print(f"OBB 框线创建失败: {e}")
        self._obb_pair_key = pair_key

    def _ensure_solo_edges_and_obb(self):
        """Solo 模式下确保 solo 边线和 OBB 框已创建"""
        if not self.current_face_pair or not self.config_dir or not self.config_name:
            return
        left_id, right_id = self.current_face_pair
        if left_id not in self.index_to_face or right_id not in self.index_to_face:
            return

        pair_key = (left_id, right_id)
        need_recreate = getattr(self, "_solo_pair_key", None) != pair_key
        if need_recreate and self.solo_edge_actor is not None:
            try:
                self.plotter.remove_actor(self.solo_edge_actor)
            except Exception:
                pass
            self.solo_edge_actor = None
        if need_recreate and self.obb_box_actor is not None:
            try:
                self.plotter.remove_actor(self.obb_box_actor)
            except Exception:
                pass
            self.obb_box_actor = None

        if self.solo_edge_actor is None:
            edge_polydatas = []
            for fid in (left_id, right_id):
                face = self.index_to_face[fid]
                mesh = triangulate_face(face)
                if mesh:
                    try:
                        edges = mesh.extract_feature_edges(
                            boundary_edges=True,
                            non_manifold_edges=False,
                            feature_edges=False,
                            manifold_edges=False
                        )
                        if edges.n_points > 0 and edges.n_cells > 0:
                            edge_polydatas.append(edges)
                    except Exception:
                        pass
            merged = _merge_deduplicate_edges(edge_polydatas) if edge_polydatas else None
            if merged is not None:
                r = self.model_extent * self.edge_line_radius_scale
                tube_mesh = _lines_to_tube_polydata(merged, r)
                if tube_mesh is not None:
                    self.solo_edge_actor = self.plotter.add_mesh(
                        tube_mesh,
                        color=tuple(self.edge_line_color.tolist()),
                        lighting=False,
                        pickable=False
                    )

        self._ensure_obb_box()

        self._solo_pair_key = pair_key
        show_edges = self.cb_show_edges.isChecked() if hasattr(self, "cb_show_edges") else True
        if self.solo_edge_actor is not None:
            self.solo_edge_actor.SetVisibility(show_edges)

    def apply_highlights(self, face_tag_groups):
        """应用高亮颜色到指定的面，并创建可交互的 checkbox 列表；每行增加“左右反转”checkbox"""
        self.highlight_groups = []
        self.group_checkboxes = []
        self.flip_checkboxes = []

        # 清除旧的 checkbox
        while self.scroll_layout.count():
            item = self.scroll_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        for i, tags in enumerate(face_tag_groups):
            color = HIGHLIGHT_COLORS[i % len(HIGHLIGHT_COLORS)]
            valid_tags = []
            for tag in tags:
                if tag in self.face_id_to_actor:
                    actor = self.face_id_to_actor[tag]
                    actor.prop.color = color
                    valid_tags.append(tag)
                else:
                    print(f"警告: Face tag {tag} 不存在")

            if valid_tags:
                group_index = len(self.highlight_groups)
                self.highlight_groups.append((color, valid_tags))
                tags_str = ", ".join(map(str, valid_tags))
                cb = QCheckBox(f"第 {i+1} 组: [{tags_str}]")
                cb.setChecked(True)
                cb.setStyleSheet(
                    f"QCheckBox {{ font-size: 12px; padding: 4px; }}"
                    f"QCheckBox::indicator:checked {{ background-color: {color}; border: 1px solid #333; }}"
                    f"QCheckBox::indicator:unchecked {{ background-color: #d0d0d0; border: 1px solid #999; }}"
                    f"QCheckBox::indicator {{ width: 14px; height: 14px; }}"
                )
                cb.toggled.connect(lambda checked, idx=group_index: self.on_group_toggled(idx, checked))

                flip_cb = QCheckBox("反")
                flip_cb.setChecked(False)
                flip_cb.setToolTip("勾选：反转该面对左右顺序，并对当前点云 offset 做 1-offset")
                flip_cb.stateChanged.connect(lambda state, idx=group_index: self._on_flip_pair_toggled(idx, state == Qt.Checked))

                row_widget = QWidget()
                row_layout = QHBoxLayout()
                row_layout.setContentsMargins(0, 2, 0, 2)
                row_layout.addWidget(cb)
                row_layout.addWidget(flip_cb)
                row_layout.addStretch()
                row_widget.setLayout(row_layout)
                self.scroll_layout.addWidget(row_widget)
                self.group_checkboxes.append(cb)
                self.flip_checkboxes.append(flip_cb)
        self.plotter.render()

    def _get_row_index_for_pair(self, left_id, right_id):
        """返回 face_tag_groups 中与 (left_id, right_id) 或 (right_id, left_id) 匹配的行索引，无则返回 None"""
        if not getattr(self, "face_tag_groups", None):
            return None
        for i, tags in enumerate(self.face_tag_groups):
            if len(tags) != 2:
                continue
            if (int(tags[0]), int(tags[1])) == (int(left_id), int(right_id)) or (
                int(tags[0]), int(tags[1])
            ) == (int(right_id), int(left_id)):
                return i
        return None

    def _flip_current_pair(self):
        """对当前选中的面对执行取反：找到对应行的“反”checkbox 并切换，从而触发左右交换与 offset 取反"""
        if not self.current_face_pair or not getattr(self, "face_tag_groups", None):
            if hasattr(self, "lbl_info"):
                self.lbl_info.setText("请先选中一个面对（点击面加载点云）")
            return
        row_i = self._get_row_index_for_pair(self.current_face_pair[0], self.current_face_pair[1])
        if row_i is None or row_i >= len(getattr(self, "flip_checkboxes", [])):
            if hasattr(self, "lbl_info"):
                self.lbl_info.setText("当前面对不在高亮列表中")
            return
        flip_cb = self.flip_checkboxes[row_i]
        flip_cb.setChecked(not flip_cb.isChecked())
        if hasattr(self, "lbl_info"):
            self.lbl_info.setText("已取反当前面对（左右顺序与 offset 已更新）")

    def _on_flip_pair_toggled(self, row_index, checked):
        """每行“反”checkbox 勾选时：1）反转该面对的左右 idx；2）若当前点云是该面对则对 offset 做 1-offset"""
        if row_index >= len(self.face_tag_groups) or row_index >= len(self.highlight_groups):
            return
        tags = list(self.face_tag_groups[row_index])
        new_tags = [tags[1], tags[0]]
        self.face_tag_groups[row_index] = new_tags
        color, valid_tags = self.highlight_groups[row_index]
        new_valid = [valid_tags[1], valid_tags[0]]
        self.highlight_groups[row_index] = (color, new_valid)
        if row_index < len(self.group_checkboxes):
            self.group_checkboxes[row_index].setText(
                f"第 {row_index+1} 组: [{new_tags[0]}, {new_tags[1]}]"
            )
        # 若当前加载的点云正是该面对，则对 offset 取反并刷新显示
        if self.current_face_pair and self.point_cloud_data:
            a, b = tags[0], tags[1]
            if (self.current_face_pair[0], self.current_face_pair[1]) == (a, b) or (
                self.current_face_pair[0], self.current_face_pair[1]
            ) == (b, a):
                for key in ("offset_pred", "offset_gt"):
                    if key in self.point_cloud_data:
                        arr = np.asarray(self.point_cloud_data[key], dtype=np.float64)
                        self.point_cloud_data[key] = 1.0 - arr
                self.current_face_pair = tuple(new_tags)
                self._refresh_point_cloud_display()
                if self.isosurface_surface_actors or self.batch_fit_done:
                    self.plotter.render()
        # 概览模式：取反该面对时，仅对该面对的代表性点云实时重新着色
        if getattr(self, "combo_render_mode", None) is not None and self.combo_render_mode.currentIndex() == 2:
            self._recolor_overview_pair(row_index)

    def on_group_toggled(self, group_index, checked):
        """当某一组的 checkbox 状态改变时，切换该组的高亮显示"""
        if group_index >= len(self.highlight_groups):
            return
        
        color, tags = self.highlight_groups[group_index]
        
        for tag in tags:
            if tag in self.face_id_to_actor:
                actor = self.face_id_to_actor[tag]
                if checked:
                    actor.prop.color = color
                else:
                    actor.prop.color = "lightgrey"
        
        self.plotter.render()

    def select_all_groups(self):
        """全选所有高亮组"""
        for cb in self.group_checkboxes:
            cb.setChecked(True)

    def deselect_all_groups(self):
        """取消全选所有高亮组"""
        for cb in self.group_checkboxes:
            cb.setChecked(False)

    def clear_highlights(self):
        """清除所有高亮，恢复默认颜色"""
        for face_id, actor in self.face_id_to_actor.items():
            actor.prop.color = "lightgrey"

        self.highlight_groups = []
        self.group_checkboxes = []
        self.flip_checkboxes = []
        self._clear_point_cloud()
        self._remove_batch_isosurfaces()

        # 清除 checkbox 列表
        while self.scroll_layout.count():
            item = self.scroll_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        
        self.plotter.render()
        self.lbl_info.setText("已清除所有高亮")

    def _clear_point_cloud(self):
        """移除点云渲染"""
        for actor in self.point_cloud_actors:
            try:
                self.plotter.remove_actor(actor)
            except Exception:
                pass
        self.point_cloud_actors = []
        self.point_cloud_data = None
        self.current_face_pair = None
        self._obb_pair_key = None
        self.isosurface_filter_active = False
        if hasattr(self, "btn_isosurface_filter"):
            self.btn_isosurface_filter.setChecked(False)
        self._remove_isosurface_surface()
        self._apply_solo_mode()  # 清除后面无选中面对，恢复全部显示

    def _update_albedo_button_style(self, endpoint):
        """更新指定端点的 Albedo 颜色按钮样式"""
        col = getattr(self, f"albedo_color_{endpoint}")
        btn = getattr(self, f"btn_color_{endpoint}")
        r, g, b = int(col[0]*255), int(col[1]*255), int(col[2]*255)
        fg = "white" if (r + g + b) / 3 < 128 else "black"
        btn.setStyleSheet(f"background-color: rgb({r},{g},{b}); color: {fg};")

    def _pick_albedo_color(self, endpoint):
        """选择 Albedo 颜色条端点颜色（endpoint: low / mid / high）"""
        col = getattr(self, f"albedo_color_{endpoint}")
        qcol = QColorDialog.getColor(
            QColor(int(col[0]*255), int(col[1]*255), int(col[2]*255)),
            self, {"low": "选择起始色", "mid": "选择中间色", "high": "选择结束色"}[endpoint]
        )
        if qcol.isValid():
            rgb = np.array([qcol.red()/255, qcol.green()/255, qcol.blue()/255])
            setattr(self, f"albedo_color_{endpoint}", rgb)
            self._update_albedo_button_style(endpoint)
            self._update_color_bar_preview()
            self._update_point_cloud_colors()

    def _albedo_interp_three(self, t):
        """三端颜色插值：t∈[0,1] -> low(0) -> mid(0.5) -> high(1)"""
        t = np.asarray(t)
        low, mid, high = self.albedo_color_low, self.albedo_color_mid, self.albedo_color_high
        # t <= 0.5: low -> mid
        # t > 0.5:  mid -> high
        t = np.clip(t, 0, 1)
        mask_low = t <= 0.5
        mask_high = ~mask_low
        t_low = t[mask_low] / 0.5 if np.any(mask_low) else np.array([])
        t_high = (t[mask_high] - 0.5) / 0.5 if np.any(mask_high) else np.array([])
        n = len(t)
        colors = np.zeros((n, 3), dtype=np.float64)
        if np.any(mask_low):
            colors[mask_low] = (1 - t_low)[:, np.newaxis] * low + t_low[:, np.newaxis] * mid
        if np.any(mask_high):
            colors[mask_high] = (1 - t_high)[:, np.newaxis] * mid + t_high[:, np.newaxis] * high
        return colors

    def _update_color_bar_preview(self):
        """更新三端颜色条预览"""
        c1, c2, c3 = self.albedo_color_low, self.albedo_color_mid, self.albedo_color_high
        r1, g1, b1 = int(c1[0]*255), int(c1[1]*255), int(c1[2]*255)
        r2, g2, b2 = int(c2[0]*255), int(c2[1]*255), int(c2[2]*255)
        r3, g3, b3 = int(c3[0]*255), int(c3[1]*255), int(c3[2]*255)
        self.color_bar_preview.setStyleSheet(
            f"background: qlineargradient(x1:0, y1:0, x2:1, y2:0, "
            f"stop:0 rgb({r1},{g1},{b1}), stop:0.5 rgb({r2},{g2},{b2}), stop:1 rgb({r3},{g3},{b3})); "
            "border: 1px solid #666;"
        )

    def _pick_solo_color(self, is_left=True):
        """选择 Solo 模式下左/右面的颜色"""
        col = self.solo_left_color if is_left else self.solo_right_color
        qcol = QColorDialog.getColor(
            QColor(int(col[0]*255), int(col[1]*255), int(col[2]*255)),
            self, "选择颜色"
        )
        if qcol.isValid():
            rgb = np.array([qcol.red()/255, qcol.green()/255, qcol.blue()/255])
            if is_left:
                self.solo_left_color = rgb
                self.btn_solo_left_color.setStyleSheet(
                    f"background-color: rgb({qcol.red()},{qcol.green()},{qcol.blue()}); "
                    f"color: {'white' if (qcol.red()+qcol.green()+qcol.blue())/3 < 128 else 'black'};"
                )
            else:
                self.solo_right_color = rgb
                self.btn_solo_right_color.setStyleSheet(
                    f"background-color: rgb({qcol.red()},{qcol.green()},{qcol.blue()}); "
                    f"color: {'white' if (qcol.red()+qcol.green()+qcol.blue())/3 < 128 else 'black'};"
                )
            self._apply_solo_mode()

    def _on_solo_opacity_changed(self, is_left, value):
        """Solo 模式透明度变化时更新"""
        if is_left:
            self.solo_left_opacity = value
        else:
            self.solo_right_opacity = value
        self._apply_solo_mode()

    def _on_show_edges_changed(self, checked):
        """边线显示复选框变化时更新"""
        self._apply_solo_mode()

    def _on_show_obb_changed(self, checked):
        """OBB 框线显示复选框变化时更新"""
        self._apply_solo_mode()

    def _on_stf_toggled(self, checked):
        """STF 模式开关：仅 model mode 有效，进入后点选面可切换其是否在透明面列表中"""
        self.stf_mode = checked
        if checked:
            self.btn_stf.setStyleSheet("QPushButton:checked { background-color: #4a9eff; font-weight: bold; }")
        else:
            self.btn_stf.setStyleSheet("")
        self._apply_solo_mode()

    def _on_transparent_opacity_changed(self, value):
        """透明面透明度滑条变化"""
        self.transparent_face_opacity = value / 100.0
        if hasattr(self, "lbl_transparent_opacity"):
            self.lbl_transparent_opacity.setText(f"{self.transparent_face_opacity:.2f}")
        self._apply_solo_mode()

    def _update_default_model_color_button(self):
        """更新 Model 默认颜色按钮样式"""
        if hasattr(self, "btn_default_model_color") and hasattr(self, "default_model_color"):
            c = self.default_model_color
            r, g, b = int(c[0]*255), int(c[1]*255), int(c[2]*255)
            self.btn_default_model_color.setStyleSheet(f"background-color: rgb({r},{g},{b});")

    def _pick_default_model_color(self):
        """选择 Model 模式默认颜色"""
        c = getattr(self, "default_model_color", np.array([0.83, 0.83, 0.83]))
        qcol = QColorDialog.getColor(
            QColor(int(c[0]*255), int(c[1]*255), int(c[2]*255)),
            self, "Model 默认颜色"
        )
        if qcol.isValid():
            self.default_model_color = np.array([qcol.red()/255, qcol.green()/255, qcol.blue()/255])
            self._update_default_model_color_button()
            self._apply_solo_mode()

    def _on_default_model_opacity_changed(self, value):
        """Model 默认透明度滑条变化"""
        self.default_model_opacity = value / 100.0
        if hasattr(self, "lbl_default_model_opacity"):
            self.lbl_default_model_opacity.setText(f"{self.default_model_opacity:.2f}")
        self._apply_solo_mode()

    def _pick_stf_color(self, is_highlight=True):
        """选择 STF 模式下的普通色或高亮色"""
        col = self.stf_color_highlight if is_highlight else self.stf_color_normal
        qcol = QColorDialog.getColor(
            QColor(int(col[0]*255), int(col[1]*255), int(col[2]*255)),
            self, "选择 STF 颜色"
        )
        if qcol.isValid():
            rgb = np.array([qcol.red()/255, qcol.green()/255, qcol.blue()/255])
            if is_highlight:
                self.stf_color_highlight = rgb
                self.btn_stf_highlight_color.setStyleSheet(
                    f"background-color: rgb({qcol.red()},{qcol.green()},{qcol.blue()});"
                )
            else:
                self.stf_color_normal = rgb
                self.btn_stf_normal_color.setStyleSheet(
                    f"background-color: rgb({qcol.red()},{qcol.green()},{qcol.blue()});"
                )
            self._apply_solo_mode()

    def _pick_edge_color(self):
        col = self.edge_line_color
        qcol = QColorDialog.getColor(
            QColor(int(col[0]*255), int(col[1]*255), int(col[2]*255)),
            self, "选择边线颜色"
        )
        if qcol.isValid():
            self.edge_line_color = np.array([qcol.red()/255, qcol.green()/255, qcol.blue()/255])
            self.btn_edge_color.setStyleSheet(
                f"background-color: rgb({qcol.red()},{qcol.green()},{qcol.blue()}); "
                f"color: {'white' if (qcol.red()+qcol.green()+qcol.blue())/3 < 128 else 'black'};"
            )
            self._update_edge_actors_color()

    def _pick_obb_color(self):
        col = self.obb_line_color
        qcol = QColorDialog.getColor(
            QColor(int(col[0]*255), int(col[1]*255), int(col[2]*255)),
            self, "选择 OBB 框线颜色"
        )
        if qcol.isValid():
            self.obb_line_color = np.array([qcol.red()/255, qcol.green()/255, qcol.blue()/255])
            self.btn_obb_color.setStyleSheet(
                f"background-color: rgb({qcol.red()},{qcol.green()},{qcol.blue()}); "
                f"color: {'white' if (qcol.red()+qcol.green()+qcol.blue())/3 < 128 else 'black'};"
            )
            if self.obb_box_actor is not None:
                self.obb_box_actor.prop.color = tuple(self.obb_line_color.tolist())
                self.plotter.render()

    def _update_edge_actors_color(self):
        c = tuple(self.edge_line_color.tolist())
        if self.model_edge_actor is not None:
            self.model_edge_actor.prop.color = c
        if self.solo_edge_actor is not None:
            self.solo_edge_actor.prop.color = c
        self.plotter.render()

    def _on_edge_radius_changed(self, value):
        self.edge_line_radius_scale = value
        self._recreate_edge_tubes()

    def _on_obb_radius_changed(self, value):
        self.obb_line_radius_scale = value
        self._recreate_obb_tube()

    def _recreate_edge_tubes(self):
        if not hasattr(self, "index_to_face") or not self.index_to_face:
            return
        r = self.model_extent * self.edge_line_radius_scale
        edge_polydatas = []
        for face in self.index_to_face.values():
            mesh = triangulate_face(face)
            if mesh:
                try:
                    edges = mesh.extract_feature_edges(
                        boundary_edges=True,
                        non_manifold_edges=False,
                        feature_edges=False,
                        manifold_edges=False
                    )
                    if edges.n_points > 0 and edges.n_cells > 0:
                        edge_polydatas.append(edges)
                except Exception:
                    pass
        merged = _merge_deduplicate_edges(edge_polydatas) if edge_polydatas else None
        if merged is not None:
            tube_mesh = _lines_to_tube_polydata(merged, r)
            if tube_mesh is not None:
                if self.model_edge_actor is not None:
                    try:
                        self.plotter.remove_actor(self.model_edge_actor)
                    except Exception:
                        pass
                self.model_edge_actor = self.plotter.add_mesh(
                    tube_mesh,
                    color=tuple(self.edge_line_color.tolist()),
                    lighting=False,
                    pickable=False
                )
        if self.solo_edge_actor is not None and self.current_face_pair:
            left_id, right_id = self.current_face_pair
            edge_polydatas = []
            for fid in (left_id, right_id):
                if fid in self.index_to_face:
                    mesh = triangulate_face(self.index_to_face[fid])
                    if mesh:
                        try:
                            edges = mesh.extract_feature_edges(
                                boundary_edges=True,
                                non_manifold_edges=False,
                                feature_edges=False,
                                manifold_edges=False
                            )
                            if edges.n_points > 0 and edges.n_cells > 0:
                                edge_polydatas.append(edges)
                        except Exception:
                            pass
            merged = _merge_deduplicate_edges(edge_polydatas) if edge_polydatas else None
            if merged is not None:
                tube_mesh = _lines_to_tube_polydata(merged, r)
                if tube_mesh is not None:
                    try:
                        self.plotter.remove_actor(self.solo_edge_actor)
                    except Exception:
                        pass
                    self.solo_edge_actor = self.plotter.add_mesh(
                        tube_mesh,
                        color=tuple(self.edge_line_color.tolist()),
                        lighting=False,
                        pickable=False
                    )
        self._apply_solo_mode()

    def _recreate_obb_tube(self):
        if self.obb_box_actor is None or not self.current_face_pair or not self.config_dir or not self.config_name:
            return
        left_id, right_id = self.current_face_pair
        step_path = os.path.join(self.config_dir, f"{self.config_name}.step")
        try:
            obb_o, obb_x, obb_y, obb_z = compute_obb_for_faces(step_path, left_id, right_id)
            obb_extent = np.linalg.norm(obb_x) + np.linalg.norm(obb_y) + np.linalg.norm(obb_z)
            r = obb_extent * self.obb_line_radius_scale
            obb_lines = _create_obb_wireframe_polydata(obb_o, obb_x, obb_y, obb_z)
            obb_tube = _lines_to_tube_polydata(obb_lines, r)
            if obb_tube is not None:
                try:
                    self.plotter.remove_actor(self.obb_box_actor)
                except Exception:
                    pass
                self.obb_box_actor = self.plotter.add_mesh(
                    obb_tube,
                    color=tuple(self.obb_line_color.tolist()),
                    lighting=False,
                    pickable=False
                )
                self.plotter.render()
        except Exception as e:
            print(f"OBB 框线重建失败: {e}")

    def _on_point_cloud_opacity_changed(self, value):
        """点云小球透明度滑条变化"""
        self.point_cloud_opacity = value / 100.0
        if hasattr(self, "lbl_point_opacity"):
            self.lbl_point_opacity.setText(f"{self.point_cloud_opacity:.2f}")
        if not self.isosurface_filter_active:
            for actor in self.point_cloud_actors:
                actor.prop.opacity = self.point_cloud_opacity
        if self.point_cloud_actors:
            self.plotter.render()

    def _update_isosurface_color_button(self):
        """更新等值面色按钮样式"""
        if hasattr(self, "btn_isosurface_color") and hasattr(self, "isosurface_color"):
            c = self.isosurface_color
            r, g, b = int(c[0]*255), int(c[1]*255), int(c[2]*255)
            fg = "white" if (r + g + b) / 3 < 128 else "black"
            self.btn_isosurface_color.setStyleSheet(f"background-color: rgb({r},{g},{b}); color: {fg};")

    def _pick_isosurface_color(self):
        """选择等值面颜色"""
        c = getattr(self, "isosurface_color", np.array([0.2, 0.8, 0.4]))
        qcol = QColorDialog.getColor(
            QColor(int(c[0]*255), int(c[1]*255), int(c[2]*255)),
            self, "等值面颜色"
        )
        if qcol.isValid():
            self.isosurface_color = np.array([qcol.red()/255, qcol.green()/255, qcol.blue()/255])
            self._update_isosurface_color_button()
            if self.isosurface_filter_active:
                self._refresh_point_cloud_display()
            for actor in self.isosurface_surface_actors:
                actor.prop.color = tuple(self.isosurface_color.tolist())
            if self.isosurface_surface_actors:
                self.plotter.render()

    def _on_isosurface_source_changed(self, idx):
        self.isosurface_offset_source = self.combo_isosurface_source.currentData()
        self._sync_isosurface_offset_range()
        if self.isosurface_filter_active:
            self._refresh_point_cloud_display()

    def _on_isosurface_offset_changed(self, value):
        self.isosurface_offset_value = value
        if self.isosurface_filter_active:
            self._refresh_point_cloud_display()

    def _on_isosurface_tolerance_changed(self, value):
        self.isosurface_tolerance = value
        if self.isosurface_filter_active:
            self._refresh_point_cloud_display()

    def _on_isosurface_new_behavior_changed(self, state):
        """新行为(双区间) checkbox：勾选=新行为，不勾选=旧行为"""
        self.isosurface_new_behavior = state == Qt.Checked

    def _on_isosurface_opacity_changed(self, value):
        self.isosurface_opacity = value / 100.0
        if hasattr(self, "lbl_isosurface_opacity"):
            self.lbl_isosurface_opacity.setText(f"{self.isosurface_opacity:.2f}")
        if self.isosurface_filter_active:
            self._refresh_point_cloud_display()
        for actor in self.isosurface_surface_actors:
            actor.prop.opacity = self.isosurface_opacity
        if self.isosurface_surface_actors:
            self.plotter.render()

    def _on_isosurface_ext_v_changed(self, value):
        """extV 滑条 (0~100) 映射为 0~1"""
        self.isosurface_ext_v = value / 100.0
        if hasattr(self, "lbl_isosurface_ext_v"):
            self.lbl_isosurface_ext_v.setText(f"{self.isosurface_ext_v:.2f}")

    def _on_isosurface_extend_clicked(self):
        """对批量拟合结果中 type=bspline 的等值曲面做 UV 延展，并刷新显示"""
        if not self.batch_fit_done or not self.batch_isosurface_meshes:
            return
        ext_v = getattr(self, "isosurface_ext_v", 0.0)
        if ext_v <= 0:
            self.lbl_info.setText("延展: extV 需大于 0")
            return
        extended_count = 0
        for idx, item in enumerate(self.batch_isosurface_meshes):
            surface, grid_points, triangles = item[0], item[1], item[2]
            if surface.get("type") != "bspline":
                continue
            surface_new = extend_bspline_surface(surface, ext_v)
            grid_new, tri_new = sample_surface_to_mesh(surface_new, ISOSURFACE_FIT_N_GRID)
            if grid_new is None:
                continue
            self.batch_isosurface_meshes[idx] = (surface_new, grid_new, tri_new)
            try:
                self.plotter.remove_actor(self.batch_isosurface_actors[idx])
            except Exception:
                pass
            faces = np.column_stack([np.full(tri_new.shape[0], 3, dtype=np.int32), tri_new]).ravel()
            mesh = pv.PolyData(grid_new, faces)
            actor = self.plotter.add_mesh(
                mesh,
                color=tuple(self.isosurface_color.tolist()),
                opacity=self.isosurface_opacity,
                lighting=True,
                pickable=False,
            )
            self.batch_isosurface_actors[idx] = actor
            extended_count += 1
        self.lbl_info.setText(f"延展完成: {extended_count} 个 B-spline 等值面 (extV={ext_v:.2f})")
        self.plotter.render()

    def _on_isosurface_filter_clicked(self):
        """筛选显示按钮：只显示 offset 在 [value-tol, value+tol] 内的点"""
        self.isosurface_filter_active = self.btn_isosurface_filter.isChecked()
        self._refresh_point_cloud_display()
        n = self._get_filtered_point_count() if self.isosurface_filter_active else None
        if self.isosurface_filter_active and n is not None:
            self.lbl_info.setText(f"筛选显示: 共 {n} 个点参与拟合 (offset ∈ [{self.isosurface_offset_value - self.isosurface_tolerance:.4f}, {self.isosurface_offset_value + self.isosurface_tolerance:.4f}])")
        elif not self.isosurface_filter_active and self.point_cloud_data:
            total = len(self.point_cloud_data["query_points_ws"])
            self.lbl_info.setText(f"显示全部: {total} 个点")

    def _remove_isosurface_surface(self):
        """移除单次拟合的所有曲面"""
        for actor in self.isosurface_surface_actors:
            try:
                self.plotter.remove_actor(actor)
            except Exception:
                pass
        self.isosurface_surface_actors = []

    def _remove_batch_isosurfaces(self):
        """移除所有批量拟合曲面"""
        for actor in self.batch_isosurface_actors:
            try:
                self.plotter.remove_actor(actor)
            except Exception:
                pass
        self.batch_isosurface_actors = []
        self.batch_isosurface_meshes = []
        self.batch_fit_done = False
        if hasattr(self, "btn_export_midsurf"):
            self.btn_export_midsurf.setEnabled(False)
        if hasattr(self, "btn_isosurface_extend"):
            self.btn_isosurface_extend.setEnabled(False)

    def _batch_fit_all_isosurfaces(self):
        """对所有面对批量拟合等值面并显示"""
        if not self.config_dir or not self.config_name:
            self.lbl_info.setText("请先加载配置文件 (TXT)")
            return
        if not self.current_solid:
            self.lbl_info.setText("请先加载 STEP 模型")
            return
        if not self.face_tag_groups:
            self.lbl_info.setText("配置文件中无面对")
            return
        self._remove_batch_isosurfaces()
        self._remove_isosurface_surface()
        offset_src = self.isosurface_offset_source
        value = float(self.isosurface_offset_value)
        tol = float(self.isosurface_tolerance)
        use_new = getattr(self, "isosurface_new_behavior", False)
        if use_new and abs(value - 0.5) >= 1e-9:
            centers = [value, 1.0 - value]
        else:
            centers = [value]
        ok_count = 0
        skip_count = 0
        flip_cbs = getattr(self, "flip_checkboxes", [])
        for row_i, tags in enumerate(self.face_tag_groups):
            if len(tags) != 2:
                continue
            left_id, right_id = tags[0], tags[1]
            data = self._load_point_cloud_data_for_pair(left_id, right_id)
            if data is None:
                skip_count += 1
                continue
            offset = np.asarray(data[offset_src], dtype=np.float64)
            if offset_src in ("offset_pred", "offset_gt") and row_i < len(flip_cbs) and flip_cbs[row_i].isChecked():
                offset = 1.0 - offset
            validity_key = "validity_pred" if offset_src == "offset_pred" else "validity_gt"
            validity = np.asarray(data[validity_key]) if validity_key in data else None
            for center in centers:
                lo = center - tol
                hi = center + tol
                mask = (offset >= lo) & (offset <= hi)
                if validity is not None:
                    mask = mask & (validity > 0)
                points = np.asarray(data["query_points_ws"])[mask]
                if len(points) < 16:
                    skip_count += 1
                    continue
                left_face = self.index_to_face.get(left_id) if getattr(self, "index_to_face", None) else None
                right_face = self.index_to_face.get(right_id) if getattr(self, "index_to_face", None) else None
                surface, info = fit_isosurface_by_face_type(
                    points, left_face, right_face, n_grid=ISOSURFACE_FIT_N_GRID
                )
                if surface is None:
                    skip_count += 1
                    continue
                grid_points, triangles = sample_surface_to_mesh(surface, ISOSURFACE_FIT_N_GRID)
                if grid_points is None:
                    skip_count += 1
                    continue
                self.batch_isosurface_meshes.append((surface, grid_points, triangles))
                faces = np.column_stack([np.full(triangles.shape[0], 3, dtype=np.int32), triangles]).ravel()
                mesh = pv.PolyData(grid_points, faces)
                actor = self.plotter.add_mesh(
                    mesh,
                    color=tuple(self.isosurface_color.tolist()),
                    opacity=self.isosurface_opacity,
                    lighting=True,
                    pickable=False,
                )
                self.batch_isosurface_actors.append(actor)
                ok_count += 1
        self.batch_fit_done = ok_count > 0
        if hasattr(self, "btn_export_midsurf"):
            self.btn_export_midsurf.setEnabled(self.batch_fit_done)
        if hasattr(self, "btn_isosurface_extend"):
            self.btn_isosurface_extend.setEnabled(self.batch_fit_done)
        self.lbl_info.setText(
            f"批量拟合完成: {ok_count} 个等值面 | 跳过 {skip_count} 个面对"
        )
        self.plotter.render()

    def _export_midsurf_step(self):
        """将原始 solid + 批量拟合的等值面导出为 _midsurf.step"""
        if not self.batch_fit_done or not self.batch_isosurface_meshes:
            self.lbl_info.setText("请先执行「批量拟合」")
            return
        if not self.current_solid:
            self.lbl_info.setText("无模型数据")
            return
        if not self.config_dir or not self.config_name:
            self.lbl_info.setText("无配置路径")
            return
        out_name = f"{self.config_name}_midsurf.step"
        out_path = os.path.join(self.config_dir, out_name)
        try:
            solid_shape = self.current_solid.topods_shape()
        except Exception as e:
            self.lbl_info.setText(f"获取 solid 失败: {e}")
            return
        shapes = [solid_shape]
        for item in self.batch_isosurface_meshes:
            surface, grid_points, triangles = item[0], item[1], item[2]
            occ_face = surface_to_occ_face(surface)
            if occ_face is not None:
                shapes.append(occ_face)
            else:
                occ_shape = polydata_to_occ_shape(grid_points, triangles)
                if occ_shape is not None:
                    shapes.append(occ_shape)
        builder = BRep_Builder()
        compound = TopoDS_Compound()
        builder.MakeCompound(compound)
        for s in shapes:
            builder.Add(compound, s)
        writer = STEPControl_Writer()
        writer.Transfer(compound, STEPControl_AsIs)
        status = writer.Write(out_path)
        if status == IFSelect_RetDone:
            self.lbl_info.setText(f"已导出: {out_path}")
        else:
            self.lbl_info.setText(f"STEP 导出失败: {out_path}")

    def _on_isosurface_fit_clicked(self):
        """拟合按钮：根据 offset 区间对当前面对进行等值面拟合并渲染。

        旧行为（offset_value==0.5）：一次操作，显示一个曲面。
        新行为（offset_value!=0.5）：两次操作（原区间 + 以 0.5 为中心的对称区间），同时显示两个结果曲面。
        """
        if not self.point_cloud_data or self.isosurface_offset_source not in self.point_cloud_data:
            self.lbl_info.setText("无点云数据，请先加载面对点云")
            return
        offset = np.asarray(self.point_cloud_data[self.isosurface_offset_source])
        query_ws = self.point_cloud_data["query_points_ws"]
        validity_key = "validity_pred" if self.isosurface_offset_source == "offset_pred" else "validity_gt"
        validity = np.asarray(self.point_cloud_data[validity_key]) if validity_key in self.point_cloud_data else None
        value = float(self.isosurface_offset_value)
        tol = float(self.isosurface_tolerance)
        use_new = getattr(self, "isosurface_new_behavior", False)
        if use_new and abs(value - 0.5) >= 1e-9:
            centers = [value, 1.0 - value]
        else:
            centers = [value]

        left_face = self.index_to_face.get(self.current_face_pair[0]) if getattr(self, "current_face_pair", None) and getattr(self, "index_to_face", None) else None
        right_face = self.index_to_face.get(self.current_face_pair[1]) if getattr(self, "current_face_pair", None) and getattr(self, "index_to_face", None) else None

        self._remove_isosurface_surface()
        success_count = 0
        last_info = None

        for center in centers:
            lo = center - tol
            hi = center + tol
            mask = (offset >= lo) & (offset <= hi)
            if validity is not None:
                mask = mask & (validity > 0)
            points = np.asarray(query_ws)[mask]
            if len(points) < 16:
                continue
            surface, info = fit_isosurface_by_face_type(
                points, left_face, right_face, n_grid=ISOSURFACE_FIT_N_GRID
            )
            if surface is None:
                continue
            grid_points, triangles = sample_surface_to_mesh(surface, ISOSURFACE_FIT_N_GRID)
            if grid_points is None:
                continue
            faces = np.column_stack([np.full(triangles.shape[0], 3, dtype=np.int32), triangles]).ravel()
            mesh = pv.PolyData(grid_points, faces)
            actor = self.plotter.add_mesh(
                mesh,
                color=tuple(self.isosurface_color.tolist()),
                opacity=self.isosurface_opacity,
                lighting=True,
                pickable=False,
            )
            self.isosurface_surface_actors.append(actor)
            success_count += 1
            last_info = info

        if success_count == 0:
            self.lbl_info.setText("等值面拟合失败：区间内有效点数不足或曲面拟合失败")
            return
        fit_type = last_info.get("type", "non-periodic") if last_info else "non-periodic"
        n_input = last_info.get("n_input_points") if last_info and "n_input_points" in last_info else "?"
        msg = f"拟合完成 ({fit_type}): {n_input} 点 → {ISOSURFACE_FIT_N_GRID}×{ISOSURFACE_FIT_N_GRID} 网格"
        if success_count == 2:
            msg += "（两个曲面已同时显示）"
        self.lbl_info.setText(msg)
        self.plotter.render()

    def _get_filtered_point_count(self):
        """返回符合筛选条件的点数（offset 在范围内且 validity=1），无数据返回 None"""
        if not self.point_cloud_data or self.isosurface_offset_source not in self.point_cloud_data:
            return None
        offset = np.asarray(self.point_cloud_data[self.isosurface_offset_source])
        validity_key = "validity_pred" if self.isosurface_offset_source == "offset_pred" else "validity_gt"
        if validity_key not in self.point_cloud_data:
            return None
        validity = np.asarray(self.point_cloud_data[validity_key])
        lo = self.isosurface_offset_value - self.isosurface_tolerance
        hi = self.isosurface_offset_value + self.isosurface_tolerance
        mask = (offset >= lo) & (offset <= hi) & (validity > 0)
        return int(np.sum(mask))

    def _clear_point_cloud_actors_only(self):
        """仅移除当前单面对点云的 actor，不清空 point_cloud_data（用于切换至概览前隐藏单面对点云）"""
        for actor in self.point_cloud_actors:
            try:
                self.plotter.remove_actor(actor)
            except Exception:
                pass
        self.point_cloud_actors.clear()

    def _clear_overview_point_clouds(self):
        """移除概览模式下所有面对的点云 actor，并清除实时重着色用的缓存"""
        for actor in self.overview_point_cloud_actors:
            try:
                self.plotter.remove_actor(actor)
            except Exception:
                pass
        self.overview_point_cloud_actors.clear()
        for attr in ("_overview_pair_data", "_overview_global_vmin", "_overview_global_vmax", "_overview_albedo_key"):
            if hasattr(self, attr):
                delattr(self, attr)

    def _create_overview_point_clouds(self):
        """概览模式：为每个面对加载点云并按配额采样后显示，颜色与 Model/Solo 一致，由点的 o/v 映射到 albedo 颜色条。
        每面对最大点数由文件顶部常量 OVERVIEW_POINTS_PER_PAIR 控制（默认 400）。"""
        if not self.config_dir or not self.config_name or not getattr(self, "face_tag_groups", None):
            return
        # 与单面对点云一致的数据源（offset_pred/offset_gt/validity_pred/validity_gt）
        key = self.combo_albedo_source.currentData() if hasattr(self, "combo_albedo_source") else None
        if not key:
            key = "offset_pred"
        extent = getattr(self, "model_extent", 1.0) or 1.0
        base_radius = extent * 0.0008
        scale = getattr(self, "point_cloud_radius_scale", 1.0)
        sphere_radius = base_radius * scale
        sphere_geom = pv.Sphere(radius=sphere_radius, theta_resolution=5, phi_resolution=5)
        op = getattr(self, "point_cloud_opacity", 0.9)

        # 第一遍：收集 (row_i, pts, scalars_raw)，scalars_raw 为未取反的标量；并计算全局 vmin/vmax（按当前取反状态）
        all_scalars = []
        pair_data_list = []
        flip_cbs = getattr(self, "flip_checkboxes", [])
        for i, tags in enumerate(self.face_tag_groups):
            if len(tags) != 2:
                continue
            left_id, right_id = tags[0], tags[1]
            data = self._load_point_cloud_data_for_pair(left_id, right_id)
            if data is None or "query_points_ws" not in data or key not in data:
                continue
            pts = np.asarray(data["query_points_ws"], dtype=np.float64)
            scalars_raw = np.asarray(data[key], dtype=np.float64)
            if len(pts) == 0 or len(scalars_raw) != len(pts):
                continue
            pair_data_list.append((i, pts, scalars_raw))
            scalars_display = (1.0 - scalars_raw) if (key in ("offset_pred", "offset_gt") and i < len(flip_cbs) and flip_cbs[i].isChecked()) else scalars_raw
            all_scalars.append(scalars_display)
        if not pair_data_list:
            return
        all_scalars = np.concatenate(all_scalars)
        vmin, vmax = float(np.min(all_scalars)), float(np.max(all_scalars))
        if vmax <= vmin:
            vmax = vmin + 1e-6
        self._overview_global_vmin = vmin
        self._overview_global_vmax = vmax
        self._overview_albedo_key = key

        # 第二遍：按配额采样、着色，并保存 (row_i, actor, pts_s, scalars_raw_s) 供取反时实时重着色
        self._overview_pair_data = []
        for row_i, pts, scalars_raw in pair_data_list:
            n = len(pts)
            quota = min(n, OVERVIEW_POINTS_PER_PAIR)
            if n > quota:
                rng = np.random.default_rng(42)
                idx = rng.choice(n, size=quota, replace=False)
                pts_s = pts[idx]
                scalars_raw_s = scalars_raw[idx]
            else:
                pts_s = pts
                scalars_raw_s = scalars_raw
            flip = key in ("offset_pred", "offset_gt") and row_i < len(flip_cbs) and flip_cbs[row_i].isChecked()
            scalars = (1.0 - scalars_raw_s) if flip else scalars_raw_s
            t = (scalars - vmin) / (vmax - vmin)
            t = np.clip(t, 0, 1)
            colors = self._albedo_interp_three(t)
            colors = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
            pc_mesh = pv.PolyData(pts_s)
            pc_mesh["colors"] = colors
            glyph_mesh = pc_mesh.glyph(scale=False, orient=False, geom=sphere_geom)
            actor = self.plotter.add_mesh(
                glyph_mesh,
                scalars="colors",
                rgb=True,
                lighting=False,
                pickable=False,
            )
            actor.prop.opacity = op
            self.overview_point_cloud_actors.append(actor)
            self._overview_pair_data.append((row_i, actor, pts_s.copy(), scalars_raw_s.copy()))
            mapper = actor.GetMapper()
            if mapper and mapper.GetInput():
                mapper.SetScalarModeToUsePointFieldData()
                mapper.SelectColorArray("colors")
                mapper.SetLookupTable(None)
                mapper.SetColorModeToDirectScalars()
        if self.overview_point_cloud_actors:
            self.plotter.render()

    def _recolor_overview_pair(self, row_index):
        """概览模式下，对指定行（面对）的代表性点云按当前取反状态重新着色，实现实时更新。"""
        pair_data = getattr(self, "_overview_pair_data", None)
        if not pair_data:
            return
        vmin = getattr(self, "_overview_global_vmin", None)
        vmax = getattr(self, "_overview_global_vmax", None)
        key = getattr(self, "_overview_albedo_key", "offset_pred")
        if vmin is None or vmax is None or vmax <= vmin:
            return
        flip_cbs = getattr(self, "flip_checkboxes", [])
        extent = getattr(self, "model_extent", 1.0) or 1.0
        base_radius = extent * 0.0008
        scale = getattr(self, "point_cloud_radius_scale", 1.0)
        sphere_radius = base_radius * scale
        sphere_geom = pv.Sphere(radius=sphere_radius, theta_resolution=5, phi_resolution=5)
        op = getattr(self, "point_cloud_opacity", 0.9)
        for idx, (row_i, old_actor, pts_s, scalars_raw_s) in enumerate(pair_data):
            if row_i != row_index:
                continue
            try:
                self.plotter.remove_actor(old_actor)
            except Exception:
                pass
            flip = key in ("offset_pred", "offset_gt") and row_index < len(flip_cbs) and flip_cbs[row_index].isChecked()
            scalars = (1.0 - scalars_raw_s) if flip else scalars_raw_s
            t = (scalars - vmin) / (vmax - vmin)
            t = np.clip(t, 0, 1)
            colors = self._albedo_interp_three(t)
            colors = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
            pc_mesh = pv.PolyData(pts_s)
            pc_mesh["colors"] = colors
            glyph_mesh = pc_mesh.glyph(scale=False, orient=False, geom=sphere_geom)
            new_actor = self.plotter.add_mesh(
                glyph_mesh,
                scalars="colors",
                rgb=True,
                lighting=False,
                pickable=False,
            )
            new_actor.prop.opacity = op
            mapper = new_actor.GetMapper()
            if mapper and mapper.GetInput():
                mapper.SetScalarModeToUsePointFieldData()
                mapper.SelectColorArray("colors")
                mapper.SetLookupTable(None)
                mapper.SetColorModeToDirectScalars()
            self._overview_pair_data[idx] = (row_i, new_actor, pts_s, scalars_raw_s)
            self.overview_point_cloud_actors[idx] = new_actor
            self.plotter.render()
            return

    def _refresh_point_cloud_display(self):
        """根据当前筛选状态重建点云显示"""
        if not self.point_cloud_data:
            return
        for actor in self.point_cloud_actors:
            try:
                self.plotter.remove_actor(actor)
            except Exception:
                pass
        self.point_cloud_actors.clear()
        self._create_point_cloud_glyphs()
        if self.point_cloud_actors:
            self.plotter.render()

    def _on_point_radius_changed(self, value):
        """点云小球半径滑条变化时更新"""
        self.point_cloud_radius_scale = value / 100.0
        if hasattr(self, "lbl_radius_value"):
            self.lbl_radius_value.setText(f"{self.point_cloud_radius_scale:.1f}×")
        if self.point_cloud_data and self.point_cloud_actors:
            # 移除旧 glyph，用新半径重新创建
            for actor in self.point_cloud_actors:
                try:
                    self.plotter.remove_actor(actor)
                except Exception:
                    pass
            self.point_cloud_actors.clear()
            self._create_point_cloud_glyphs()
            self.plotter.render()

    def _on_albedo_source_changed(self):
        """Albedo 数据源切换时更新点云颜色；概览模式下重建概览点云以应用新数据源"""
        if getattr(self, "combo_render_mode", None) is not None and self.combo_render_mode.currentIndex() == 2:
            self._clear_overview_point_clouds()
            self._create_overview_point_clouds()
        else:
            self._update_point_cloud_colors()

    def _load_point_cloud_data_for_pair(self, left_id, right_id):
        """
        加载指定面对的点云数据（npz 或 txt fallback），不修改场景。
        返回 dict(query_points_ws, offset_pred, offset_gt, ...) 或 None。
        """
        if not self.config_dir or not self.config_name:
            return None
        npz_name = f"{self.config_name}_{left_id}_{right_id}_result.npz"
        npz_path = os.path.join(self.config_dir, npz_name)
        if not os.path.exists(npz_path):
            npz_name_alt = f"{self.config_name}_{right_id}_{left_id}_result.npz"
            npz_path = os.path.join(self.config_dir, npz_name_alt)
            if not os.path.exists(npz_path):
                npz_path = None
            else:
                left_id, right_id = right_id, left_id
        if npz_path is not None:
            try:
                data = np.load(npz_path)
                query_points = np.asarray(data["query_points"], dtype=np.float64)
                offset_pred, offset_gt = data["offset_pred"], data["offset_gt"]
                validity_pred, validity_gt = data["validity_pred"], data["validity_gt"]
            except Exception:
                npz_path = None
        if npz_path is None:
            result = self._load_point_cloud_from_fallback_txt(left_id, right_id)
            if result is None:
                return None
            query_points, offset_pred, offset_gt, validity_pred, validity_gt = result
        step_path = os.path.join(self.config_dir, f"{self.config_name}.step")
        try:
            obb_origin, obb_x_vec, obb_y_vec, obb_z_vec = compute_obb_for_faces(
                step_path, left_id, right_id
            )
        except Exception:
            return None
        query_points_ws = obb_normalized_to_world(
            query_points, obb_origin, obb_x_vec, obb_y_vec, obb_z_vec
        )
        if POINT_CLOUD_MAX_POINTS > 0 and len(query_points_ws) > POINT_CLOUD_MAX_POINTS:
            step = max(1, len(query_points_ws) // POINT_CLOUD_MAX_POINTS)
            idx = np.arange(0, len(query_points_ws), step)[:POINT_CLOUD_MAX_POINTS]
            query_points_ws = query_points_ws[idx]
            offset_pred = np.asarray(offset_pred)[idx]
            offset_gt = np.asarray(offset_gt)[idx]
            validity_pred = np.asarray(validity_pred)[idx]
            validity_gt = np.asarray(validity_gt)[idx]
        return {
            "query_points_ws": query_points_ws,
            "offset_pred": offset_pred,
            "offset_gt": offset_gt,
            "validity_pred": validity_pred,
            "validity_gt": validity_gt,
        }

    def _load_point_cloud_from_fallback_txt(self, left_id, right_id):
        """
        Fallback: 在 config_dir/fallback 中查找 config_name_left_right.txt，
        解析 SAMPLES_DATA（x y z o v），offset_pred=offset_gt, validity_pred=validity_gt。
        返回 (query_points, offset_pred, offset_gt, validity_pred, validity_gt) 或 None
        """
        fallback_dir = os.path.join(self.config_dir, "fallback")
        if not os.path.isdir(fallback_dir):
            return None
        for (lid, rid) in [(left_id, right_id), (right_id, left_id)]:
            txt_name = f"{self.config_name}_{lid}_{rid}.txt"
            txt_path = os.path.join(fallback_dir, txt_name)
            result = parse_fallback_txt_samples(txt_path)
            if result is not None:
                query_points, offset_gt, validity_gt = result
                offset_pred = np.asarray(offset_gt).copy()
                validity_pred = np.asarray(validity_gt).copy()
                return (query_points, offset_pred, offset_gt, validity_pred, validity_gt)
        return None

    def _get_albedo_scalars(self):
        """根据当前选中的数据源返回标量数组"""
        if not self.point_cloud_data:
            return None
        key = self.combo_albedo_source.currentData()
        if key and key in self.point_cloud_data:
            return self.point_cloud_data[key]
        return None

    def _update_point_cloud_colors(self):
        """根据 Albedo 映射设置更新点云颜色"""
        if not self.point_cloud_data or not self.point_cloud_actors:
            return
        scalars = self._get_albedo_scalars()
        if scalars is None:
            return
        actor = self.point_cloud_actors[0]
        mapper = actor.GetMapper()
        if not mapper or not mapper.GetInput():
            return
        vmin, vmax = float(np.min(scalars)), float(np.max(scalars))
        if vmax <= vmin:
            vmax = vmin + 1e-6
        t = (scalars - vmin) / (vmax - vmin)
        t = np.clip(t, 0, 1)
        colors = self._albedo_interp_three(t)
        colors = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
        vtk_mesh = mapper.GetInput()
        n_centers = len(scalars)
        n_points = vtk_mesh.GetNumberOfPoints()
        # Glyph 输出：每个中心点对应多个顶点，需将颜色按顶点数复制
        if n_points > n_centers:
            n_per = n_points // n_centers
            colors = np.repeat(colors, n_per, axis=0)
        from vtk.util.numpy_support import numpy_to_vtk
        vtk_arr = numpy_to_vtk(colors, deep=True)
        vtk_arr.SetName("colors")
        vtk_mesh.GetPointData().SetScalars(vtk_arr)
        vtk_mesh.Modified()
        mapper.SetScalarModeToUsePointFieldData()
        mapper.SelectColorArray("colors")
        mapper.SetLookupTable(None)
        mapper.SetColorModeToDirectScalars()
        self.plotter.render()

    def _load_and_render_point_cloud(self, left_id, right_id):
        """加载 npz 并按 OBB 还原世界坐标，渲染点云。npz 失败时 fallback 到 fallback 文件夹中的 txt"""
        if not self.config_dir or not self.config_name:
            return False
        npz_name = f"{self.config_name}_{left_id}_{right_id}_result.npz"
        npz_path = os.path.join(self.config_dir, npz_name)
        if not os.path.exists(npz_path):
            npz_name_alt = f"{self.config_name}_{right_id}_{left_id}_result.npz"
            npz_path = os.path.join(self.config_dir, npz_name_alt)
            if not os.path.exists(npz_path):
                npz_path = None
            else:
                left_id, right_id = right_id, left_id
        if npz_path is not None:
            try:
                data = np.load(npz_path)
                query_points = np.asarray(data["query_points"], dtype=np.float64)
                offset_pred = data["offset_pred"]
                offset_gt = data["offset_gt"]
                validity_pred = data["validity_pred"]
                validity_gt = data["validity_gt"]
            except Exception as e:
                print(f"加载 npz 失败: {e}")
                npz_path = None
        if npz_path is None:
            result = self._load_point_cloud_from_fallback_txt(left_id, right_id)
            if result is None:
                return False
            query_points, offset_pred, offset_gt, validity_pred, validity_gt = result
        step_path = os.path.join(self.config_dir, f"{self.config_name}.step")
        try:
            obb_origin, obb_x_vec, obb_y_vec, obb_z_vec = compute_obb_for_faces(
                step_path, left_id, right_id
            )
        except Exception as e:
            print(f"OBB 计算失败: {e}")
            return False
        query_points_ws = obb_normalized_to_world(
            query_points, obb_origin, obb_x_vec, obb_y_vec, obb_z_vec
        )
        # 可选：数量限制（0=无限制）
        if POINT_CLOUD_MAX_POINTS > 0 and len(query_points_ws) > POINT_CLOUD_MAX_POINTS:
            step = max(1, len(query_points_ws) // POINT_CLOUD_MAX_POINTS)
            idx = np.arange(0, len(query_points_ws), step)[:POINT_CLOUD_MAX_POINTS]
            query_points_ws = query_points_ws[idx]
            offset_pred = np.asarray(offset_pred)[idx]
            offset_gt = np.asarray(offset_gt)[idx]
            validity_pred = np.asarray(validity_pred)[idx]
            validity_gt = np.asarray(validity_gt)[idx]
        self._clear_point_cloud()
        obb_extent = np.linalg.norm(obb_x_vec) + np.linalg.norm(obb_y_vec) + np.linalg.norm(obb_z_vec)
        self._point_cloud_obb_extent = obb_extent
        self.point_cloud_data = {
            "offset_pred": offset_pred,
            "offset_gt": offset_gt,
            "validity_pred": validity_pred,
            "validity_gt": validity_gt,
            "query_points_ws": query_points_ws,
        }
        self.current_face_pair = (left_id, right_id)
        # 若该行“反”已勾选，则对刚加载的 offset 做 1-offset，与列表状态一致
        row_i = self._get_row_index_for_pair(left_id, right_id)
        if (
            row_i is not None
            and row_i < len(getattr(self, "flip_checkboxes", []))
            and self.flip_checkboxes[row_i].isChecked()
        ):
            for key in ("offset_pred", "offset_gt"):
                if key in self.point_cloud_data:
                    arr = np.asarray(self.point_cloud_data[key], dtype=np.float64)
                    self.point_cloud_data[key] = 1.0 - arr
        self._log_offset_distribution()
        self._sync_isosurface_offset_range()
        self._create_point_cloud_glyphs()
        self.plotter.render()
        return True

    def _log_offset_distribution(self):
        """统计点云 offset 各区间分布并输出（在每次加载点云数据时调用一次）"""
        if not self.point_cloud_data:
            return
        n_bins = 20
        for key in ("offset_pred", "offset_gt"):
            if key not in self.point_cloud_data:
                continue
            arr = np.asarray(self.point_cloud_data[key], dtype=np.float64)
            if len(arr) == 0:
                continue
            lo, hi = float(np.min(arr)), float(np.max(arr))
            if hi <= lo:
                edges = np.array([lo, lo + 1e-9])
                counts, _ = np.histogram(arr, bins=edges)
            else:
                edges = np.linspace(lo, hi, n_bins + 1)
                counts, _ = np.histogram(arr, bins=edges)
            total = int(np.sum(counts))
            label = "offset_pred (预测)" if key == "offset_pred" else "offset_gt (真值)"
            print(f"[Offset 分布] {label}: 区间数={len(edges)-1}, 总数={total}, 范围=[{lo:.6g}, {hi:.6g}]")
            for i in range(len(counts)):
                a, b = edges[i], edges[i + 1]
                c = int(counts[i])
                pct = (100.0 * c / total) if total else 0
                print(f"  [{a:.6g}, {b:.6g}): {c:6d} ({pct:5.2f}%)")
            print()

    def _sync_isosurface_offset_range(self):
        """根据当前点云 offset 数据同步等值 offset_value 范围"""
        if not self.point_cloud_data or not hasattr(self, "spin_isosurface_offset"):
            return
        key = getattr(self, "isosurface_offset_source", "offset_pred")
        if key not in self.point_cloud_data:
            key = "offset_pred" if "offset_pred" in self.point_cloud_data else "offset_gt"
        if key not in self.point_cloud_data:
            return
        arr = np.asarray(self.point_cloud_data[key])
        lo, hi = float(np.min(arr)), float(np.max(arr))
        pad = max(0.1, (hi - lo) * 0.1) if hi > lo else 0.1
        self.spin_isosurface_offset.setRange(lo - pad, hi + pad)
        v = self.spin_isosurface_offset.value()
        if v < lo - pad or v > hi + pad:
            self.spin_isosurface_offset.setValue((lo + hi) / 2)

    def _create_point_cloud_glyphs(self):
        """根据当前点云数据与半径缩放创建/更新 glyph 球体。筛选显示时只渲染 offset∈[value-tol, value+tol] 的点，用等值面色与透明度"""
        if not self.point_cloud_data or "query_points_ws" not in self.point_cloud_data:
            return
        query_points_ws = self.point_cloud_data["query_points_ws"]
        
        use_filter = getattr(self, "isosurface_filter_active", False)
        if use_filter and getattr(self, "isosurface_offset_source", "") in self.point_cloud_data:
            offset = np.asarray(self.point_cloud_data[self.isosurface_offset_source])
            validity_key = "validity_pred" if self.isosurface_offset_source == "offset_pred" else "validity_gt"
            validity = np.asarray(self.point_cloud_data[validity_key]) if validity_key in self.point_cloud_data else None
            lo = self.isosurface_offset_value - self.isosurface_tolerance
            hi = self.isosurface_offset_value + self.isosurface_tolerance
            mask = (offset >= lo) & (offset <= hi)
            if validity is not None:
                mask = mask & (validity > 0)
            if not np.any(mask):
                return
            query_points_ws = query_points_ws[mask]
            c = self.isosurface_color
            colors = np.tile((np.clip(c, 0, 1) * 255).astype(np.uint8), (len(query_points_ws), 1))
            op = self.isosurface_opacity
        else:
            scalars = self._get_albedo_scalars()
            if scalars is None:
                scalars = self.point_cloud_data.get("offset_pred")
            if scalars is None:
                return
            vmin, vmax = float(np.min(scalars)), float(np.max(scalars))
            if vmax <= vmin:
                vmax = vmin + 1e-6
            t = (scalars - vmin) / (vmax - vmin)
            t = np.clip(t, 0, 1)
            colors = self._albedo_interp_three(t)
            colors = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
            op = self.point_cloud_opacity
        
        obb_extent = self._point_cloud_obb_extent or 1.0
        base_radius = obb_extent * 0.0008
        scale = self.point_cloud_radius_scale
        sphere_radius = base_radius * scale
        
        pc_mesh = pv.PolyData(query_points_ws)
        pc_mesh["colors"] = colors
        sphere_geom = pv.Sphere(radius=sphere_radius, theta_resolution=5, phi_resolution=5)
        glyph_mesh = pc_mesh.glyph(scale=False, orient=False, geom=sphere_geom)
        
        actor = self.plotter.add_mesh(
            glyph_mesh,
            scalars="colors",
            rgb=True,
            lighting=False,
            pickable=False,
        )
        actor.prop.opacity = op
        self.point_cloud_actors.append(actor)
        mapper = actor.GetMapper()
        if mapper and mapper.GetInput():
            mapper.SetScalarModeToUsePointFieldData()
            mapper.SelectColorArray("colors")
            mapper.SetLookupTable(None)
            mapper.SetColorModeToDirectScalars()

    def on_left_click(self, obj, event):
        """G-Buffer (Hardware) Picking - 点选面，若存在面对则加载并渲染点云"""
        click_pos = self.plotter.iren.get_event_position()
        x, y = click_pos[0], click_pos[1]

        selector = vtk.vtkHardwareSelector()
        selector.SetRenderer(self.plotter.renderer)
        selector.SetArea(x, y, x, y)
        selector.SetFieldAssociation(vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS)

        selection = selector.Select()
        
        picked_actor = None
        
        if selection and selection.GetNumberOfNodes() > 0:
            selection_node = selection.GetNode(0)
            picked_actor = selection_node.GetProperties().Get(vtk.vtkSelectionNode.PROP())
        
        if not picked_actor or picked_actor not in self.actor_to_face_id:
            return
        
        face_id = self.actor_to_face_id[picked_actor]
        solo = self.combo_render_mode.currentIndex() if hasattr(self, "combo_render_mode") else 0

        # 概览模式：点选面对仅更新选中与 OBB，不加载单面对点云
        if solo == 2:
            face_pair = None
            for tags in self.face_tag_groups:
                if len(tags) == 2 and face_id in tags:
                    face_pair = (tags[0], tags[1])
                    break
            if face_pair:
                left_id, right_id = face_pair
                self.current_face_pair = (left_id, right_id)
                self._apply_solo_mode()
                self.lbl_info.setText(f"概览: 已选中面对 [{left_id}, {right_id}]，OBB 已更新")
                return
        
        # STF 模式（仅 model mode）：点选面切换其在透明面列表中的状态
        if solo == 0 and getattr(self, "stf_mode", False):
            if face_id in self.transparent_face_ids:
                self.transparent_face_ids.discard(face_id)
                action = "移除"
            else:
                self.transparent_face_ids.add(face_id)
                action = "加入"
            self._apply_solo_mode()
            self.lbl_info.setText(f"STF: 面 {face_id} 已{action}透明列表，共 {len(self.transparent_face_ids)} 个")
            return
        
        # 检查这个面是否在配置的面对中（面对 = 恰好 2 个 face tag 的组）
        face_pair = None
        for tags in self.face_tag_groups:
            if len(tags) == 2 and face_id in tags:
                face_pair = (tags[0], tags[1])  # 保持配置文件中的顺序，匹配 npz 命名
                break
        
        if face_pair:
            left_id, right_id = face_pair
            if self._load_and_render_point_cloud(left_id, right_id):
                self._apply_solo_mode()  # 选中面对后更新 solo 模式下的可见性
                self.lbl_info.setText(
                    f"已选中面对 [{left_id}, {right_id}]，已加载点云 (点击面 ID: {face_id})"
                )
                print(f"点选面对 [{left_id}, {right_id}]，已加载 npz 并渲染点云")
            else:
                self.lbl_info.setText(
                    f"点击的面 ID: {face_id}，面对 [{left_id}, {right_id}] 的 npz 未找到或加载失败"
                )
                print(f"未找到 npz 文件: {self.config_name}_{left_id}_{right_id}_result.npz")
        else:
            group_info = ""
            for i, (color, tags) in enumerate(self.highlight_groups):
                if face_id in tags:
                    group_info = f" (属于第 {i+1} 组高亮)"
                    break
            self.lbl_info.setText(f"点击的面 ID: {face_id}{group_info}")
            print(f"点击的面 ID: {face_id}{group_info}，未找到包含此面的面对")


def main():
    app = QApplication(sys.argv)
    window = FaceHighlighterWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
