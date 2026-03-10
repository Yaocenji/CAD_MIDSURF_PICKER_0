import os
import math
import tempfile
import tkinter as tk
from tkinter import ttk

import mitsuba as mi
mi.set_variant("cuda_ad_rgb")

import numpy as np
from PIL import Image, ImageTk
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray


from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopoDS import topods
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopAbs import TopAbs_REVERSED
from OCC.Core.Bnd import Bnd_OBB
from OCC.Core.BRepBndLib import brepbndlib_AddOBB

from occwl.compound import Compound
from occwl.entity_mapper import EntityMapper
from occwl.solid import Solid

# ============================================================
#  配置区 —— 路径 & 文件
# ============================================================
WORK_DIR = r"F:\newdata"
NPZ_FILE = "26_26_193_168_result.npz"

# ============================================================
#  配置区 —— 渲染模式
# ============================================================
# RENDER_MODE 1: 完整模型
# RENDER_MODE 2: 仅渲染左右面 (根据 NPZ 文件名解析的 left_tag, right_tag)
# RENDER_MODE 3: 同 2，但点云拟合成 OBB 体介质，Mitsuba 体渲染 (无点小球)
# RENDER_MODE 4: 仅左右面 + 点云小球，全光栅化；物体面不透明冯氏光照，小球 unlit 按 offset_pred 颜色条着色
RENDER_MODE = 4

# ============================================================
#  配置区 —— 体介质 (RENDER_MODE 3)
# ============================================================
VOL_GRID_NX = 48                      # 体网格分辨率 X
VOL_GRID_NY = 48                      # 体网格分辨率 Y
VOL_GRID_NZ = 48                      # 体网格分辨率 Z
VOL_SIGMA_T_SCALE = 3.0               #  均一密度缩放，越大越不透明
VOL_ALBEDO_GRAY = True                # True=灰度(来自 offset_pred), False=恒 albedo
VOL_ALBEDO_SMOOTH_SIGMA = 0.02        #  高斯平滑系数 (相对网格尺寸)，越大越平滑

# ============================================================
#  配置区 —— 网格化精度 (OCC BRepMesh)
# ============================================================
MESH_LINEAR_DEFLECTION = 0.1
MESH_ANGULAR_DEFLECTION = 0.5
MESH_RELATIVE = False
SWAP_YZ = False                  # True = 导入时交换 Y 轴和 Z 轴 (Z-up → Y-up)

# ============================================================
#  配置区 —— 材质参数 (Mitsuba BSDF)
# ============================================================
# BSDF 类型:
#   "thindielectric" — 薄介质 (玻璃窗效果, 真正的物理半透明)
#   "dielectric"     — 厚介质 (实心玻璃, 有折射)
#   "roughdielectric" — 粗糙介质 (磨砂玻璃)
#   "mask"           — 透明度遮罩 (控制 opacity)
#   "plastic"        — 塑料
BSDF_TYPE = "mask"
BSDF_OPACITY = 0.5                      # 仅 mask 类型: 不透明度 [0,1]
BSDF_INNER_TYPE = "dielectric"           # mask 内层 BSDF 类型
BSDF_DIFFUSE_COLOR = (0.70, 0.70, 0.72)  # 漫反射颜色 RGB
BSDF_INT_IOR = "bk7"                     # 仅 dielectric 系列: 内部折射率
BSDF_EXT_IOR = "air"                     # 仅 dielectric 系列: 外部折射率
BSDF_ROUGHNESS = 0.15                    # 仅 roughdielectric: 粗糙度

# ============================================================
#  配置区 —— 积分器 (路径追踪)
# ============================================================
INTEGRATOR_TYPE = "volpath"   # "path" 或 "volpath" (mask BSDF 推荐 volpath)
INTEGRATOR_MAX_DEPTH = 3     # 最大弹射深度

# ============================================================
#  配置区 —— 相机参数
# ============================================================
CAMERA_FOV = 30.0                      # 垂直视场角 (度)，运行中可用 [ / ] 调整
CAMERA_FOV_MIN = 5.0                   # FOV 下限 (度)
CAMERA_FOV_MAX = 120.0                 # FOV 上限 (度)
CAMERA_FOV_STEP = 2.0                  # [ / ] 键每次调整量 (度)
CAMERA_ORIGIN = None                   # None = 自动; 或 (x, y, z)
CAMERA_TARGET = None                   # None = 自动居中; 或 (x, y, z)
CAMERA_UP = (0.0, 1.0, 0.0)

# ============================================================
#  配置区 —— 天空盒 / 环境光
# ============================================================
# SKYBOX_MODE:
#   "solid"   — 纯色环境光 (使用 SKYBOX_COLOR)
#   "hdri"    — HDRI 环境贴图 (使用 SKYBOX_HDRI_PATH)
SKYBOX_MODE = "solid"
SKYBOX_COLOR = (1.0, 1.0, 1.0)      # solid 模式: 环境光颜色
SKYBOX_INTENSITY = 1                 # 环境光强度缩放
SKYBOX_HDRI_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Texture/HdrOutdoorSnowMountainsEveningClear001/HdrOutdoorSnowMountainsEveningClear001_JPG_8K.jpg")                  # hdri 模式: HDR 文件路径

# ============================================================
#  配置区 —— 主光源 (方向光 / 太阳光)
# ============================================================
KEY_LIGHT_ENABLED = True
KEY_LIGHT_DIRECTION = (0.6, -0.8, -0.5)  # 光线方向 (从光源射向场景)
KEY_LIGHT_INTENSITY = 8.0                # 辐照度强度
KEY_LIGHT_COLOR = (1.0, 1.0, 1.0)      # 光源颜色 RGB

# ============================================================
#  配置区 —— 渲染质量
# ============================================================
SPP_PREVIEW = 1        # 交互拖拽时的低质量采样
SPP_FINAL = 16        # 相机静止后的高质量采样
FILM_WIDTH = 960
FILM_HEIGHT = 720

# ============================================================
#  配置区 —— OptiX AI 降噪 (仅 cuda variant)
# ============================================================
DENOISE_PREVIEW = False           # 预览渲染后降噪 (低 spp 效果提升明显)
DENOISE_FINAL = False             # 高质量渲染后降噪
DENOISE_SCREENSHOT = False        # 截图渲染后降噪

# ============================================================
#  配置区 —— 截图渲染 (按 F12 触发)
# ============================================================
SCREENSHOT_SPP = 32             # 截图采样数 (越高越清晰)
SCREENSHOT_WIDTH = 1920          # 截图分辨率 宽
SCREENSHOT_HEIGHT = 1440         # 截图分辨率 高
SCREENSHOT_DIR = "."             # 截图保存目录 ("." = 脚本所在目录)

# ============================================================
#  配置区 —— 鼠标控制
# ============================================================
MOUSE_ORBIT_SENSITIVITY = 0.35   # 旋转灵敏度 (度/像素)
MOUSE_PAN_SENSITIVITY = 0.002    # 平移灵敏度 (距离比/像素)
MOUSE_ZOOM_FACTOR = 0.08         # 滚轮缩放比例
MOUSE_INVERT_Y = True            # 反转鼠标 Y 轴 (True=上拖抬头)

# ============================================================
#  配置区 —— 光栅化模式材质 (VTK, Tab 切换)
# ============================================================
RASTER_COLOR = (0.55, 0.70, 0.88)  # 模型颜色 RGB
RASTER_OPACITY = 0.05              # 不透明度
RASTER_AMBIENT = 0.15
RASTER_DIFFUSE = 0.7
RASTER_SPECULAR = 0.4
RASTER_SPECULAR_POWER = 30.0
RASTER_EDGE_VISIBLE = False        # 显示线框边
RASTER_BG_COLOR = (1, 1, 1)

# ============================================================
#  配置区 —— 点云显示 (查询点用半透明小球表示)
# ============================================================
POINT_CLOUD_ENABLED = True
POINT_CLOUD_COLOR = (1.0, 1.0, 1.0)   # 小球颜色 RGB
POINT_CLOUD_OPACITY = 0.05             # 不透明度 [0,1]
POINT_CLOUD_RADIUS_SCALE = 0.0008     # 小球半径 = 模型包围盒对角 × 此比例
POINT_CLOUD_MAX_POINTS = 500000        # 最大显示点数 (0=不限制, 过多影响性能)
POINT_CLOUD_SPHERE_RESOLUTION = 3     # 小球细分 (VTK glyph 用, 5≈少面数, 8=默认)
# 光栅化模式下点云加载成功时，用 OBB 线框代替小球
OBB_WIRE_COLOR = (1.0, 0.9, 0.2)     # OBB 线框颜色 RGB
OBB_WIRE_LINE_WIDTH = 1.5            # 线宽

# RENDER_MODE 4 点云颜色条 (offset_pred 0→1 双端颜色，中间线性插值)
POINT_CLOUD_COLOR_BAR_0 = (0.2, 0.4, 0.9)   # offset_pred=0 时颜色 RGB
POINT_CLOUD_COLOR_BAR_1 = (0.9, 0.3, 0.2)   # offset_pred=1 时颜色 RGB

# ============================================================
#  配置区 —— 窗口
# ============================================================
WINDOW_TITLE = "STEP Model Viewer"


# ============================================================
#  工具函数
# ============================================================

def compute_obb_for_faces(step_path: str, left_tag: int, right_tag: int):
    """根据左右面 idx 计算二者合并的 Oriented Bounding Box (OBB) 参数。
    与 point_cloud_4.py 中 PointCloudGenerator.compute_obb 保持一致。"""
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


def obb_normalized_to_world(points_01, obb_origin, obb_x_vec, obb_y_vec, obb_z_vec):
    """将 OBB 归一化空间 [0,1]³ 中的点变换回世界坐标。
    points_01: (N, 3) float, 每行 [u,v,w] 且 u,v,w in [0,1]"""
    pts = np.asarray(points_01, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    return obb_origin + (
        pts[:, 0:1] * obb_x_vec + pts[:, 1:2] * obb_y_vec + pts[:, 2:3] * obb_z_vec
    )


def world_to_obb_normalized(points_world, obb_origin, obb_x_vec, obb_y_vec, obb_z_vec):
    """将世界坐标点变换到 OBB 归一化空间 [0,1]³。"""
    pts = np.asarray(points_world, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    p_rel = pts - obb_origin
    xx = np.dot(obb_x_vec, obb_x_vec)
    yy = np.dot(obb_y_vec, obb_y_vec)
    zz = np.dot(obb_z_vec, obb_z_vec)
    u = np.clip(np.dot(p_rel, obb_x_vec) / max(xx, 1e-12), 0, 1)
    v = np.clip(np.dot(p_rel, obb_y_vec) / max(yy, 1e-12), 0, 1)
    w = np.clip(np.dot(p_rel, obb_z_vec) / max(zz, 1e-12), 0, 1)
    return np.column_stack([u, v, w])


def point_cloud_to_volume_grid(point_cloud_aos, obb_origin, obb_x, obb_y, obb_z,
                               nx, ny, nz):
    """将点云拟合成 OBB 内的 3D 体网格。
    密度 sigma_t 均一；颜色 albedo 由 offset_pred 拟合，三线性样条 + 高斯平滑保证连续。"""
    from scipy.ndimage import gaussian_filter

    weight_sum = np.zeros((nz, ny, nx), dtype=np.float32)
    albedo_sum = np.zeros((nz, ny, nx, 3), dtype=np.float32)
    pts = np.array([r["point"] for r in point_cloud_aos], dtype=np.float64)
    uvws = world_to_obb_normalized(pts, obb_origin, obb_x, obb_y, obb_z)

    uu = np.clip(uvws[:, 0] * (nx - 1), 0, nx - 1.0001)
    vv = np.clip(uvws[:, 1] * (ny - 1), 0, ny - 1.0001)
    ww = np.clip(uvws[:, 2] * (nz - 1), 0, nz - 1.0001)
    ix0 = np.floor(uu).astype(np.int32)
    iy0 = np.floor(vv).astype(np.int32)
    iz0 = np.floor(ww).astype(np.int32)
    ix0 = np.clip(ix0, 0, nx - 2)
    iy0 = np.clip(iy0, 0, ny - 2)
    iz0 = np.clip(iz0, 0, nz - 2)
    ix1 = ix0 + 1
    iy1 = iy0 + 1
    iz1 = iz0 + 1
    tx = uu - ix0
    ty = vv - iy0
    tz = ww - iz0

    for i, r in enumerate(point_cloud_aos):
        if r.get("validity_pred", 1) < 0.5:
            continue
        o = max(0, min(1, float(r.get("offset_pred", 0.5))))
        v000 = (1 - tx[i]) * (1 - ty[i]) * (1 - tz[i])
        v001 = (1 - tx[i]) * (1 - ty[i]) * tz[i]
        v010 = (1 - tx[i]) * ty[i] * (1 - tz[i])
        v011 = (1 - tx[i]) * ty[i] * tz[i]
        v100 = tx[i] * (1 - ty[i]) * (1 - tz[i])
        v101 = tx[i] * (1 - ty[i]) * tz[i]
        v110 = tx[i] * ty[i] * (1 - tz[i])
        v111 = tx[i] * ty[i] * tz[i]
        albedo_sum[iz0[i], iy0[i], ix0[i], :] += v000 * o
        albedo_sum[iz1[i], iy0[i], ix0[i], :] += v001 * o
        albedo_sum[iz0[i], iy1[i], ix0[i], :] += v010 * o
        albedo_sum[iz1[i], iy1[i], ix0[i], :] += v011 * o
        albedo_sum[iz0[i], iy0[i], ix1[i], :] += v100 * o
        albedo_sum[iz1[i], iy0[i], ix1[i], :] += v101 * o
        albedo_sum[iz0[i], iy1[i], ix1[i], :] += v110 * o
        albedo_sum[iz1[i], iy1[i], ix1[i], :] += v111 * o
        weight_sum[iz0[i], iy0[i], ix0[i]] += v000
        weight_sum[iz1[i], iy0[i], ix0[i]] += v001
        weight_sum[iz0[i], iy1[i], ix0[i]] += v010
        weight_sum[iz1[i], iy1[i], ix0[i]] += v011
        weight_sum[iz0[i], iy0[i], ix1[i]] += v100
        weight_sum[iz1[i], iy0[i], ix1[i]] += v101
        weight_sum[iz0[i], iy1[i], ix1[i]] += v110
        weight_sum[iz1[i], iy1[i], ix1[i]] += v111

    m = np.maximum(weight_sum, 1e-9)
    albedo_sum[:, :, :, 0] /= m
    albedo_sum[:, :, :, 1] /= m
    albedo_sum[:, :, :, 2] /= m

    sigma_t = np.ones((nz, ny, nx, 1), dtype=np.float32)

    smooth = max(1.0, min(nx, ny, nz) * VOL_ALBEDO_SMOOTH_SIGMA)
    albedo_sum[:, :, :, 0] = gaussian_filter(albedo_sum[:, :, :, 0], sigma=smooth, mode="constant", cval=0.5)
    albedo_sum[:, :, :, 1] = gaussian_filter(albedo_sum[:, :, :, 1], sigma=smooth, mode="constant", cval=0.5)
    albedo_sum[:, :, :, 2] = gaussian_filter(albedo_sum[:, :, :, 2], sigma=smooth, mode="constant", cval=0.5)
    albedo_sum = np.clip(albedo_sum, 0, 1).astype(np.float32)

    return sigma_t, albedo_sum


def load_query_points_aos(npz_path: str = None, step_path: str = None,
                          left_tag: int = None, right_tag: int = None,
                          work_dir: str = None, to_world_space: bool = True):
    """从 NPZ 文件读取查询点坐标、预测 offset、预测 validity，整理为 AOS 数据列表。

    字段说明 (见 renderer文档.md):
      - query_points: (N, 3) float32，查询点 xyz 坐标（归一化到 OBB [0,1]³）
      - offset_pred:  (N,)   float32，Stage2 预测 offset
      - validity_pred:(N,)   float32，Stage1 预测有效性 (1=有效)

    若 to_world_space=True（默认），需提供 step_path 与 left_tag、right_tag 以计算 OBB，
    并将 point 从 OBB [0,1] 空间变换回世界坐标。未提供时从 npz 文件名解析。

    返回: (aos, obb_center, obb_vectors)。obb_vectors=(origin,x,y,z) 用于 OBB 线框，无则为 None
    """
    work_dir = work_dir or WORK_DIR
    if npz_path is None:
        npz_path = os.path.join(work_dir, NPZ_FILE)
    data = np.load(npz_path)
    query_points = np.asarray(data["query_points"], dtype=np.float64)
    offset_pred = data["offset_pred"]
    validity_pred = data["validity_pred"]

    points_to_use = query_points
    obb_center = None
    if to_world_space:
        if step_path is None or left_tag is None or right_tag is None:
            folder, file_id, lt, rt = parse_npz_filename(os.path.basename(npz_path))
            step_path = step_path or os.path.join(work_dir, folder, f"{file_id}.step")
            left_tag = left_tag if left_tag is not None else lt
            right_tag = right_tag if right_tag is not None else rt
        obb_origin, obb_x, obb_y, obb_z = compute_obb_for_faces(step_path, left_tag, right_tag)
        points_to_use = obb_normalized_to_world(query_points, obb_origin, obb_x, obb_y, obb_z)
        obb_center = obb_origin + 0.5 * obb_x + 0.5 * obb_y + 0.5 * obb_z
        obb_vectors = (obb_origin, obb_x, obb_y, obb_z)
    else:
        obb_vectors = None

    aos = []
    for i in range(len(query_points)):
        pt = points_to_use[i].tolist()
        aos.append({
            "point": pt,
            "offset_pred": float(offset_pred[i]),
            "validity_pred": float(validity_pred[i]),
        })
    return aos, obb_center, obb_vectors


def parse_npz_filename(npz_name: str):
    base = npz_name.replace("_result.npz", "")
    parts = base.split("_")
    folder, file_id = parts[0], parts[1]
    left_tag, right_tag = int(parts[2]), int(parts[3])
    return folder, file_id, left_tag, right_tag


def load_step(step_path: str):
    reader = STEPControl_Reader()
    status = reader.ReadFile(step_path)
    if status != 1:
        raise RuntimeError(f"无法读取 STEP 文件: {step_path}  (status={status})")
    reader.TransferRoots()
    return reader.OneShape()


def shape_to_numpy(shape):
    """将 OCC Shape 三角化并返回 (vertices, faces, normals) numpy 数组。
    自动修正 Face Orientation 导致的绕序反转。"""
    BRepMesh_IncrementalMesh(
        shape, MESH_LINEAR_DEFLECTION, MESH_RELATIVE,
        MESH_ANGULAR_DEFLECTION, True,
    )

    all_verts = []
    all_faces = []
    offset = 0

    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = topods.Face(explorer.Current())
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation(face, loc)
        if tri is None:
            explorer.Next()
            continue

        reversed_face = (face.Orientation() == TopAbs_REVERSED)
        trsf = loc.Transformation()
        nb_n = tri.NbNodes()
        nb_t = tri.NbTriangles()

        for i in range(1, nb_n + 1):
            pt = tri.Node(i)
            pt.Transform(trsf)
            all_verts.append((pt.X(), pt.Y(), pt.Z()))

        for i in range(1, nb_t + 1):
            t = tri.Triangle(i)
            n1, n2, n3 = t.Get()
            i0 = n1 - 1 + offset
            i1 = n2 - 1 + offset
            i2 = n3 - 1 + offset
            if reversed_face:
                i1, i2 = i2, i1
            all_faces.append((i0, i1, i2))

        offset += nb_n
        explorer.Next()

    if not all_verts:
        raise RuntimeError("三角化后未获得任何顶点。")

    vertices = np.array(all_verts, dtype=np.float32)
    faces = np.array(all_faces, dtype=np.uint32)

    if SWAP_YZ:
        vertices[:, [1, 2]] = vertices[:, [2, 1]]

    normals = _compute_vertex_normals(vertices, faces)

    return vertices, faces, normals


def _triangulate_face_to_verts_faces(face_shape, offset, all_verts, all_faces):
    """将单个 Face 三角化并追加到 all_verts, all_faces，返回新 offset。"""
    BRepMesh_IncrementalMesh(
        face_shape, MESH_LINEAR_DEFLECTION, MESH_RELATIVE,
        MESH_ANGULAR_DEFLECTION, True,
    )
    loc = TopLoc_Location()
    tri = BRep_Tool.Triangulation(face_shape, loc)
    if tri is None:
        return offset
    reversed_face = (topods.Face(face_shape).Orientation() == TopAbs_REVERSED)
    trsf = loc.Transformation()
    nb_n = tri.NbNodes()
    nb_t = tri.NbTriangles()
    for i in range(1, nb_n + 1):
        pt = tri.Node(i)
        pt.Transform(trsf)
        all_verts.append((pt.X(), pt.Y(), pt.Z()))
    for i in range(1, nb_t + 1):
        t = tri.Triangle(i)
        n1, n2, n3 = t.Get()
        i0 = n1 - 1 + offset
        i1 = n2 - 1 + offset
        i2 = n3 - 1 + offset
        if reversed_face:
            i1, i2 = i2, i1
        all_faces.append((i0, i1, i2))
    return offset + nb_n


def shape_to_numpy_left_right_only(step_path: str, left_tag: int, right_tag: int):
    """仅三角化并返回左右面的 (vertices, faces, normals)。与 shape_to_numpy 逻辑一致。"""
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

    all_verts = []
    all_faces = []
    offset = 0
    offset = _triangulate_face_to_verts_faces(left_shape, offset, all_verts, all_faces)
    offset = _triangulate_face_to_verts_faces(right_shape, offset, all_verts, all_faces)

    if not all_verts:
        raise RuntimeError("左右面三角化后未获得任何顶点。")

    vertices = np.array(all_verts, dtype=np.float32)
    faces = np.array(all_faces, dtype=np.uint32)

    if SWAP_YZ:
        vertices[:, [1, 2]] = vertices[:, [2, 1]]

    normals = _compute_vertex_normals(vertices, faces)
    return vertices, faces, normals


def _compute_vertex_normals(vertices, faces):
    """从面片计算面积加权顶点法线, 并修复零法线。"""
    normals = np.zeros_like(vertices)

    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)

    for k in range(3):
        np.add.at(normals, faces[:, k], face_normals)

    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    degenerate = (lengths < 1e-12).squeeze()
    lengths[degenerate] = 1.0
    normals /= lengths

    if np.any(degenerate):
        normals[degenerate] = [0.0, 1.0, 0.0]
        n_bad = int(np.sum(degenerate))
        print(f"[WARN] {n_bad} 个顶点法线退化，已修复为默认方向")

    return normals


def export_ply(vertices, faces, normals, path):
    """将顶点、法线和面片写入 PLY 文件。"""
    nv, nf = len(vertices), len(faces)
    with open(path, "wb") as f:
        header = (
            f"ply\n"
            f"format binary_little_endian 1.0\n"
            f"element vertex {nv}\n"
            f"property float x\n"
            f"property float y\n"
            f"property float z\n"
            f"property float nx\n"
            f"property float ny\n"
            f"property float nz\n"
            f"element face {nf}\n"
            f"property list uchar int vertex_indices\n"
            f"end_header\n"
        )
        f.write(header.encode("ascii"))
        vn = np.hstack([vertices.astype("<f4"), normals.astype("<f4")])
        f.write(vn.tobytes())
        for tri in faces:
            f.write(np.uint8(3).tobytes())
            f.write(tri.astype("<i4").tobytes())


def build_bsdf_dict():
    """根据配置区构建 BSDF 字典, 自动用 twosided 包裹单面材质。"""
    r, g, b = BSDF_DIFFUSE_COLOR

    if BSDF_TYPE == "mask":
        inner = _make_inner_bsdf(BSDF_INNER_TYPE, r, g, b)
        core = {
            "type": "mask",
            "opacity": {"type": "rgb", "value": [BSDF_OPACITY] * 3},
            "nested_bsdf": {"type": "twosided", "material": inner},
        }
        return core
    elif BSDF_TYPE == "thindielectric":
        return {
            "type": "thindielectric",
            "int_ior": BSDF_INT_IOR,
            "ext_ior": BSDF_EXT_IOR,
        }
    elif BSDF_TYPE == "dielectric":
        return {
            "type": "dielectric",
            "int_ior": BSDF_INT_IOR,
            "ext_ior": BSDF_EXT_IOR,
        }
    elif BSDF_TYPE == "roughdielectric":
        return {
            "type": "roughdielectric",
            "int_ior": BSDF_INT_IOR,
            "ext_ior": BSDF_EXT_IOR,
            "alpha": BSDF_ROUGHNESS,
        }
    elif BSDF_TYPE == "plastic":
        return {"type": "twosided", "material": _make_inner_bsdf("plastic", r, g, b)}
    else:
        core = {"type": "diffuse", "reflectance": {"type": "rgb", "value": [r, g, b]}}
        return {"type": "twosided", "material": core}


def _make_inner_bsdf(bsdf_type, r, g, b):
    if bsdf_type == "plastic":
        return {
            "type": "plastic",
            "diffuse_reflectance": {"type": "rgb", "value": [r, g, b]},
            "int_ior": 1.5,
        }
    return {"type": "diffuse", "reflectance": {"type": "rgb", "value": [r, g, b]}}


def build_point_cloud_bsdf_dict(offset_pred=None):
    """根据配置区构建点云小球的 BSDF。offset_pred 为 [0,1] 时用灰度着色，否则用 POINT_CLOUD_COLOR。"""
    if offset_pred is not None:
        v = max(0.0, min(1.0, float(offset_pred)))
        r, g, b = v, v, v
    else:
        r, g, b = POINT_CLOUD_COLOR
    inner = {"type": "diffuse", "reflectance": {"type": "rgb", "value": [r, g, b]}}
    return {
        "type": "mask",
        "opacity": {"type": "rgb", "value": [POINT_CLOUD_OPACITY] * 3},
        "nested_bsdf": {"type": "twosided", "material": inner},
    }


def build_emitter_dict():
    """根据 SKYBOX_MODE 构建环境光字典。"""
    if SKYBOX_MODE == "hdri" and SKYBOX_HDRI_PATH:
        return {
            "type": "envmap",
            "filename": SKYBOX_HDRI_PATH,
            "scale": SKYBOX_INTENSITY,
        }
    r, g, b = SKYBOX_COLOR
    return {
        "type": "constant",
        "radiance": {
            "type": "rgb",
            "value": [r * SKYBOX_INTENSITY, g * SKYBOX_INTENSITY, b * SKYBOX_INTENSITY],
        },
    }


def compute_auto_camera(vertices, target_override=None):
    """从网格包围盒自动计算相机位姿。target_override 为 OBB 中心时，相机将聚焦该点。"""
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0
    extent = np.linalg.norm(bbox_max - bbox_min)
    dist = extent / (2.0 * math.tan(math.radians(CAMERA_FOV) / 2.0)) * 1.2

    if CAMERA_TARGET is not None:
        target = CAMERA_TARGET
    elif target_override is not None:
        t = np.asarray(target_override)
        target = tuple(t.tolist())
    else:
        target = tuple(center.tolist())

    if CAMERA_ORIGIN is None:
        t_arr = np.array(target, dtype=np.float64)
        origin = tuple((t_arr + np.array([0, 0, dist])).tolist())
    else:
        origin = CAMERA_ORIGIN

    return origin, target


def _obb_to_world_matrix(obb_origin, obb_x, obb_y, obb_z):
    """构建将 [0,1]³ 映射到 OBB 的 4x4 变换矩阵 (用于 volume grid)。"""
    m = np.eye(4, dtype=np.float32)
    m[0:3, 0] = obb_x
    m[0:3, 1] = obb_y
    m[0:3, 2] = obb_z
    m[0:3, 3] = obb_origin
    return mi.ScalarTransform4f(m)


def _obb_cube_to_world_matrix(obb_origin, obb_x, obb_y, obb_z):
    """构建将 [-1,1]³ cube 映射到 OBB 的 4x4 变换矩阵。"""
    center = obb_origin + 0.5 * obb_x + 0.5 * obb_y + 0.5 * obb_z
    m = np.eye(4, dtype=np.float32)
    m[0:3, 0] = 0.5 * obb_x
    m[0:3, 1] = 0.5 * obb_y
    m[0:3, 2] = 0.5 * obb_z
    m[0:3, 3] = center
    return mi.ScalarTransform4f(m)


def build_scene_dict(ply_path, origin, target,
                     film_w=None, film_h=None,
                     point_cloud_aos=None, scene_extent=None, fov=None,
                     render_mode=1, obb_vectors=None):
    """构建完整的 Mitsuba 场景字典。"""
    bsdf = build_bsdf_dict()
    emitter = build_emitter_dict()
    fw = film_w or FILM_WIDTH
    fh = film_h or FILM_HEIGHT

    use_volume_mode = (
        render_mode == 3 and obb_vectors is not None and
        point_cloud_aos is not None and len(point_cloud_aos) > 0
    )
    scene = {
        "type": "scene",
        "integrator": {
            "type": "volpath" if use_volume_mode else INTEGRATOR_TYPE,
            "max_depth": INTEGRATOR_MAX_DEPTH,
        },
        "sensor": {
            "type": "perspective",
            "fov": fov if fov is not None else CAMERA_FOV,
            "to_world": mi.ScalarTransform4f.look_at(
                origin=origin, target=target, up=CAMERA_UP,
            ),
            "film": {
                "type": "hdrfilm",
                "width": fw,
                "height": fh,
                "pixel_format": "rgb",
                "rfilter": {"type": "box"},
            },
            "sampler": {"type": "independent", "sample_count": SPP_PREVIEW},
        },
        "env_emitter": emitter,
        "mesh": {
            "type": "ply",
            "filename": ply_path,
            "bsdf": bsdf,
        },
    }
    if use_volume_mode:
        obb_origin, obb_x, obb_y, obb_z = obb_vectors
        sigma_t, albedo = point_cloud_to_volume_grid(
            point_cloud_aos, obb_origin, obb_x, obb_y, obb_z,
            VOL_GRID_NX, VOL_GRID_NY, VOL_GRID_NZ,
        )
        vol_to_world = _obb_to_world_matrix(obb_origin, obb_x, obb_y, obb_z)
        cube_to_world = _obb_cube_to_world_matrix(obb_origin, obb_x, obb_y, obb_z)
        sigma_t_tensor = mi.TensorXf(sigma_t)
        if VOL_ALBEDO_GRAY:
            albedo_tensor = mi.TensorXf(albedo)
            albedo_vol = {
                "type": "gridvolume",
                "grid": mi.VolumeGrid(albedo_tensor),
                "to_world": vol_to_world,
                "raw": True,
            }
        else:
            albedo_vol = 0.85
        scene["vol_box"] = {
            "type": "cube",
            "to_world": cube_to_world,
            "bsdf": {"type": "null"},
            "interior": {
                "type": "heterogeneous",
                "sigma_t": {
                    "type": "gridvolume",
                    "grid": mi.VolumeGrid(sigma_t_tensor),
                    "to_world": vol_to_world,
                    "raw": True,
                },
                "albedo": albedo_vol,
                "scale": VOL_SIGMA_T_SCALE,
                "phase": {"type": "isotropic"},
            },
        }
    elif (POINT_CLOUD_ENABLED and point_cloud_aos is not None and
            len(point_cloud_aos) > 0 and scene_extent is not None and scene_extent > 1e-9):
        radius = scene_extent * POINT_CLOUD_RADIUS_SCALE
        n_max = POINT_CLOUD_MAX_POINTS if POINT_CLOUD_MAX_POINTS > 0 else len(point_cloud_aos)
        items = point_cloud_aos
        if len(items) > n_max:
            step = max(1, len(items) // n_max)
            items = items[::step][:n_max]
        for i, r in enumerate(items):
            pt = r["point"]
            offset_pred = r.get("offset_pred", None)
            pt_bsdf = build_point_cloud_bsdf_dict(offset_pred=offset_pred)
            scene[f"point_sphere_{i}"] = {
                "type": "sphere",
                "radius": radius,
                "to_world": mi.ScalarTransform4f.translate([float(pt[0]), float(pt[1]), float(pt[2])]),
                "bsdf": pt_bsdf,
            }

    if KEY_LIGHT_ENABLED:
        dr, dg, db = KEY_LIGHT_COLOR
        dx, dy, dz = KEY_LIGHT_DIRECTION
        scene["key_light"] = {
            "type": "directional",
            "direction": [dx, dy, dz],
            "irradiance": {
                "type": "rgb",
                "value": [dr * KEY_LIGHT_INTENSITY,
                          dg * KEY_LIGHT_INTENSITY,
                          db * KEY_LIGHT_INTENSITY],
            },
        }

    return scene


_denoiser_cache = {}

def _get_denoiser(h, w):
    """缓存 OptixDenoiser 实例 (按分辨率)。"""
    key = (h, w)
    if key not in _denoiser_cache:
        _denoiser_cache[key] = mi.OptixDenoiser(
            input_size=[w, h], albedo=False, normals=False, temporal=False,
        )
    return _denoiser_cache[key]


def render_image(scene, spp, seed=0, denoise=False):
    """渲染并返回 uint8 RGB numpy 数组。denoise=True 时启用 OptiX AI 降噪。"""
    img = mi.render(scene, spp=spp, seed=seed)

    if denoise and "cuda" in mi.variant():
        denoiser = _get_denoiser(img.shape[0], img.shape[1])
        img = denoiser(img)

    bmp = mi.util.convert_to_bitmap(img)
    arr = np.array(bmp, copy=False)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    return arr


def build_vtk_obb_wireframe_polydata(obb_origin, obb_x, obb_y, obb_z):
    """从 OBB 参数构建 12 条边的线框 vtkPolyData。"""
    corners = []
    for i in [0, 1]:
        for j in [0, 1]:
            for k in [0, 1]:
                pt = obb_origin + i * obb_x + j * obb_y + k * obb_z
                corners.append(pt)
    corners = np.array(corners, dtype=np.float64)
    # 立方体 12 条边: 底面 4 + 顶面 4 + 竖边 4
    edges = [
        (0, 1), (1, 3), (3, 2), (2, 0),  # z=0 底面
        (4, 5), (5, 7), (7, 6), (6, 4),  # z=1 顶面
        (0, 4), (1, 5), (2, 6), (3, 7),  # 竖边
    ]
    pts_vtk = vtk.vtkPoints()
    pts_vtk.SetData(numpy_to_vtk(corners, deep=True))
    lines = vtk.vtkCellArray()
    for i, j in edges:
        ids = vtk.vtkIdList()
        ids.InsertNextId(i)
        ids.InsertNextId(j)
        lines.InsertNextCell(ids)
    pd = vtk.vtkPolyData()
    pd.SetPoints(pts_vtk)
    pd.SetLines(lines)
    return pd


def build_vtk_point_cloud_polydata(pts_world, radius, scene_extent):
    """将世界坐标点云构建为带球体 glyph 的 vtkPolyData。"""
    radius = radius if radius > 0 else scene_extent * POINT_CLOUD_RADIUS_SCALE
    pts = np.asarray(pts_world, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    n_max = POINT_CLOUD_MAX_POINTS if POINT_CLOUD_MAX_POINTS > 0 else len(pts)
    if len(pts) > n_max:
        step = max(1, len(pts) // n_max)
        pts = pts[::step][:n_max]

    pts_vtk = vtk.vtkPoints()
    pts_vtk.SetData(numpy_to_vtk(pts, deep=True))

    pc_pd = vtk.vtkPolyData()
    pc_pd.SetPoints(pts_vtk)

    sphere = vtk.vtkSphereSource()
    sphere.SetRadius(radius)
    res = max(3, POINT_CLOUD_SPHERE_RESOLUTION)
    sphere.SetPhiResolution(res)
    sphere.SetThetaResolution(res)
    sphere.Update()

    glyph = vtk.vtkGlyph3D()
    glyph.SetInputData(pc_pd)
    glyph.SetSourceConnection(sphere.GetOutputPort())
    glyph.ScalingOff()
    glyph.Update()
    return glyph.GetOutput()


def build_vtk_point_cloud_polydata_with_colors(point_cloud_aos, radius, scene_extent,
                                               color_bar_0, color_bar_1):
    """将点云构建为带球体 glyph 的 vtkPolyData，每点颜色由 offset_pred 在 color_bar_0~1 间线性插值。"""
    if not point_cloud_aos or len(point_cloud_aos) == 0:
        empty_pd = vtk.vtkPolyData()
        empty_pd.SetPoints(vtk.vtkPoints())
        return empty_pd
    radius = radius if radius > 0 else scene_extent * POINT_CLOUD_RADIUS_SCALE
    c0 = np.array(color_bar_0, dtype=np.float64)
    c1 = np.array(color_bar_1, dtype=np.float64)
    pts = []
    colors = []
    n_max = POINT_CLOUD_MAX_POINTS if POINT_CLOUD_MAX_POINTS > 0 else len(point_cloud_aos)
    items = point_cloud_aos[:n_max] if n_max >= len(point_cloud_aos) else point_cloud_aos[::max(1, len(point_cloud_aos) // n_max)][:n_max]
    for r in items:
        pt = r["point"]
        t = max(0.0, min(1.0, float(r.get("offset_pred", 0.5))))
        pts.append(pt)
        rgb = (1.0 - t) * c0 + t * c1
        rgb = np.clip(rgb, 0, 1)
        colors.append((int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)))
    pts = np.array(pts, dtype=np.float64)
    colors = np.array(colors, dtype=np.uint8)

    pts_vtk = vtk.vtkPoints()
    pts_vtk.SetData(numpy_to_vtk(pts, deep=True))
    pc_pd = vtk.vtkPolyData()
    pc_pd.SetPoints(pts_vtk)
    # 使用 (N,3) 形状以便 numpy_to_vtk 正确识别为 RGB 三分量
    color_arr = numpy_to_vtk(colors, deep=True)
    color_arr.SetName("Colors")
    pc_pd.GetPointData().SetScalars(color_arr)

    sphere = vtk.vtkSphereSource()
    sphere.SetRadius(radius)
    res = max(3, POINT_CLOUD_SPHERE_RESOLUTION)
    sphere.SetPhiResolution(res)
    sphere.SetThetaResolution(res)
    sphere.Update()

    glyph = vtk.vtkGlyph3D()
    glyph.SetInputData(pc_pd)
    glyph.SetSourceConnection(sphere.GetOutputPort())
    glyph.ScalingOff()
    glyph.Update()
    return glyph.GetOutput()


def build_vtk_polydata(vertices, faces, normals):
    """从 numpy 数据构建 vtkPolyData。"""
    points = vtk.vtkPoints()
    pts_vtk = numpy_to_vtk(vertices.astype(np.float64), deep=True)
    points.SetData(pts_vtk)

    n_faces = len(faces)
    conn = np.empty((n_faces, 4), dtype=np.int64)
    conn[:, 0] = 3
    conn[:, 1:] = faces.astype(np.int64)
    cells = vtk.vtkCellArray()
    vtk_id_arr = numpy_to_vtkIdTypeArray(conn.ravel(), deep=True)
    cells.SetCells(n_faces, vtk_id_arr)

    pd = vtk.vtkPolyData()
    pd.SetPoints(points)
    pd.SetPolys(cells)

    nrm_vtk = numpy_to_vtk(normals.astype(np.float64), deep=True)
    nrm_vtk.SetName("Normals")
    pd.GetPointData().SetNormals(nrm_vtk)

    return pd


def raster_render_to_array(polydata, origin, target, up, fov, width, height,
                           obb_wireframe_polydata=None):
    """用 VTK 离屏光栅化渲染, 返回 uint8 RGB numpy 数组。
    点云加载成功时传入 obb_wireframe_polydata，用 OBB 线框代替点云小球。"""
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    prop = actor.GetProperty()
    prop.SetColor(RASTER_COLOR)
    prop.SetOpacity(RASTER_OPACITY)
    prop.SetAmbient(RASTER_AMBIENT)
    prop.SetDiffuse(RASTER_DIFFUSE)
    prop.SetSpecular(RASTER_SPECULAR)
    prop.SetSpecularPower(RASTER_SPECULAR_POWER)
    if RASTER_EDGE_VISIBLE:
        prop.EdgeVisibilityOn()

    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)

    if POINT_CLOUD_ENABLED and obb_wireframe_polydata is not None:
        wire_mapper = vtk.vtkPolyDataMapper()
        wire_mapper.SetInputData(obb_wireframe_polydata)
        wire_actor = vtk.vtkActor()
        wire_actor.SetMapper(wire_mapper)
        wire_prop = wire_actor.GetProperty()
        wire_prop.SetColor(OBB_WIRE_COLOR)
        wire_prop.SetLineWidth(OBB_WIRE_LINE_WIDTH)
        wire_prop.SetOpacity(1.0)
        wire_prop.SetAmbient(0.5)
        wire_prop.SetDiffuse(0.5)
        renderer.AddActor(wire_actor)
    renderer.SetBackground(RASTER_BG_COLOR)

    cam = renderer.GetActiveCamera()
    cam.SetPosition(origin)
    cam.SetFocalPoint(target)
    cam.SetViewUp(up)
    cam.SetViewAngle(fov)
    renderer.ResetCameraClippingRange()

    rw = vtk.vtkRenderWindow()
    rw.SetOffScreenRendering(1)
    rw.AddRenderer(renderer)
    rw.SetSize(width, height)
    rw.SetAlphaBitPlanes(1)
    rw.SetMultiSamples(0)
    renderer.SetUseDepthPeeling(1)
    renderer.SetMaximumNumberOfPeels(100)
    renderer.SetOcclusionRatio(0.0)
    rw.Render()

    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(rw)
    w2i.SetInputBufferTypeToRGB()
    w2i.Update()
    vtk_img = w2i.GetOutput()

    dims = vtk_img.GetDimensions()
    flat = np.frombuffer(
        vtk_img.GetPointData().GetScalars(), dtype=np.uint8
    ).reshape(dims[1], dims[0], -1)
    return np.flipud(flat[:, :, :3]).copy()


def raster_render_mode4_to_array(mesh_polydata, point_sphere_polydata,
                                  origin, target, up, fov, width, height):
    """Mode 4 专用: 网格不透明冯氏光照 + 点云小球 unlit 按 offset_pred 颜色条着色。"""
    # 网格: 不透明、冯氏光照
    mesh_mapper = vtk.vtkPolyDataMapper()
    mesh_mapper.SetInputData(mesh_polydata)
    mesh_actor = vtk.vtkActor()
    mesh_actor.SetMapper(mesh_mapper)
    mesh_prop = mesh_actor.GetProperty()
    mesh_prop.SetColor(RASTER_COLOR)
    mesh_prop.SetOpacity(1.0)
    mesh_prop.SetAmbient(RASTER_AMBIENT)
    mesh_prop.SetDiffuse(RASTER_DIFFUSE)
    mesh_prop.SetSpecular(RASTER_SPECULAR)
    mesh_prop.SetSpecularPower(RASTER_SPECULAR_POWER)
    if RASTER_EDGE_VISIBLE:
        mesh_prop.EdgeVisibilityOn()

    # 点云小球: unlit (ambient=1 即纯色), 使用每点颜色
    sphere_mapper = vtk.vtkPolyDataMapper()
    sphere_mapper.SetInputData(point_sphere_polydata)
    sphere_mapper.SetColorModeToDirectScalars()
    sphere_mapper.SetScalarModeToUsePointData()
    sphere_mapper.ScalarVisibilityOn()
    # 显式指定使用 "Colors" 数组，并设为直接 RGB 模式
    sphere_mapper.SelectColorArray("Colors")
    sphere_actor = vtk.vtkActor()
    sphere_actor.SetMapper(sphere_mapper)
    sphere_prop = sphere_actor.GetProperty()
    sphere_prop.SetAmbient(1.0)
    sphere_prop.SetDiffuse(0.0)
    sphere_prop.SetSpecular(0.0)
    sphere_prop.SetOpacity(1.0)

    renderer = vtk.vtkRenderer()
    renderer.AddActor(mesh_actor)
    if point_sphere_polydata.GetNumberOfPoints() > 0:
        renderer.AddActor(sphere_actor)
    renderer.SetBackground(RASTER_BG_COLOR)

    cam = renderer.GetActiveCamera()
    cam.SetPosition(origin)
    cam.SetFocalPoint(target)
    cam.SetViewUp(up)
    cam.SetViewAngle(fov)
    renderer.ResetCameraClippingRange()

    rw = vtk.vtkRenderWindow()
    rw.SetOffScreenRendering(1)
    rw.AddRenderer(renderer)
    rw.SetSize(width, height)
    rw.SetMultiSamples(4)
    rw.Render()

    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(rw)
    w2i.SetInputBufferTypeToRGB()
    w2i.Update()
    vtk_img = w2i.GetOutput()
    dims = vtk_img.GetDimensions()
    flat = np.frombuffer(
        vtk_img.GetPointData().GetScalars(), dtype=np.uint8
    ).reshape(dims[1], dims[0], -1)
    return np.flipud(flat[:, :, :3]).copy()



# ============================================================
#  参数面板 UI
# ============================================================

def _make_slider_row(parent, label, from_, to, resolution, get_val, set_global_and_refresh, var_name):
    f = ttk.Frame(parent)
    ttk.Label(f, text=label, width=18).pack(side=tk.LEFT, padx=(0, 4))
    v = tk.DoubleVar(value=get_val())
    s = ttk.Scale(f, from_=from_, to=to, variable=v, orient=tk.HORIZONTAL, length=140, command=lambda _: None)
    s.pack(side=tk.LEFT, fill=tk.X, expand=True)
    e = ttk.Entry(f, width=6)
    e.pack(side=tk.LEFT, padx=(4, 0))
    def _on_scale(v):
        val = float(v)
        globals()[var_name] = val
        e.delete(0, tk.END)
        e.insert(0, f"{val:.3f}")
        set_global_and_refresh()
    def _on_entry(ev=None):
        try:
            val = float(e.get())
            val = max(from_, min(to, val))
            globals()[var_name] = val
            v.set(val)
            set_global_and_refresh()
        except ValueError:
            pass
    s.configure(command=lambda x: _on_scale(x))
    e.insert(0, f"{get_val():.3f}")
    e.bind("<Return>", _on_entry)
    return f, v, e, _on_scale


class ParamsPanel:
    """运行时可调参数面板 (Toplevel)，修改后触发重渲染。"""

    def __init__(self, parent, viewer):
        self.viewer = viewer
        self.win = tk.Toplevel(parent)
        self.win.title("参数调节")
        self.win.geometry("400x520")
        self.win.resizable(True, True)
        nb = ttk.Notebook(self.win)
        nb.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        def refresh():
            self.viewer._dirty = True
            self.viewer._do_render()

        # ---- 渲染模式 ----
        f1 = ttk.LabelFrame(nb, text="渲染模式 / 模型", padding=6)
        mode_options = ["1: 完整模型", "2: 仅左右面", "3: 仅左右面+体介质", "4: 仅左右面+点云光栅化"]
        ttk.Label(f1, text="RENDER_MODE:").grid(row=0, column=0, sticky=tk.W, pady=2)
        cb = ttk.Combobox(f1, values=mode_options, state="readonly", width=22)
        cb.set(mode_options[RENDER_MODE - 1] if 1 <= RENDER_MODE <= 4 else mode_options[0])
        cb.grid(row=0, column=1, sticky=tk.W, pady=2)

        def _on_mode(e):
            global RENDER_MODE
            RENDER_MODE = cb.current() + 1
            self.viewer.render_mode = RENDER_MODE
            refresh()
        cb.bind("<<ComboboxSelected>>", _on_mode)
        nb.add(f1, text="模式")

        # ---- Mode4 颜色条 ----
        f1b = ttk.LabelFrame(nb, text="点云颜色条 (RENDER_MODE 4)", padding=6)
        vc0r = tk.DoubleVar(value=POINT_CLOUD_COLOR_BAR_0[0])
        vc0g = tk.DoubleVar(value=POINT_CLOUD_COLOR_BAR_0[1])
        vc0b = tk.DoubleVar(value=POINT_CLOUD_COLOR_BAR_0[2])
        vc1r = tk.DoubleVar(value=POINT_CLOUD_COLOR_BAR_1[0])
        vc1g = tk.DoubleVar(value=POINT_CLOUD_COLOR_BAR_1[1])
        vc1b = tk.DoubleVar(value=POINT_CLOUD_COLOR_BAR_1[2])
        def _on_color_bar_0(_=None):
            global POINT_CLOUD_COLOR_BAR_0
            POINT_CLOUD_COLOR_BAR_0 = (vc0r.get(), vc0g.get(), vc0b.get())
            refresh()
        def _on_color_bar_1(_=None):
            global POINT_CLOUD_COLOR_BAR_1
            POINT_CLOUD_COLOR_BAR_1 = (vc1r.get(), vc1g.get(), vc1b.get())
            refresh()
        for i, (label, vr, vg, vb, cb) in enumerate([
            ("offset=0 颜色 R/G/B:", vc0r, vc0g, vc0b, _on_color_bar_0),
            ("offset=1 颜色 R/G/B:", vc1r, vc1g, vc1b, _on_color_bar_1),
        ]):
            ttk.Label(f1b, text=label).grid(row=i*4, column=0, sticky=tk.W, pady=2)
            ttk.Scale(f1b, from_=0, to=1, variable=vr, orient=tk.HORIZONTAL, length=140, command=cb).grid(row=i*4, column=1, sticky=tk.EW, pady=2)
            ttk.Scale(f1b, from_=0, to=1, variable=vg, orient=tk.HORIZONTAL, length=140, command=cb).grid(row=i*4+1, column=1, sticky=tk.EW, pady=2)
            ttk.Scale(f1b, from_=0, to=1, variable=vb, orient=tk.HORIZONTAL, length=140, command=cb).grid(row=i*4+2, column=1, sticky=tk.EW, pady=2)
        f1b.columnconfigure(1, weight=1)
        nb.add(f1b, text="Mode4 颜色条")

        # ---- 材质 ----
        f2 = ttk.LabelFrame(nb, text="材质 (Mitsuba)", padding=6)
        row = 0
        global BSDF_OPACITY, BSDF_DIFFUSE_COLOR
        ttk.Label(f2, text="BSDF_OPACITY:").grid(row=row, column=0, sticky=tk.W, pady=2)
        v2 = tk.DoubleVar(value=BSDF_OPACITY)
        def _on_opacity(v):
            global BSDF_OPACITY
            BSDF_OPACITY = float(v)
            refresh()
        ttk.Scale(f2, from_=0, to=1, variable=v2, orient=tk.HORIZONTAL, length=160, command=_on_opacity).grid(row=row, column=1, sticky=tk.EW, pady=2)
        row += 1
        vR, vG, vB = tk.DoubleVar(value=BSDF_DIFFUSE_COLOR[0]), tk.DoubleVar(value=BSDF_DIFFUSE_COLOR[1]), tk.DoubleVar(value=BSDF_DIFFUSE_COLOR[2])
        def _on_diffuse(_=None):
            global BSDF_DIFFUSE_COLOR
            BSDF_DIFFUSE_COLOR = (vR.get(), vG.get(), vB.get())
            refresh()
        ttk.Label(f2, text="BSDF_DIFFUSE_COLOR R:").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Scale(f2, from_=0, to=1, variable=vR, orient=tk.HORIZONTAL, length=160, command=_on_diffuse).grid(row=row, column=1, sticky=tk.EW, pady=2)
        row += 1
        ttk.Label(f2, text="G:").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Scale(f2, from_=0, to=1, variable=vG, orient=tk.HORIZONTAL, length=160, command=_on_diffuse).grid(row=row, column=1, sticky=tk.EW, pady=2)
        row += 1
        ttk.Label(f2, text="B:").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Scale(f2, from_=0, to=1, variable=vB, orient=tk.HORIZONTAL, length=160, command=_on_diffuse).grid(row=row, column=1, sticky=tk.EW, pady=2)
        row += 1
        f2.columnconfigure(1, weight=1)
        nb.add(f2, text="材质")

        # ---- 体介质 (Mode 3) ----
        f3 = ttk.LabelFrame(nb, text="体介质 (RENDER_MODE 3)", padding=6)
        ttk.Label(f3, text="VOL_SIGMA_T_SCALE:").grid(row=0, column=0, sticky=tk.W, pady=2)
        vs = tk.DoubleVar(value=VOL_SIGMA_T_SCALE)
        def _on_vol_scale(v):
            global VOL_SIGMA_T_SCALE
            VOL_SIGMA_T_SCALE = float(v)
            refresh()
        ttk.Scale(f3, from_=0.1, to=50, variable=vs, orient=tk.HORIZONTAL, length=160, command=_on_vol_scale).grid(row=0, column=1, sticky=tk.EW, pady=2)
        ttk.Label(f3, text="VOL_ALBEDO_SMOOTH_SIGMA:").grid(row=1, column=0, sticky=tk.W, pady=2)
        vss = tk.DoubleVar(value=VOL_ALBEDO_SMOOTH_SIGMA)
        def _on_smooth_sigma(v):
            global VOL_ALBEDO_SMOOTH_SIGMA
            VOL_ALBEDO_SMOOTH_SIGMA = float(v)
            refresh()
        ttk.Scale(f3, from_=0.001, to=0.3, variable=vss, orient=tk.HORIZONTAL, length=160, command=_on_smooth_sigma).grid(row=1, column=1, sticky=tk.EW, pady=2)
        ttk.Label(f3, text="VOL_GRID 分辨率需重启生效").grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=4)
        f3.columnconfigure(1, weight=1)
        nb.add(f3, text="体介质")

        # ---- 相机 ----
        f4 = ttk.LabelFrame(nb, text="相机", padding=6)
        ttk.Label(f4, text="FOV (度):").grid(row=0, column=0, sticky=tk.W, pady=2)
        vf = tk.DoubleVar(value=self.viewer._fov)
        def _on_fov(v):
            self.viewer._fov = float(v)
            self.viewer._update_title()
            refresh()
        ttk.Scale(f4, from_=CAMERA_FOV_MIN, to=CAMERA_FOV_MAX, variable=vf, orient=tk.HORIZONTAL, length=160, command=_on_fov).grid(row=0, column=1, sticky=tk.EW, pady=2)
        f4.columnconfigure(1, weight=1)
        nb.add(f4, text="相机")

        # ---- 渲染质量 ----
        f5 = ttk.LabelFrame(nb, text="渲染质量", padding=6)
        vdn = tk.BooleanVar(value=DENOISE_PREVIEW)
        vdnf = tk.BooleanVar(value=DENOISE_FINAL)
        def _on_spp_preview(v):
            global SPP_PREVIEW
            SPP_PREVIEW = int(float(v))
            refresh()
        def _on_spp_final(v):
            global SPP_FINAL
            SPP_FINAL = int(float(v))
            refresh()
        def _on_denoise_preview():
            global DENOISE_PREVIEW
            DENOISE_PREVIEW = vdn.get()
            refresh()
        def _on_denoise_final():
            global DENOISE_FINAL
            DENOISE_FINAL = vdnf.get()
            refresh()
        ttk.Label(f5, text="SPP_PREVIEW:").grid(row=0, column=0, sticky=tk.W, pady=2)
        vp = tk.IntVar(value=SPP_PREVIEW)
        ttk.Scale(f5, from_=1, to=32, variable=vp, orient=tk.HORIZONTAL, length=160, command=_on_spp_preview).grid(row=0, column=1, sticky=tk.EW, pady=2)
        ttk.Label(f5, text="SPP_FINAL:").grid(row=1, column=0, sticky=tk.W, pady=2)
        vf2 = tk.IntVar(value=SPP_FINAL)
        ttk.Scale(f5, from_=4, to=256, variable=vf2, orient=tk.HORIZONTAL, length=160, command=_on_spp_final).grid(row=1, column=1, sticky=tk.EW, pady=2)
        ttk.Label(f5, text="DENOISE_PREVIEW:").grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Checkbutton(f5, variable=vdn, command=_on_denoise_preview).grid(row=2, column=1, sticky=tk.W, pady=2)
        ttk.Label(f5, text="DENOISE_FINAL:").grid(row=3, column=0, sticky=tk.W, pady=2)
        ttk.Checkbutton(f5, variable=vdnf, command=_on_denoise_final).grid(row=3, column=1, sticky=tk.W, pady=2)
        f5.columnconfigure(1, weight=1)
        nb.add(f5, text="质量")

        # ---- 光栅化 ----
        f6 = ttk.LabelFrame(nb, text="光栅化 (VTK)", padding=6)
        def _on_raster_opacity(v):
            global RASTER_OPACITY
            RASTER_OPACITY = float(v)
            refresh()
        def _on_raster_r(r):
            global RASTER_COLOR
            RASTER_COLOR = (float(r), RASTER_COLOR[1], RASTER_COLOR[2])
            refresh()
        ttk.Label(f6, text="RASTER_OPACITY:").grid(row=0, column=0, sticky=tk.W, pady=2)
        vro = tk.DoubleVar(value=RASTER_OPACITY)
        ttk.Scale(f6, from_=0, to=1, variable=vro, orient=tk.HORIZONTAL, length=160, command=_on_raster_opacity).grid(row=0, column=1, sticky=tk.EW, pady=2)
        ttk.Label(f6, text="RASTER_COLOR R:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Scale(f6, from_=0, to=1, variable=tk.DoubleVar(value=RASTER_COLOR[0]), orient=tk.HORIZONTAL, length=160, command=_on_raster_r).grid(row=1, column=1, sticky=tk.EW, pady=2)
        f6.columnconfigure(1, weight=1)
        nb.add(f6, text="光栅化")

        # ---- 环境 / 光源 ----
        f7 = ttk.LabelFrame(nb, text="环境 / 光源", padding=6)
        vkl = tk.BooleanVar(value=KEY_LIGHT_ENABLED)
        def _on_skybox_intensity(v):
            global SKYBOX_INTENSITY
            SKYBOX_INTENSITY = float(v)
            refresh()
        def _on_key_light():
            global KEY_LIGHT_ENABLED
            KEY_LIGHT_ENABLED = vkl.get()
            refresh()
        def _on_key_light_intensity(v):
            global KEY_LIGHT_INTENSITY
            KEY_LIGHT_INTENSITY = float(v)
            refresh()
        ttk.Label(f7, text="SKYBOX_INTENSITY:").grid(row=0, column=0, sticky=tk.W, pady=2)
        vsi = tk.DoubleVar(value=SKYBOX_INTENSITY)
        ttk.Scale(f7, from_=0.01, to=1, variable=vsi, orient=tk.HORIZONTAL, length=160, command=_on_skybox_intensity).grid(row=0, column=1, sticky=tk.EW, pady=2)
        ttk.Label(f7, text="KEY_LIGHT_ENABLED:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Checkbutton(f7, variable=vkl, command=_on_key_light).grid(row=1, column=1, sticky=tk.W, pady=2)
        ttk.Label(f7, text="KEY_LIGHT_INTENSITY:").grid(row=2, column=0, sticky=tk.W, pady=2)
        vkli = tk.DoubleVar(value=KEY_LIGHT_INTENSITY)
        ttk.Scale(f7, from_=0, to=32, variable=vkli, orient=tk.HORIZONTAL, length=160, command=_on_key_light_intensity).grid(row=2, column=1, sticky=tk.EW, pady=2)
        f7.columnconfigure(1, weight=1)
        nb.add(f7, text="环境")

        self._panels = {}
        self.win.protocol("WM_DELETE_WINDOW", self.win.withdraw)

    def show(self):
        self.win.deiconify()
        self.win.lift()
        self.win.focus_force()


# ============================================================
#  交互式查看器
# ============================================================

class InteractiveViewer:
    """基于 tkinter 的可交互查看器。
    Tab 切换:  光栅化模式 (VTK) / 路径追踪模式 (Mitsuba)。"""

    MODE_RASTER = "raster"
    MODE_PATHTRACE = "pathtrace"

    def __init__(self, ply_path, vertices, normals, faces, polydata,
                 point_cloud_aos=None, point_cloud_polydata=None,
                 obb_wireframe_polydata=None, scene_extent=None, obb_center=None,
                 obb_vectors=None, render_mode=1):
        self.ply_path = ply_path
        self.vertices = vertices
        self.normals = normals
        self.faces = faces
        self.polydata = polydata
        self.point_cloud_aos = point_cloud_aos
        self.point_cloud_polydata = point_cloud_polydata
        self.obb_wireframe_polydata = obb_wireframe_polydata
        self.scene_extent = scene_extent
        self.obb_vectors = obb_vectors
        self.render_mode = render_mode

        self.origin, self.target = compute_auto_camera(vertices, target_override=obb_center)
        self.up = np.array(CAMERA_UP, dtype=np.float64)

        self._orbit_azimuth = 0.0
        self._orbit_elevation = 0.0
        self._orbit_distance = np.linalg.norm(
            np.array(self.origin) - np.array(self.target)
        )
        self._fov = float(CAMERA_FOV)

        self._mode = self.MODE_RASTER
        self.scene = None

        self._dragging = False
        self._panning = False
        self._last_x = 0
        self._last_y = 0
        self._render_seed = 0
        self._dirty = True
        self._refine_id = None

        self.root = tk.Tk()
        self.root.configure(bg="black")

        self.canvas = tk.Canvas(
            self.root, width=FILM_WIDTH, height=FILM_HEIGHT,
            bg="black", highlightthickness=0,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<ButtonPress-2>", self._on_pan_press)
        self.canvas.bind("<B2-Motion>", self._on_pan_drag)
        self.canvas.bind("<ButtonRelease-2>", self._on_pan_release)
        self.canvas.bind("<ButtonPress-3>", self._on_pan_press)
        self.canvas.bind("<B3-Motion>", self._on_pan_drag)
        self.canvas.bind("<ButtonRelease-3>", self._on_pan_release)
        self.canvas.bind("<MouseWheel>", self._on_scroll)
        self.root.bind("<F12>", self._on_screenshot)
        self.root.bind("<F2>", self._on_params_panel)
        self.root.bind("<Tab>", self._on_toggle_mode)
        self._params_panel = None
        self.root.bind("<KeyPress-bracketleft>", self._on_fov_decrease)
        self.root.bind("<KeyPress-bracketright>", self._on_fov_increase)

        self._tk_image = None
        self._update_title()
        self._do_render()

    # ---------- 模式切换 ----------

    def _update_title(self):
        if self.render_mode == 4:
            tag = "光栅化 (模式4)"
        elif self._mode == self.MODE_RASTER:
            tag = "光栅化实时"
        else:
            tag = "路径追踪预览"
        self.root.title(f"{WINDOW_TITLE}  [{tag}]  FOV={self._fov:.0f}°  (Tab 切换 [ ] FOV  F2 参数  F12 截图)")

    def _on_toggle_mode(self, event=None):
        if self.render_mode == 4:
            return  # 模式 4 仅光栅化，不切换
        if self._mode == self.MODE_RASTER:
            self._mode = self.MODE_PATHTRACE
        else:
            self._mode = self.MODE_RASTER
        self._update_title()
        self._dirty = True
        self._do_render()

    def _on_fov_increase(self, event=None):
        self._fov = min(CAMERA_FOV_MAX, self._fov + CAMERA_FOV_STEP)
        self._update_title()
        self._dirty = True
        self._do_render()

    def _on_fov_decrease(self, event=None):
        self._fov = max(CAMERA_FOV_MIN, self._fov - CAMERA_FOV_STEP)
        self._update_title()
        self._dirty = True
        self._do_render()

    def _on_params_panel(self, event=None):
        if self._params_panel is None:
            self._params_panel = ParamsPanel(self.root, self)
        self._params_panel.show()

    # ---------- 场景 / 相机 ----------

    def _compute_origin_from_orbit(self):
        az = math.radians(self._orbit_azimuth)
        el = math.radians(self._orbit_elevation)
        d = self._orbit_distance
        t = np.array(self.target, dtype=np.float64)
        dx = d * math.cos(el) * math.sin(az)
        dy = d * math.sin(el)
        dz = d * math.cos(el) * math.cos(az)
        return tuple((t + np.array([dx, dy, dz])).tolist())

    def _rebuild_mitsuba_scene(self):
        self.origin = self._compute_origin_from_orbit()
        scene_dict = build_scene_dict(
            self.ply_path, self.origin, self.target,
            point_cloud_aos=self.point_cloud_aos,
            scene_extent=self.scene_extent,
            fov=self._fov,
            render_mode=self.render_mode,
            obb_vectors=self.obb_vectors,
        )
        self.scene = mi.load_dict(scene_dict)

    # ---------- 统一渲染入口 ----------

    def _do_render(self):
        if self._refine_id is not None:
            self.root.after_cancel(self._refine_id)
            self._refine_id = None

        self.origin = self._compute_origin_from_orbit()

        if self._mode == self.MODE_RASTER or self.render_mode == 4:
            if self.render_mode == 4:
                aos = self.point_cloud_aos or []
                point_sphere_pd = build_vtk_point_cloud_polydata_with_colors(
                    aos, 0, self.scene_extent or 1.0,
                    POINT_CLOUD_COLOR_BAR_0, POINT_CLOUD_COLOR_BAR_1,
                )
                arr = raster_render_mode4_to_array(
                    self.polydata, point_sphere_pd,
                    self.origin, self.target, tuple(self.up.tolist()),
                    self._fov, FILM_WIDTH, FILM_HEIGHT,
                )
            else:
                arr = raster_render_to_array(
                    self.polydata, self.origin, self.target,
                    tuple(self.up.tolist()), self._fov,
                    FILM_WIDTH, FILM_HEIGHT,
                    obb_wireframe_polydata=None if self.render_mode == 4 else self.obb_wireframe_polydata,
                )
            self._show_image(arr)
            self._dirty = False
        else:
            self._render_seed += 1
            self._rebuild_mitsuba_scene()
            arr = render_image(self.scene, SPP_PREVIEW, self._render_seed,
                               denoise=DENOISE_PREVIEW)
            self._show_image(arr)
            self._dirty = False
            self._refine_id = self.root.after(400, self._do_refine_render)

    def _do_refine_render(self):
        self._refine_id = None
        if self._dirty or self._mode != self.MODE_PATHTRACE:
            return
        self._render_seed += 1
        arr = render_image(self.scene, SPP_FINAL, self._render_seed,
                           denoise=DENOISE_FINAL)
        if not self._dirty:
            self._show_image(arr)

    def _show_image(self, arr):
        pil = Image.fromarray(arr)
        self._tk_image = ImageTk.PhotoImage(pil)
        self.canvas.delete("all")
        self.canvas.create_image(
            FILM_WIDTH // 2, FILM_HEIGHT // 2,
            image=self._tk_image, anchor=tk.CENTER,
        )

    # ---------- 鼠标事件: 旋转 ----------

    def _on_press(self, event):
        self._dragging = True
        self._last_x = event.x
        self._last_y = event.y

    def _on_drag(self, event):
        if not self._dragging:
            return
        dx = event.x - self._last_x
        dy = event.y - self._last_y
        self._last_x = event.x
        self._last_y = event.y

        y_sign = 1.0 if MOUSE_INVERT_Y else -1.0
        self._orbit_azimuth -= dx * MOUSE_ORBIT_SENSITIVITY
        self._orbit_elevation = max(
            -89, min(89, self._orbit_elevation + dy * MOUSE_ORBIT_SENSITIVITY * y_sign)
        )

        self._dirty = True
        self._do_render()

    def _on_release(self, event):
        self._dragging = False

    # ---------- 鼠标事件: 平移 ----------

    def _on_pan_press(self, event):
        self._panning = True
        self._last_x = event.x
        self._last_y = event.y

    def _on_pan_drag(self, event):
        if not self._panning:
            return
        dx = event.x - self._last_x
        dy = event.y - self._last_y
        self._last_x = event.x
        self._last_y = event.y

        origin = np.array(self.origin, dtype=np.float64)
        target = np.array(self.target, dtype=np.float64)
        forward = target - origin
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, self.up)
        right /= np.linalg.norm(right)
        up_actual = np.cross(right, forward)

        scale = self._orbit_distance * MOUSE_PAN_SENSITIVITY
        y_sign = -1.0 if MOUSE_INVERT_Y else 1.0
        shift = -right * dx * scale - up_actual * dy * scale * y_sign
        self.target = tuple((target + shift).tolist())

        self._dirty = True
        self._do_render()

    def _on_pan_release(self, event):
        self._panning = False

    # ---------- 鼠标事件: 缩放 ----------

    def _on_scroll(self, event):
        if event.delta > 0:
            factor = 1.0 - MOUSE_ZOOM_FACTOR
        else:
            factor = 1.0 + MOUSE_ZOOM_FACTOR
        self._orbit_distance = max(0.01, self._orbit_distance * factor)
        self._dirty = True
        self._do_render()

    # ---------- 截图 (F12, 始终用路径追踪) ----------

    def _on_screenshot(self, event=None):
        import datetime
        self.root.title(WINDOW_TITLE + "  [渲染截图中…]")
        self.root.update_idletasks()

        self.origin = self._compute_origin_from_orbit()
        if self.render_mode == 4:
            point_sphere_pd = build_vtk_point_cloud_polydata_with_colors(
                self.point_cloud_aos, 0, self.scene_extent,
                POINT_CLOUD_COLOR_BAR_0, POINT_CLOUD_COLOR_BAR_1,
            )
            arr = raster_render_mode4_to_array(
                self.polydata, point_sphere_pd,
                self.origin, self.target, tuple(self.up.tolist()),
                self._fov, SCREENSHOT_WIDTH, SCREENSHOT_HEIGHT,
            )
        else:
            scene_dict = build_scene_dict(
                self.ply_path, self.origin, self.target,
                film_w=SCREENSHOT_WIDTH, film_h=SCREENSHOT_HEIGHT,
                point_cloud_aos=self.point_cloud_aos,
                scene_extent=self.scene_extent,
                fov=self._fov,
                render_mode=self.render_mode,
                obb_vectors=self.obb_vectors,
            )
            scene = mi.load_dict(scene_dict)
            self._render_seed += 1
            arr = render_image(scene, SCREENSHOT_SPP, self._render_seed,
                               denoise=DENOISE_SCREENSHOT)

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{ts}.png"
        save_path = os.path.join(SCREENSHOT_DIR, filename)
        Image.fromarray(arr).save(save_path)
        print(f"[INFO] 截图已保存: {os.path.abspath(save_path)}")

        self._update_title()

    # ---------- 启动 ----------

    def run(self):
        self.root.mainloop()


# ============================================================
#  主流程
# ============================================================

def main():
    folder, file_id, left_tag, right_tag = parse_npz_filename(NPZ_FILE)
    step_path = os.path.join(WORK_DIR, folder, f"{file_id}.step")
    print(f"[INFO] STEP 文件路径: {step_path}")
    print(f"[INFO] 左面 tag={left_tag}, 右面 tag={right_tag}")
    mode_desc = {1: "完整模型", 2: "仅左右面", 3: "仅左右面+体介质", 4: "仅左右面+点云光栅化"}
    print(f"[INFO] 渲染模式: {RENDER_MODE} ({mode_desc.get(RENDER_MODE, '?')})")

    if RENDER_MODE == 1:
        shape = load_step(step_path)
        print("[INFO] STEP 文件加载完成，开始三角化…")
        vertices, faces, normals = shape_to_numpy(shape)
    else:
        print("[INFO] STEP 文件加载完成，三角化左右面…")
        vertices, faces, normals = shape_to_numpy_left_right_only(
            step_path, left_tag, right_tag,
        )
    print(f"[INFO] 三角化完成: {len(vertices)} 顶点, {len(faces)} 三角面片")

    ply_path = os.path.join(tempfile.gettempdir(), "mitsuba_step_model.ply")
    export_ply(vertices, faces, normals, ply_path)
    print(f"[INFO] 临时 PLY: {ply_path}")

    polydata = build_vtk_polydata(vertices, faces, normals)
    print("[INFO] VTK PolyData 已构建 (光栅化模式)")

    extent = np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))
    point_cloud_aos = None
    point_cloud_polydata = None
    obb_wireframe_polydata = None
    obb_center = None
    obb_vectors = None
    if POINT_CLOUD_ENABLED or RENDER_MODE in (3, 4):
        try:
            aos, obb_center, obb_vectors = load_query_points_aos(
                step_path=step_path, left_tag=left_tag,
                right_tag=right_tag, to_world_space=True,
            )
            point_cloud_aos = aos
            radius = extent * POINT_CLOUD_RADIUS_SCALE
            point_cloud_polydata = build_vtk_point_cloud_polydata(
                [r["point"] for r in aos], radius, extent,
            )
            if obb_vectors is not None:
                obb_wireframe_polydata = build_vtk_obb_wireframe_polydata(*obb_vectors)
            print(f"[INFO] 点云已加载: {len(point_cloud_aos)} 点 (显示上限 {POINT_CLOUD_MAX_POINTS or '不限'})")
        except Exception as e:
            print(f"[WARN] 点云加载失败，将不显示: {e}")

    origin, target = compute_auto_camera(vertices, target_override=obb_center)
    print(f"[INFO] 相机: origin={origin}, target={target}, fov={CAMERA_FOV}")

    viewer = InteractiveViewer(
        ply_path, vertices, normals, faces, polydata,
        point_cloud_aos=point_cloud_aos,
        point_cloud_polydata=point_cloud_polydata,
        obb_wireframe_polydata=obb_wireframe_polydata,
        scene_extent=extent,
        obb_center=obb_center,
        obb_vectors=obb_vectors,
        render_mode=RENDER_MODE,
    )
    viewer.run()


if __name__ == "__main__":
    main()
