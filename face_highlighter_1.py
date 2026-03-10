import sys
import os
import warnings
# 忽略 PyQt5 的 DeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import vtk
import pyvista as pv
import matplotlib.colors as mcolors
from pyvistaqt import BackgroundPlotter
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, 
                             QPushButton, QLabel, QFileDialog, QHBoxLayout, QTextEdit,
                             QCheckBox, QScrollArea, QFrame, QComboBox, QColorDialog,
                             QSlider, QDoubleSpinBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPalette

# occwl imports
from occwl.compound import Compound
from occwl.entity_mapper import EntityMapper
from occwl.face import Face
from occwl.solid import Solid

# OCC imports for triangulation and OBB
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.BRep import BRep_Tool
from OCC.Core.Bnd import Bnd_OBB
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib_Add, brepbndlib_AddOBB

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

# Albedo 映射数据源选项
ALBEDO_DATA_SOURCES = [
    ("offset_pred", "Offset 预测值"),
    ("offset_gt", "Offset 真实值"),
    ("validity_pred", "Validity 预测值"),
    ("validity_gt", "Validity 真实值"),
]


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
    return (
        np.array(query_points, dtype=np.float64),
        np.array(offset_gt),
        np.array(validity_gt),
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
        
        # 配置文件与面对 (用于 npz 查找)
        self.config_dir = None
        self.config_name = None
        self.face_tag_groups = []  # 从配置文件解析的面对列表
        
        # 点云数据与渲染
        self.point_cloud_actors = []  # 点云相关 actor，用于清除
        self.point_cloud_data = None  # dict: query_points_ws, offset_pred, offset_gt, validity_pred, validity_gt
        self.current_face_pair = None  # (left_id, right_id) 当前加载的面对
        
        # Albedo 映射
        self.albedo_color_min = np.array([0.0, 0.0, 0.3])   # 蓝
        self.albedo_color_max = np.array([1.0, 0.2, 0.2])   # 红
        
        # 镜头 target 管理
        self.model_center = None  # 模型几何中心，首次配置加载时设置
        self._prev_solo_value = 0  # 用于检测滑条 0->1 或 1->0 的切换
        
        # Solo 模式(滑条=1)下左右面自定义颜色与透明度
        self.solo_left_color = np.array([0.2, 0.6, 1.0])   # 左面默认蓝
        self.solo_left_opacity = 1.0
        self.solo_right_color = np.array([1.0, 0.4, 0.2])  # 右面默认橙
        self.solo_right_opacity = 1.0
        
        # 点云小球半径缩放 (1.0 = 默认，滑条控制)
        self.point_cloud_radius_scale = 1.0
        self._point_cloud_obb_extent = None  # 当前点云 OBB 尺寸，用于半径重算
        self.point_cloud_opacity = 1.0  # 点云小球透明度 0-1
        
        # 模型边线：世界空间粗细(相对模型尺寸)、颜色，albedo unlit
        self.model_extent = 1.0  # 模型包围盒对角线，用于线粗参考
        self.edge_line_radius_scale = 0.0003  # 边线 tube 半径 = model_extent * this
        self.edge_line_color = np.array([0.0, 0.0, 0.0])  # 黑
        
        # OBB 框线（solo mode）：世界空间粗细、颜色，albedo unlit
        self.obb_line_radius_scale = 0.0005
        self.obb_line_color = np.array([0.8, 0.2, 0.2])  # 红
        
        # 透明面列表（model mode）：在此列表中的面以透明显示；STF 模式用于维护该列表
        self.transparent_face_ids = set()
        self.transparent_face_opacity = 0.3  # 0=全透明, 1=不透明
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
        
        # Legend / Info panel
        info_panel = QWidget()
        info_layout = QVBoxLayout()
        info_panel.setLayout(info_layout)
        info_panel.setMaximumWidth(350)
        
        self.lbl_legend_title = QLabel("高亮图例 (点击切换):")
        self.lbl_legend_title.setStyleSheet("font-size: 14px; font-weight: bold;")
        info_layout.addWidget(self.lbl_legend_title)
        
        # 全选/取消全选按钮
        select_btn_layout = QHBoxLayout()
        self.btn_select_all = QPushButton("全选")
        self.btn_select_all.clicked.connect(self.select_all_groups)
        select_btn_layout.addWidget(self.btn_select_all)
        
        self.btn_deselect_all = QPushButton("取消全选")
        self.btn_deselect_all.clicked.connect(self.deselect_all_groups)
        select_btn_layout.addWidget(self.btn_deselect_all)
        info_layout.addLayout(select_btn_layout)
        
        # 可滚动的 checkbox 列表
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout()
        self.scroll_layout.setAlignment(Qt.AlignTop)
        self.scroll_widget.setLayout(self.scroll_layout)
        self.scroll_area.setWidget(self.scroll_widget)
        info_layout.addWidget(self.scroll_area)
        
        # ---- 渲染模式滑条 (0=全部 / 1=仅当前左右面) ----
        info_layout.addWidget(QLabel(""))
        lbl_solo = QLabel("渲染模式:")
        lbl_solo.setStyleSheet("font-size: 12px; font-weight: bold;")
        info_layout.addWidget(lbl_solo)
        solo_layout = QHBoxLayout()
        solo_layout.addWidget(QLabel("0"))
        self.slider_solo = QSlider(Qt.Horizontal)
        self.slider_solo.setMinimum(0)
        self.slider_solo.setMaximum(1)
        self.slider_solo.setValue(0)
        self.slider_solo.setTickPosition(QSlider.TicksBelow)
        self.slider_solo.setTickInterval(1)
        self.slider_solo.valueChanged.connect(self._on_solo_slider_changed)
        solo_layout.addWidget(self.slider_solo)
        solo_layout.addWidget(QLabel("1"))
        info_layout.addLayout(solo_layout)
        self.lbl_solo_hint = QLabel("左(0)=显示全部 | 右(1)=仅当前左右面")
        self.lbl_solo_hint.setStyleSheet("font-size: 10px; color: #666;")
        info_layout.addWidget(self.lbl_solo_hint)
        
        # ---- Model 模式：透明面 & STF ----
        info_layout.addWidget(QLabel(""))
        lbl_transp = QLabel("透明面 (Model 模式):")
        lbl_transp.setStyleSheet("font-size: 11px; font-weight: bold;")
        info_layout.addWidget(lbl_transp)
        self.btn_stf = QPushButton("选择透明面 (STF)")
        self.btn_stf.setCheckable(True)
        self.btn_stf.setChecked(False)
        self.btn_stf.clicked.connect(self._on_stf_toggled)
        self.btn_stf.setEnabled(True)
        info_layout.addWidget(self.btn_stf)
        stf_color_layout = QHBoxLayout()
        stf_color_layout.addWidget(QLabel("普通:"))
        self.btn_stf_normal_color = QPushButton()
        self.btn_stf_normal_color.setFixedWidth(60)
        self.btn_stf_normal_color.setStyleSheet("background-color: rgb(179,179,179);")
        self.btn_stf_normal_color.clicked.connect(lambda: self._pick_stf_color(is_highlight=False))
        stf_color_layout.addWidget(self.btn_stf_normal_color)
        stf_color_layout.addWidget(QLabel("高亮:"))
        self.btn_stf_highlight_color = QPushButton()
        self.btn_stf_highlight_color.setFixedWidth(60)
        self.btn_stf_highlight_color.setStyleSheet("background-color: rgb(51,242,128);")
        self.btn_stf_highlight_color.clicked.connect(lambda: self._pick_stf_color(is_highlight=True))
        stf_color_layout.addWidget(self.btn_stf_highlight_color)
        info_layout.addLayout(stf_color_layout)
        transp_opacity_layout = QHBoxLayout()
        transp_opacity_layout.addWidget(QLabel("透明度:"))
        self.slider_transparent_opacity = QSlider(Qt.Horizontal)
        self.slider_transparent_opacity.setMinimum(0)
        self.slider_transparent_opacity.setMaximum(100)
        self.slider_transparent_opacity.setValue(30)
        self.slider_transparent_opacity.valueChanged.connect(self._on_transparent_opacity_changed)
        transp_opacity_layout.addWidget(self.slider_transparent_opacity)
        self.lbl_transparent_opacity = QLabel("0.30")
        transp_opacity_layout.addWidget(self.lbl_transparent_opacity)
        info_layout.addLayout(transp_opacity_layout)
        
        # Solo 模式下面颜色（滑条=1 时生效，取代 HIGHLIGHT_COLORS）
        solo_color_layout = QVBoxLayout()
        lbl_solo_colors = QLabel("Solo 模式下面颜色:")
        lbl_solo_colors.setStyleSheet("font-size: 11px; font-weight: bold;")
        solo_color_layout.addWidget(lbl_solo_colors)
        left_row = QHBoxLayout()
        left_row.addWidget(QLabel("左面:"))
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
        right_row.addWidget(QLabel("右面:"))
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
        info_layout.addLayout(solo_color_layout)
        
        # ---- 点云 Albedo 映射 ----
        info_layout.addWidget(QLabel(""))  # 分隔
        lbl_albedo = QLabel("点云 Albedo 映射:")
        lbl_albedo.setStyleSheet("font-size: 12px; font-weight: bold;")
        info_layout.addWidget(lbl_albedo)
        
        albedo_combo_layout = QHBoxLayout()
        albedo_combo_layout.addWidget(QLabel("数据源:"))
        self.combo_albedo_source = QComboBox()
        for key, label in ALBEDO_DATA_SOURCES:
            self.combo_albedo_source.addItem(label, key)
        self.combo_albedo_source.currentIndexChanged.connect(self._on_albedo_source_changed)
        albedo_combo_layout.addWidget(self.combo_albedo_source)
        info_layout.addLayout(albedo_combo_layout)
        
        # 颜色条：起始色 | [渐变色条] | 结束色
        color_bar_layout = QHBoxLayout()
        self.btn_color_min = QPushButton("起始色")
        self.btn_color_min.setStyleSheet("background-color: rgb(0,0,76); color: white;")
        self.btn_color_min.clicked.connect(lambda: self._pick_albedo_color(is_min=True))
        self.btn_color_max = QPushButton("结束色")
        self.btn_color_max.setStyleSheet("background-color: rgb(255,51,51); color: white;")
        self.btn_color_max.clicked.connect(lambda: self._pick_albedo_color(is_min=False))
        color_bar_layout.addWidget(self.btn_color_min)
        self.color_bar_preview = QFrame()
        self.color_bar_preview.setFixedHeight(24)
        self.color_bar_preview.setStyleSheet(
            "background: qlineargradient(x1:0, y1:0, x2:1, y2:0, "
            "stop:0 rgb(0,0,76), stop:1 rgb(255,51,51)); border: 1px solid #666;"
        )
        color_bar_layout.addWidget(self.color_bar_preview, stretch=1)
        color_bar_layout.addWidget(self.btn_color_max)
        info_layout.addLayout(color_bar_layout)
        
        # ---- 模型边线 ----
        info_layout.addWidget(QLabel(""))
        lbl_edges = QLabel("模型边线:")
        lbl_edges.setStyleSheet("font-size: 11px; font-weight: bold;")
        info_layout.addWidget(lbl_edges)
        self.cb_show_edges = QCheckBox("显示边线")
        self.cb_show_edges.setChecked(True)
        self.cb_show_edges.toggled.connect(self._on_show_edges_changed)
        info_layout.addWidget(self.cb_show_edges)
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
        info_layout.addLayout(edge_row)
        
        # ---- OBB 框线 (Solo 模式) ----
        info_layout.addWidget(QLabel(""))
        lbl_obb = QLabel("OBB 框线:")
        lbl_obb.setStyleSheet("font-size: 11px; font-weight: bold;")
        info_layout.addWidget(lbl_obb)
        self.cb_show_obb = QCheckBox("显示 OBB 框线")
        self.cb_show_obb.setChecked(True)
        self.cb_show_obb.toggled.connect(self._on_show_obb_changed)
        info_layout.addWidget(self.cb_show_obb)
        obb_row = QHBoxLayout()
        self.btn_obb_color = QPushButton("颜色")
        self.btn_obb_color.setStyleSheet("background-color: rgb(204,51,51); color: white;")
        self.btn_obb_color.clicked.connect(self._pick_obb_color)
        obb_row.addWidget(self.btn_obb_color)
        obb_row.addWidget(QLabel("粗细:"))
        self.spin_obb_radius = QDoubleSpinBox()
        self.spin_obb_radius.setRange(0.00001, 0.01)
        self.spin_obb_radius.setSingleStep(0.0001)
        self.spin_obb_radius.setValue(0.0005)
        self.spin_obb_radius.setDecimals(5)
        self.spin_obb_radius.valueChanged.connect(self._on_obb_radius_changed)
        obb_row.addWidget(self.spin_obb_radius)
        info_layout.addLayout(obb_row)
        
        # ---- 点云小球半径 ----
        info_layout.addWidget(QLabel(""))
        lbl_radius = QLabel("点云小球半径:")
        lbl_radius.setStyleSheet("font-size: 11px; font-weight: bold;")
        info_layout.addWidget(lbl_radius)
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
        info_layout.addLayout(radius_layout)
        point_opacity_layout = QHBoxLayout()
        point_opacity_layout.addWidget(QLabel("透明度:"))
        self.slider_point_opacity = QSlider(Qt.Horizontal)
        self.slider_point_opacity.setMinimum(0)
        self.slider_point_opacity.setMaximum(100)
        self.slider_point_opacity.setValue(100)
        self.slider_point_opacity.valueChanged.connect(self._on_point_cloud_opacity_changed)
        point_opacity_layout.addWidget(self.slider_point_opacity)
        self.lbl_point_opacity = QLabel("1.00")
        point_opacity_layout.addWidget(self.lbl_point_opacity)
        info_layout.addLayout(point_opacity_layout)
        
        content_layout.addWidget(info_panel, stretch=1)
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

    def _on_solo_slider_changed(self, value):
        """滑条值变化时更新渲染模式和镜头 target"""
        prev = self._prev_solo_value
        self._prev_solo_value = value
        
        # 镜头 target 管理：仅在切换时刻更新
        if prev == 0 and value == 1:
            # 滑条从 0 滑向 1：target 设为左右面 OBB 几何中心
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
            # 滑条从 1 滑回 0：target 设回模型几何中心
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
        根据滑条值(0/1)和当前选中的面对，控制面的显示/隐藏及颜色。
        - 滑条=0: 显示所有面，model 边线，无 OBB
        - 滑条=1: 仅显示当前左右面，solo 边线，OBB 框
        """
        solo = self.slider_solo.value() if hasattr(self, "slider_solo") else 0
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
                    return "lightgrey"
            return "lightgrey"

        stf_mode = getattr(self, "stf_mode", False)
        transp_opacity = getattr(self, "transparent_face_opacity", 0.3)
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
                    actor.prop.color = _get_normal_color(face_id)
                    if show_ids is None and face_id in self.transparent_face_ids:
                        actor.prop.opacity = transp_opacity
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
        """应用高亮颜色到指定的面，并创建可交互的 checkbox 列表"""
        self.highlight_groups = []
        self.group_checkboxes = []
        
        # 清除旧的 checkbox
        while self.scroll_layout.count():
            item = self.scroll_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        
        for i, tags in enumerate(face_tag_groups):
            # 循环使用颜色
            color = HIGHLIGHT_COLORS[i % len(HIGHLIGHT_COLORS)]
            
            # 高亮这一组的所有面
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
                
                # 创建 checkbox
                tags_str = ", ".join(map(str, valid_tags))
                cb = QCheckBox(f"第 {i+1} 组: [{tags_str}]")
                cb.setChecked(True)
                cb.setStyleSheet(
                    f"QCheckBox {{ font-size: 12px; padding: 4px; }}"
                    f"QCheckBox::indicator:checked {{ background-color: {color}; border: 1px solid #333; }}"
                    f"QCheckBox::indicator:unchecked {{ background-color: #d0d0d0; border: 1px solid #999; }}"
                    f"QCheckBox::indicator {{ width: 14px; height: 14px; }}"
                )
                # 用默认参数捕获当前 group_index
                cb.toggled.connect(lambda checked, idx=group_index: self.on_group_toggled(idx, checked))
                self.scroll_layout.addWidget(cb)
                self.group_checkboxes.append(cb)
        
        self.plotter.render()

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
        self._clear_point_cloud()
        
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
        self._apply_solo_mode()  # 清除后面无选中面对，恢复全部显示

    def _pick_albedo_color(self, is_min=True):
        """选择 Albedo 颜色条端点颜色"""
        col = self.albedo_color_min if is_min else self.albedo_color_max
        qcol = QColorDialog.getColor(
            QColor(int(col[0]*255), int(col[1]*255), int(col[2]*255)),
            self, "选择颜色"
        )
        if qcol.isValid():
            rgb = qcol.red()/255, qcol.green()/255, qcol.blue()/255
            if is_min:
                self.albedo_color_min = np.array(rgb)
                self.btn_color_min.setStyleSheet(
                    f"background-color: rgb({qcol.red()},{qcol.green()},{qcol.blue()}); "
                    f"color: {'white' if (qcol.red()+qcol.green()+qcol.blue())/3 < 128 else 'black'};"
                )
            else:
                self.albedo_color_max = np.array(rgb)
                self.btn_color_max.setStyleSheet(
                    f"background-color: rgb({qcol.red()},{qcol.green()},{qcol.blue()}); "
                    f"color: {'white' if (qcol.red()+qcol.green()+qcol.blue())/3 < 128 else 'black'};"
                )
            self._update_color_bar_preview()
            self._update_point_cloud_colors()

    def _update_color_bar_preview(self):
        """更新颜色条预览"""
        c1 = self.albedo_color_min
        c2 = self.albedo_color_max
        r1, g1, b1 = int(c1[0]*255), int(c1[1]*255), int(c1[2]*255)
        r2, g2, b2 = int(c2[0]*255), int(c2[1]*255), int(c2[2]*255)
        self.color_bar_preview.setStyleSheet(
            f"background: qlineargradient(x1:0, y1:0, x2:1, y2:0, "
            f"stop:0 rgb({r1},{g1},{b1}), stop:1 rgb({r2},{g2},{b2})); "
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
        for actor in self.point_cloud_actors:
            actor.prop.opacity = self.point_cloud_opacity
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
        """Albedo 数据源切换时更新点云颜色"""
        self._update_point_cloud_colors()

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
        colors = (1 - t)[:, np.newaxis] * self.albedo_color_min + t[:, np.newaxis] * self.albedo_color_max
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
        self._create_point_cloud_glyphs()
        self.plotter.render()
        return True

    def _create_point_cloud_glyphs(self):
        """根据当前点云数据与半径缩放创建/更新 glyph 球体"""
        if not self.point_cloud_data or "query_points_ws" not in self.point_cloud_data:
            return
        query_points_ws = self.point_cloud_data["query_points_ws"]
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
        colors = (1 - t)[:, np.newaxis] * self.albedo_color_min + t[:, np.newaxis] * self.albedo_color_max
        colors = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
        
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
        actor.prop.opacity = self.point_cloud_opacity
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
        solo = self.slider_solo.value() if hasattr(self, "slider_solo") else 0
        
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
