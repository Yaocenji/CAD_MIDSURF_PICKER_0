import sys
import os
import warnings
# 忽略 PyQt5 的 DeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import vtk
import pyvista as pv
from pyvistaqt import BackgroundPlotter
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, 
                             QPushButton, QLabel, QFileDialog, QHBoxLayout, QTextEdit,
                             QCheckBox, QScrollArea, QFrame)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPalette

# occwl imports
from occwl.compound import Compound
from occwl.entity_mapper import EntityMapper
from occwl.face import Face
from occwl.solid import Solid

# OCC imports for triangulation
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.BRep import BRep_Tool

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


def parse_highlight_file(filepath):
    """
    解析高亮配置文件。
    格式:
      第一行: STEP 文件路径
      之后每行: 一个或多个 face tag (用空格/逗号/制表符分隔)
    
    返回: (step_file_path, list_of_face_tag_lists)
    例如: ("model.stp", [[1, 2], [3, 4], [5]])
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    if not lines:
        raise ValueError("文件为空")
    
    # 第一行是 STEP 文件路径
    step_file_path = lines[0].strip()
    
    # 后续每行是 face tag 列表
    face_tag_groups = []
    for line in lines[1:]:
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
    
    return step_file_path, face_tag_groups


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
        self.highlight_groups = []  # [(color, [face_ids]), ...]
        self.group_checkboxes = []  # [QCheckBox, ...]
        
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
            step_file_path, face_tag_groups = parse_highlight_file(filename)
            
            # 如果是相对路径，以配置文件所在目录为基准
            if not os.path.isabs(step_file_path):
                config_dir = os.path.dirname(filename)
                step_file_path = os.path.join(config_dir, step_file_path)
            
            if not os.path.exists(step_file_path):
                self.lbl_info.setText(f"错误: STEP 文件不存在: {step_file_path}")
                return
            
            self.lbl_info.setText(f"正在加载 {os.path.basename(step_file_path)}...")
            QApplication.processEvents()
            
            # 加载 STEP 文件
            self.load_step_file(step_file_path)
            
            # 应用高亮
            self.apply_highlights(face_tag_groups)
            
            self.lbl_info.setText(
                f"已加载: {os.path.basename(step_file_path)} | "
                f"面数: {len(list(self.current_solid.faces()))} | "
                f"高亮组数: {len(face_tag_groups)}"
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
        
        # 可视化
        self.visualize_solid()

    def visualize_solid(self):
        """可视化 solid 的所有面"""
        self.plotter.clear()
        self.plotter.enable_lightkit()
        self.face_id_to_actor = {}
        self.actor_to_face_id = {}
        
        faces = list(self.current_solid.faces())
        print(f"可视化 {len(faces)} 个面...")
        
        for face in faces:
            # 获取 EntityMapper ID (与 cad_picker_2 一致)
            face_id = self.entity_mapper.face_index(face)
            
            # 三角化
            mesh = triangulate_face(face)
            if mesh:
                # 添加到 plotter (与 cad_picker_2 渲染效果一致)
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
                
                # 存储映射
                self.face_id_to_actor[face_id] = actor
                self.actor_to_face_id[actor] = face_id
                
                # 添加边界线
                try:
                    edges = mesh.extract_feature_edges(
                        boundary_edges=True,
                        non_manifold_edges=False,
                        feature_edges=False,
                        manifold_edges=False
                    )
                    
                    if edges.n_points > 0:
                        self.plotter.add_mesh(
                            edges,
                            color="black",
                            line_width=1.5,
                            pickable=False
                        )
                except Exception as e:
                    print(f"警告: 提取面 {face_id} 的边界线失败: {e}")
        
        self.plotter.reset_camera()

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
        
        # 清除 checkbox 列表
        while self.scroll_layout.count():
            item = self.scroll_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        
        self.plotter.render()
        self.lbl_info.setText("已清除所有高亮")

    def on_left_click(self, obj, event):
        """G-Buffer (Hardware) Picking - 点击显示 face id"""
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
        
        if picked_actor and picked_actor in self.actor_to_face_id:
            face_id = self.actor_to_face_id[picked_actor]
            
            # 检查这个面是否在高亮组中
            group_info = ""
            for i, (color, tags) in enumerate(self.highlight_groups):
                if face_id in tags:
                    group_info = f" (属于第 {i+1} 组高亮)"
                    break
            
            print(f"点击的面 ID: {face_id}{group_info}")
            self.lbl_info.setText(f"点击的面 ID: {face_id}{group_info}")


def main():
    app = QApplication(sys.argv)
    window = FaceHighlighterWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
