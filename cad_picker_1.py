import sys
import os
import warnings
# 忽略 PyQt5 的 DeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import pyvista as pv
from pyvistaqt import BackgroundPlotter
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, 
                             QPushButton, QLabel, QFileDialog)
from PyQt5.QtCore import Qt

# occwl imports
from occwl.compound import Compound
from occwl.entity_mapper import EntityMapper
from occwl.solid import Solid

# OCC imports
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.BRep import BRep_Tool
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop_SurfaceProperties

def triangulate_face(face, face_id, deflection=0.001):
    """
    Triangulate an occwl Face and assign the face_id to its cell data.
    """
    BRepMesh_IncrementalMesh(face.topods_shape(), deflection)
    
    loc = TopLoc_Location()
    triangulation = BRep_Tool.Triangulation(face.topods_shape(), loc)
    
    if triangulation is None:
        return None
        
    trsf = loc.Transformation()
    
    # Extract Nodes
    nodes = []
    for i in range(1, triangulation.NbNodes() + 1):
        pnt = triangulation.Node(i).Transformed(trsf)
        nodes.append([pnt.X(), pnt.Y(), pnt.Z()])
    nodes = np.array(nodes)
    
    # Extract Triangles
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
    
    # ==========================================================
    # 核心修改：将 Face ID 写入到网格的 Cell Data (三角形数据) 中
    # 这就像给每个三角形涂上了代表 ID 的颜色代码
    # ==========================================================
    # 创建一个与三角形数量相同的数组，全部填充为 face_id
    id_array = np.full(mesh.n_cells, face_id, dtype=np.int32)
    mesh.cell_data["FaceID"] = id_array
    
    # 计算法线保证渲染光滑
    mesh = mesh.compute_normals(cell_normals=False, point_normals=True)
    
    return mesh

class CADPickerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CAD Face Picker (G-Buffer Logic)")
        self.setGeometry(100, 100, 1200, 800)
        
        self.current_solid = None
        self.entity_mapper = None
        
        # 存储 FaceID -> 原始 occwl.Face 对象
        self.id_to_face = {} 
        # 存储 FaceID -> PyVista Mesh (用于单独高亮显示)
        self.id_to_mesh = {}
        
        self.highlight_actor = None
        
        self.init_ui()
        
    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()
        main_widget.setLayout(layout)
        
        # Toolbar
        info_layout = QVBoxLayout()
        self.btn_load = QPushButton("Load STEP File")
        self.btn_load.clicked.connect(self.load_step_file)
        self.btn_load.setFixedHeight(40)
        info_layout.addWidget(self.btn_load)
        
        self.lbl_info = QLabel("Please load a STEP file.")
        info_layout.addWidget(self.lbl_info)
        
        self.lbl_details = QLabel("Selected Face ID: None")
        self.lbl_details.setStyleSheet("font-size: 16px; color: blue; padding: 5px;")
        info_layout.addWidget(self.lbl_details)
        
        layout.addLayout(info_layout)
        
        # Plotter
        self.plotter = BackgroundPlotter(show=False)
        layout.addWidget(self.plotter)
        
        # ==========================================================
        # 核心修改：使用 Cell Picking (单元拾取)
        # 这会返回被点击的具体三角形，而不是 Actor
        # through=False 表示只拾取最表面的，不穿透
        # ==========================================================
        self.plotter.enable_cell_picking(
            callback=self.on_cell_picked,
            through=False,
            show=False,
            show_message=False,
            color='pink' # 调试时临时高亮颜色，我们会在回调里自己处理高亮
        )
        
    def load_step_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open STEP File", "", "STEP Files (*.stp *.step)")
        if not filename:
            return
            
        self.lbl_info.setText(f"Loading {os.path.basename(filename)}...")
        QApplication.processEvents()
        
        try:
            loaded_obj = Compound.load_from_step(filename)
            if isinstance(loaded_obj, Compound):
                solids = list(loaded_obj.solids())
                if not solids and isinstance(loaded_obj, Solid):
                     solids = [loaded_obj]
            else:
                solids = list(loaded_obj)
            
            # 简单处理：取第一个实体
            self.current_solid = None
            for s in solids:
                self.current_solid = s
                break
                
            if self.current_solid is None:
                self.lbl_info.setText("No valid Solid found.")
                return
                
            self.entity_mapper = EntityMapper(self.current_solid)
            self.visualize_solid()
            self.lbl_info.setText(f"Loaded: {os.path.basename(filename)}")
            
        except Exception as e:
            self.lbl_info.setText(f"Error: {str(e)}")
            print(e)

    def visualize_solid(self):
        self.plotter.clear()
        self.id_to_face = {}
        self.id_to_mesh = {}
        self.highlight_actor = None
        
        faces = list(self.current_solid.faces())
        print(f"Processing {len(faces)} faces...")
        
        # 收集所有子网格
        all_meshes = []
        
        for face in faces:
            face_id = self.entity_mapper.face_index(face)
            
            # 1. 生成网格并打上 ID 标签
            mesh = triangulate_face(face, face_id)
            
            if mesh and mesh.n_points > 0:
                self.id_to_face[face_id] = face
                self.id_to_mesh[face_id] = mesh # 保存一份副本用于高亮
                all_meshes.append(mesh)
        
        if not all_meshes:
            return

        # 2. 关键步骤：将所有面的网格合并成一个单一的大网格 (Merge)
        # 这样在渲染器中只有一个 Actor，不会有包围盒重叠产生的拾取混淆
        print("Merging meshes...")
        combined_mesh = all_meshes[0].merge(all_meshes[1:])
        
        # 3. 添加这个合并后的主体
        self.plotter.add_mesh(
            combined_mesh,
            color="lightgrey",
            show_edges=False,
            specular=0.5,
            smooth_shading=True,
            pickable=True, # 允许拾取这个大物体
            name="main_body"
        )
        
        # 4. 为了视觉效果，添加黑色轮廓线 (可选)
        # 由于我们合并了网格，直接提取整个实体的 Feature Edges 效率更高
        try:
            edges = combined_mesh.extract_feature_edges(
                boundary_edges=True,
                non_manifold_edges=True,
                feature_edges=True,
                manifold_edges=False,
                feature_angle=30
            )
            self.plotter.add_mesh(edges, color="black", line_width=1.5, pickable=False)
        except Exception:
            pass
            
        self.plotter.reset_camera()

    def on_cell_picked(self, picked_mesh, cell_id):
        """
        picked_mesh: 被点击的那个大网格对象
        cell_id: 被点击的那个三角形在整个大网格中的索引
        """
        if picked_mesh is None or cell_id is None or cell_id < 0:
            return
            
        # 1. 从合并网格的 Cell Data 中读取我们预存的 FaceID
        try:
            face_ids = picked_mesh.cell_data["FaceID"]
            selected_face_id = face_ids[cell_id]
        except KeyError:
            print("Picked mesh has no FaceID data.")
            return

        print(f"Debug: Hit Triangle {cell_id} -> Face ID {selected_face_id}")
        
        # 2. 获取原始几何信息
        if selected_face_id in self.id_to_face:
            face = self.id_to_face[selected_face_id]
            
            # 计算面积 (证明我们找对了面)
            try:
                props = GProp_GProps()
                brepgprop_SurfaceProperties(face.topods_shape(), props)
                area = props.Mass()
            except:
                area = 0.0
                
            self.lbl_details.setText(f"Selected Face ID: {selected_face_id} | Area: {area:.4f}")
            
            # 3. 高亮逻辑
            # 我们不改变大网格的颜色，而是在大网格之上叠加绘制被选中的那个面的网格
            # 先移除旧的高亮
            if self.highlight_actor:
                self.plotter.remove_actor(self.highlight_actor)
            
            # 获取对应的独立网格
            target_mesh = self.id_to_mesh[selected_face_id]
            
            # 叠加绘制红色高亮面
            # 稍微偏移一点点 (offset) 防止 Z-fighting (闪烁)
            # 或者利用 PyVista 的 render_lines_as_tubes 等特性，
            # 但这里最简单的是直接绘制红色，通常 PyVista 会处理得不错。
            self.highlight_actor = self.plotter.add_mesh(
                target_mesh,
                color="red",
                show_edges=False,
                lighting=False, # 纯色无光照，高亮更明显
                pickable=False, # 高亮层不参与拾取
                name="highlight"
            )
        else:
            self.lbl_details.setText(f"Face ID {selected_face_id} not found in map.")

def main():
    # 依然推荐开启 High DPI，虽然本方法的算法不受分辨率影响，但UI会更好看
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    window = CADPickerWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()