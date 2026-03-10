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
                             QPushButton, QLabel, QFileDialog, QDockWidget, QTextEdit)
from PyQt5.QtCore import Qt

# occwl imports
from occwl.compound import Compound
from occwl.entity_mapper import EntityMapper
from occwl.face import Face
from occwl.solid import Solid

# OCC imports for triangulation
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import topods_Face
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop_SurfaceProperties

def triangulate_face(face, deflection=0.001):
    """
    Triangulate an occwl Face for visualization using PyVista.
    """
    # Ensure triangulation exists
    # 使用更小的 deflection (0.001) 来获得更精细的网格
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
        # OCC uses 1-based indexing, PyVista uses 0-based
        n1, n2, n3 = tri.Get()
        # PyVista format: [3, v1, v2, v3] for each triangle
        triangles.append([3, n1 - 1, n2 - 1, n3 - 1])
    
    if not triangles:
        return None
        
    triangles = np.hstack(triangles)
    
    # Create PyVista mesh
    mesh = pv.PolyData(nodes, triangles)
    
    # 计算顶点法线以实现平滑着色
    mesh = mesh.compute_normals(cell_normals=False, point_normals=True)
    
    return mesh

class CADPickerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CAD Face Picker (EntityMapper ID)")
        self.setGeometry(100, 100, 1200, 800)
        
        # Data storage
        self.current_solid = None
        self.entity_mapper = None
        self.actor_to_face_info = {}  # Changed from actor_to_face_id to store (id, face)
        self.selected_actor = None
        self.original_colors = {}
        
        self.init_ui()
        
    def init_ui(self):
        # Main layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()
        main_widget.setLayout(layout)
        
        # Toolbar / Info area
        info_layout = QVBoxLayout()
        
        self.btn_load = QPushButton("Load STEP File")
        self.btn_load.clicked.connect(self.load_step_file)
        info_layout.addWidget(self.btn_load)
        
        self.lbl_info = QLabel("Please load a STEP file.")
        self.lbl_info.setStyleSheet("font-size: 14px; font-weight: bold; padding: 5px;")
        info_layout.addWidget(self.lbl_info)
        
        self.lbl_details = QLabel("Selected Face ID: None")
        self.lbl_details.setStyleSheet("font-size: 16px; color: blue; padding: 5px;")
        info_layout.addWidget(self.lbl_details)
        
        layout.addLayout(info_layout)
        
        # PyVista Plotter
        self.plotter = BackgroundPlotter(show=False)
        layout.addWidget(self.plotter)
        
        # Enable picking
        # 回退到 enable_mesh_picking，因为 enable_cell_picking 在某些版本中可能无响应
        # 我们通过打印调试信息来排查误判问题
        # self.plotter.enable_mesh_picking(
        #     callback=self.on_pick, 
        #     show=False, 
        #     show_message=False, 
        #     use_actor=True,     # 直接返回 Actor
        #     left_clicking=True  # 左键触发
        # )

        self.plotter.iren.add_observer("LeftButtonPressEvent", self.on_left_click)
        
    def load_step_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open STEP File", "", "STEP Files (*.stp *.step)")
        if not filename:
            return
            
        self.lbl_info.setText(f"Loading {os.path.basename(filename)}...")
        QApplication.processEvents()
        
        try:
            # Load solids using occwl
            # Compound.load_from_step returns a single Compound object
            loaded_obj = Compound.load_from_step(filename)
            
            # Handle return type (single Compound vs list)
            if isinstance(loaded_obj, Compound):
                solids = [loaded_obj]
            else:
                solids = list(loaded_obj)
            
            if not solids:
                self.lbl_info.setText("No solids found in file.")
                return
                
            # For simplicity, we process the first solid found, or merge them if needed.
            # The reference code usually processes one solid at a time.
            # Let's take the first valid solid.
            self.current_solid = None
            for s in solids:
                if isinstance(s, Solid):
                    self.current_solid = s
                    break
                elif isinstance(s, Compound):
                    # Try to get solids from compound
                    sub_solids = list(s.solids())
                    if sub_solids:
                        self.current_solid = sub_solids[0]
                        break
            
            if self.current_solid is None:
                self.lbl_info.setText("No valid Solid found.")
                return
                
            # Initialize EntityMapper (CRITICAL: This matches the reference logic)
            self.entity_mapper = EntityMapper(self.current_solid)
            
            self.visualize_solid()
            self.lbl_info.setText(f"Loaded: {os.path.basename(filename)} | Faces: {len(list(self.current_solid.faces()))}")
            
        except Exception as e:
            self.lbl_info.setText(f"Error loading file: {str(e)}")
            print(e)

    def visualize_solid(self):
        self.plotter.clear()
        self.plotter.enable_lightkit() 
        self.actor_to_face_info = {}
        self.original_colors = {}
        
        # Iterate over faces and visualize them
        # We use the mapper to get the ID
        
        faces = list(self.current_solid.faces())
        print(f"Visualizing {len(faces)} faces...")
        
        for face in faces:
            # Get ID from EntityMapper
            face_id = self.entity_mapper.face_index(face)
            
            # Triangulate
            mesh = triangulate_face(face)
            if mesh:
                # Add to plotter
                # 优化渲染效果：关闭 show_edges 以隐藏三角网格，仅显示光滑曲面
                actor = self.plotter.add_mesh(
                    mesh, 
                    color="lightgrey", 
                    show_edges=False,    # <--- 关键修改：关闭三角网格显示
                    pickable=True,
                    
                    # Blinn-phong
                    diffuse=0.8,         # 漫反射强度 (0-1)
                    specular=0.6,        # 高光反射强度 (0-1)
                    specular_power=30.0, # 高光锐度/光泽度 (值越大，高光越集中/锐利)
                    ambient=0.15,        # 环境光强度 (防止背光面全黑)
                    smooth_shading=True  # 平滑着色
                )
                
                # Store mapping
                self.actor_to_face_info[actor] = (face_id, face)
                self.original_colors[actor] = "lightgrey"

                # 添加边界线 (模拟 CAD 软件的视觉效果：光滑曲面 + 黑色轮廓线)
                # 提取网格的边界边 (Boundary Edges)
                try:
                    edges = mesh.extract_feature_edges(
                        boundary_edges=True,
                        non_manifold_edges=False,
                        feature_edges=False,
                        manifold_edges=False
                    )
                    
                    # 如果提取到了边界线，将其绘制为黑色线条
                    if edges.n_points > 0:
                        self.plotter.add_mesh(
                            edges,
                            color="black",
                            line_width=1.5,
                            pickable=False  # 边界线不参与拾取，避免干扰
                        )
                except Exception as e:
                    print(f"Warning: Failed to extract edges for face {face_id}: {e}")
        
        self.plotter.reset_camera()

    def on_pick(self, picked_actor):
        # Debug info
        print(f"Debug: Pick event. Actor: {picked_actor}")
        
        if picked_actor is None:
            return

        # Reset previous highlight
        if self.selected_actor and self.selected_actor in self.original_colors:
            try:
                self.selected_actor.prop.color = self.original_colors[self.selected_actor]
            except:
                pass 
        
        if picked_actor in self.actor_to_face_info:
            face_id, face = self.actor_to_face_info[picked_actor]
            
            # Calculate area
            try:
                props = GProp_GProps()
                brepgprop_SurfaceProperties(face.topods_shape(), props)
                area = props.Mass()
            except Exception as e:
                print(f"Error calculating area: {e}")
                area = 0.0
            
            print(f"Debug: Selected Face ID {face_id}")
            self.lbl_details.setText(f"Selected Face ID: {face_id} | Area: {area:.4f}")
            
            # Highlight
            picked_actor.prop.color = "red"
            self.selected_actor = picked_actor
        else:
            print("Debug: Picked actor not in face map (maybe an edge or other object?)")
            self.lbl_details.setText("Selected Face ID: None")
            self.selected_actor = None

    # [修改 3/3] 添加以下两个方法，替换原来的 on_pick 逻辑

    def on_left_click(self, obj, event):
        """
        G-Buffer (Hardware) Picking implementation.
        """
        # 1. 获取鼠标在窗口中的像素位置
        click_pos = self.plotter.iren.get_event_position()
        x, y = click_pos[0], click_pos[1]

        # 2. 初始化硬件选择器 (HardwareSelector)
        # 这就是你要的 G-Buffer 模式：显卡渲染 ID 图并读取像素
        selector = vtk.vtkHardwareSelector()
        selector.SetRenderer(self.plotter.renderer)
        selector.SetArea(x, y, x, y) # 只读取 1x1 的像素区域
        selector.SetFieldAssociation(vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS)

        # 3. 执行拾取 (CaptureBuffers)
        selection = selector.Select()
        
        picked_actor = None
        
        # 4. 解析结果
        if selection and selection.GetNumberOfNodes() > 0:
            selection_node = selection.GetNode(0)
            # 从选择结果中提取 Actor (PROP)
            picked_actor = selection_node.GetProperties().Get(vtk.vtkSelectionNode.PROP())
        
        # 5. 调用处理逻辑
        self.handle_pick_result(picked_actor)

    def handle_pick_result(self, picked_actor):
        """
        处理拿到 Actor 后的高亮和数据显示逻辑 (原 on_pick 的逻辑)
        """
        # 1. 如果点击空白处 (picked_actor is None)
        if picked_actor is None:
            # 这里的逻辑主要为了防止误触，不做清空操作，或者你可以选择清空
            return

        # 2. 检查是否是我们记录的面 (过滤掉边界线或其他杂物)
        if picked_actor not in self.actor_to_face_info:
            print("Debug: Picked object is not a registered Face (might be an edge).")
            return

        # 3. 恢复上一个选中对象的颜色
        if self.selected_actor and self.selected_actor in self.original_colors:
            try:
                self.selected_actor.prop.color = self.original_colors[self.selected_actor]
            except:
                pass # Actor 可能已被删除

        # 4. 获取信息
        face_id, face = self.actor_to_face_info[picked_actor]
        
        # 计算面积
        try:
            props = GProp_GProps()
            brepgprop_SurfaceProperties(face.topods_shape(), props)
            area = props.Mass()
        except Exception:
            area = 0.0
        
        print(f"Debug: Selected Face ID {face_id}")
        self.lbl_details.setText(f"Selected Face ID: {face_id} | Area: {area:.4f}")
        
        # 5. 高亮新选中的对象 (红色)
        picked_actor.prop.color = "red"
        self.selected_actor = picked_actor
        
        # 强制刷新窗口以立即显示颜色变化
        self.plotter.render()

def main():
    app = QApplication(sys.argv)
    window = CADPickerWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
