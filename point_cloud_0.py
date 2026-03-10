import numpy as np
import os
import random
import pyvista as pv  # [新增] 引入 PyVista

from occwl.compound import Compound
from occwl.entity_mapper import EntityMapper
from occwl.solid import Solid

# OCC 核心库
from OCC.Core.Bnd import Bnd_OBB
from OCC.Core.BRepBndLib import brepbndlib_AddOBB
from OCC.Core.gp import gp_Pnt, gp_Pnt2d, gp_Dir, gp_Vec, gp_Ax2, gp_Trsf
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
from OCC.Core.BRepClass import BRepClass_FaceClassifier
from OCC.Core.TopAbs import TopAbs_IN, TopAbs_ON
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepTools import breptools_Clean

# ==========================================
# 辅助函数：三角化 Face 用于 PyVista 显示
# (逻辑源自你的 input_file_0.py)
# ==========================================
def triangulate_face(face, deflection=1e-3):
    breptools_Clean(face.topods_shape()) # 清除旧网格
    BRepMesh_IncrementalMesh(face.topods_shape(), deflection)
    loc = TopLoc_Location()
    triangulation = BRep_Tool.Triangulation(face.topods_shape(), loc)
    if triangulation is None: return None
        
    nodes = []
    for i in range(1, triangulation.NbNodes() + 1):
        pnt = triangulation.Node(i).Transformed(loc.Transformation())
        nodes.append([pnt.X(), pnt.Y(), pnt.Z()])
    nodes = np.array(nodes)
    
    triangles = []
    for i in range(1, triangulation.NbTriangles() + 1):
        tri = triangulation.Triangle(i)
        n1, n2, n3 = tri.Get()
        triangles.append([3, n1 - 1, n2 - 1, n3 - 1])
    
    if not triangles: return None
    return pv.PolyData(nodes, np.hstack(triangles))

# ==========================================
# 核心生成器类 (保持不变)
# ==========================================
class PointCloudGenerator:
    def __init__(self, step_path, left_id, right_id, sample_num=1024):
        self.step_path = step_path
        self.left_id = int(left_id)
        self.right_id = int(right_id)
        self.sample_num = sample_num
        self.solid = None
        self.mapper = None
        self.left_face = None
        self.right_face = None
        self.obb_origin = None
        self.obb_x_vec = None
        self.obb_y_vec = None
        self.obb_z_vec = None

    def load_model(self):
        if not os.path.exists(self.step_path):
            raise FileNotFoundError(f"File not found: {self.step_path}")
        loaded_obj = Compound.load_from_step(self.step_path)
        solids = []
        if isinstance(loaded_obj, Compound):
            solids = list(loaded_obj.solids())
            if not solids and isinstance(loaded_obj, Solid): solids = [loaded_obj]
        elif isinstance(loaded_obj, Solid): solids = [loaded_obj]
        else:
            try: solids = list(loaded_obj)
            except: pass
        if not solids: raise ValueError("No solids found")
            
        self.solid = solids[0]
        self.mapper = EntityMapper(self.solid)
        index_to_face = {}
        for face in self.solid.faces():
            idx = self.mapper.face_index(face)
            index_to_face[idx] = face
            
        if self.left_id not in index_to_face or self.right_id not in index_to_face:
            raise ValueError(f"ID not found. Available: {list(index_to_face.keys())[:5]}...")
        self.left_face = index_to_face[self.left_id]
        self.right_face = index_to_face[self.right_id]

    def compute_obb(self):
        obb = Bnd_OBB()
        brepbndlib_AddOBB(self.left_face.topods_shape(), obb)
        brepbndlib_AddOBB(self.right_face.topods_shape(), obb)
        center = obb.Center()
        xh, yh, zh = obb.XHSize(), obb.YHSize(), obb.ZHSize()
        x_dir, y_dir, z_dir = obb.XDirection(), obb.YDirection(), obb.ZDirection()
        
        self.obb_origin = np.array([
            center.X() - xh*x_dir.X() - yh*y_dir.X() - zh*z_dir.X(),
            center.Y() - xh*x_dir.Y() - yh*y_dir.Y() - zh*z_dir.Y(),
            center.Z() - xh*x_dir.Z() - yh*y_dir.Z() - zh*z_dir.Z()
        ])
        self.obb_x_vec = np.array([x_dir.X(), x_dir.Y(), x_dir.Z()]) * (2.0 * xh)
        self.obb_y_vec = np.array([y_dir.X(), y_dir.Y(), y_dir.Z()]) * (2.0 * yh)
        self.obb_z_vec = np.array([z_dir.X(), z_dir.Y(), z_dir.Z()]) * (2.0 * zh)

    def to_world(self, points_os):
        """辅助函数：将归一化坐标转回世界坐标用于显示"""
        points_ws = []
        for p in points_os:
            ws = (self.obb_origin + 
                  p[0] * self.obb_x_vec + 
                  p[1] * self.obb_y_vec + 
                  p[2] * self.obb_z_vec)
            points_ws.append(ws)
        return np.array(points_ws)

    def sample_in_obb(self):
        samples_data = [] 
        random_points_os = np.random.rand(self.sample_num, 3)
        dist_shape_l = self.left_face.topods_shape()
        dist_shape_r = self.right_face.topods_shape()
        
        for i in range(self.sample_num):
            pt_os = random_points_os[i]
            pt_ws_np = (self.obb_origin + pt_os[0]*self.obb_x_vec + pt_os[1]*self.obb_y_vec + pt_os[2]*self.obb_z_vec)
            occ_vertex = BRepBuilderAPI_MakeVertex(gp_Pnt(float(pt_ws_np[0]), float(pt_ws_np[1]), float(pt_ws_np[2]))).Vertex()

            dist_calc_l = BRepExtrema_DistShapeShape(dist_shape_l, occ_vertex)
            if not dist_calc_l.IsDone(): continue
            dist_l = dist_calc_l.Value()
            closest_pt_l = dist_calc_l.PointOnShape1(1)
            
            dist_calc_r = BRepExtrema_DistShapeShape(dist_shape_r, occ_vertex)
            if not dist_calc_r.IsDone(): continue
            dist_r = dist_calc_r.Value()
            closest_pt_r = dist_calc_r.PointOnShape1(1)
            
            total_dist = dist_l + dist_r
            offset = 0.5 if total_dist < 1e-9 else dist_l / total_dist
                
            vec_l = np.array([closest_pt_l.X()-pt_ws_np[0], closest_pt_l.Y()-pt_ws_np[1], closest_pt_l.Z()-pt_ws_np[2]])
            vec_r = np.array([closest_pt_r.X()-pt_ws_np[0], closest_pt_r.Y()-pt_ws_np[1], closest_pt_r.Z()-pt_ws_np[2]])
            
            norm_l = np.linalg.norm(vec_l)
            norm_r = np.linalg.norm(vec_r)
            if norm_l > 1e-9: vec_l /= norm_l
            if norm_r > 1e-9: vec_r /= norm_r
            v_val = np.dot(vec_l, vec_r)
            
            samples_data.append([pt_os[0], pt_os[1], pt_os[2], offset, v_val])
        return np.array(samples_data)

    def sample_on_face(self, face, count):
        points_os = [] 
        adaptor = BRepAdaptor_Surface(face.topods_shape())
        u_min, u_max, v_min, v_max = adaptor.FirstUParameter(), adaptor.LastUParameter(), adaptor.FirstVParameter(), adaptor.LastVParameter()
        classifier = BRepClass_FaceClassifier()
        valid_count = 0
        attempts = 0
        max_attempts = count * 200 
        
        len_x = np.linalg.norm(self.obb_x_vec)
        len_y = np.linalg.norm(self.obb_y_vec)
        len_z = np.linalg.norm(self.obb_z_vec)
        axis_x = self.obb_x_vec / len_x if len_x > 0 else np.array([1,0,0])
        axis_y = self.obb_y_vec / len_y if len_y > 0 else np.array([0,1,0])
        axis_z = self.obb_z_vec / len_z if len_z > 0 else np.array([0,0,1])

        while valid_count < count and attempts < max_attempts:
            attempts += 1
            u = random.uniform(u_min, u_max)
            v = random.uniform(v_min, v_max)
            classifier.Perform(face.topods_shape(), gp_Pnt2d(u, v), 1e-7)
            if classifier.State() in [TopAbs_IN, TopAbs_ON]:
                pnt = adaptor.Value(u, v)
                p_rel = np.array([pnt.X(), pnt.Y(), pnt.Z()]) - self.obb_origin
                x_norm = np.dot(p_rel, axis_x) / len_x
                y_norm = np.dot(p_rel, axis_y) / len_y
                z_norm = np.dot(p_rel, axis_z) / len_z
                points_os.append([x_norm, y_norm, z_norm])
                valid_count += 1
        return np.array(points_os)

    def export_data(self, output_path, samples_data, left_pts, right_pts):
        try:
            with open(output_path, 'w') as f:
                f.write(f"ModelPath: {self.step_path}\nSampleCount: {self.sample_num}\n\n")
                f.write("SAMPLES_DATA (Format: x y z o v)\n")
                for d in samples_data: f.write(f"{d[0]:.6f} {d[1]:.6f} {d[2]:.6f} {d[3]:.6f} {d[4]:.6f}\n")
                f.write("\nLEFT_POINTS (Format: x y z)\n")
                for p in left_pts: f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
                f.write("\nRIGHT_POINTS (Format: x y z)\n")
                for p in right_pts: f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
            print(f"Data saved to {output_path}")
        except Exception as e: print(f"Error saving: {e}")

# ==========================================
# [新增] 可视化功能
# ==========================================
def visualize_result(generator, vol_samples, left_samples, right_samples):
    """
    使用 PyVista 可视化生成结果
    """
    print("Preparing visualization...")
    p = pv.Plotter(window_size=[1200, 800], title="Dataset Generation Visualization")
    
    # 1. 绘制原始 CAD 模型 (半透明背景)
    for face in generator.solid.faces():
        mesh = triangulate_face(face)
        if mesh:
            # 默认灰色半透明
            color = "lightgrey"
            opacity = 0.3
            
            # 高亮选中的 Left/Right 面
            idx = generator.mapper.face_index(face)
            if idx == generator.left_id:
                color = "blue"
                opacity = 0.6
            elif idx == generator.right_id:
                color = "red"
                opacity = 0.6
                
            p.add_mesh(mesh, color=color, opacity=opacity, smooth_shading=True, show_edges=False)

    # 2. 转换坐标：归一化 -> 世界坐标
    # 因为生成的数据是 [0,1] 空间的，需要变换回 World Space 才能和 CAD 模型对齐
    
    # 体素点 (x,y,z)
    vol_ws = generator.to_world(vol_samples[:, :3])
    offsets = vol_samples[:, 3] # 取出 Offset 值用于染色
    
    # 左右面采样点
    left_ws = generator.to_world(left_samples)
    right_ws = generator.to_world(right_samples)
    
    # 3. 绘制体素点 (Volumetric Samples)
    # 使用 coolwarm 颜色映射：蓝色(0.0) -> 红色(1.0)
    # 0.0 代表接近左面，1.0 代表接近右面
    p.add_points(
        vol_ws, 
        scalars=offsets, 
        cmap="coolwarm", 
        point_size=3, 
        render_points_as_spheres=True,
        scalar_bar_args={'title': 'Offset Value'}
    )
    
    # 4. 绘制表面采样点 (Surface Samples)
    # 左面点 (绿色)
    if len(left_ws) > 0:
        p.add_points(left_ws, color="green", point_size=4, render_points_as_spheres=True, label="Left Surface Samples")
    
    # 右面点 (黄色)
    if len(right_ws) > 0:
        p.add_points(right_ws, color="yellow", point_size=4, render_points_as_spheres=True, label="Right Surface Samples")

    # 5. 绘制 OBB 线框
    # 构造 OBB 的 8 个顶点并绘制 Lines
    origin = generator.obb_origin
    vx = generator.obb_x_vec
    vy = generator.obb_y_vec
    vz = generator.obb_z_vec
    
    # 简易画法：创建一个 Box 网格并变换
    # 也可以手动画线
    def draw_line(p1, p2):
        line = pv.Line(p1, p2)
        p.add_mesh(line, color="black", line_width=2)

    # OBB 顶点
    p000 = origin
    p100 = origin + vx
    p010 = origin + vy
    p001 = origin + vz
    p110 = origin + vx + vy
    p101 = origin + vx + vz
    p011 = origin + vy + vz
    p111 = origin + vx + vy + vz
    
    # 绘制框线
    draw_line(p000, p100); draw_line(p000, p010); draw_line(p000, p001)
    draw_line(p100, p110); draw_line(p100, p101); draw_line(p010, p110)
    draw_line(p010, p011); draw_line(p001, p101); draw_line(p001, p011)
    draw_line(p110, p111); draw_line(p101, p111); draw_line(p011, p111)

    p.add_axes()
    p.add_legend()
    p.show()

# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    # 配置
    # 请务必修改为你电脑上的真实路径
    STEP_FILE = r"D:\MidSurf\Dataset_new\GrabCAD_Dataset\arduino-uno-r3-1.snapshot.5\0.step"  
    OUTPUT_FILE = "testData_visualized.txt"
    
    # 填入 pairs.txt 里的 ID
    # 示例 ID
    LEFT_ID = 22  
    RIGHT_ID = 16
    
    try:
        gen = PointCloudGenerator(STEP_FILE, LEFT_ID, RIGHT_ID, sample_num=2048) # 稍微增加点数看效果
        
        print("1. Loading Model...")
        gen.load_model()
        
        print("2. Computing OBB...")
        gen.compute_obb()
        
        print("3. Sampling Volumetric Points...")
        vol_samples = gen.sample_in_obb()
        
        print("4. Sampling Surface Points...")
        left_surf = gen.sample_on_face(gen.left_face, 500)
        right_surf = gen.sample_on_face(gen.right_face, 500)
        
        print("5. Exporting Data...")
        gen.export_data(OUTPUT_FILE, vol_samples, left_surf, right_surf)
        
        print("6. Visualizing...")
        visualize_result(gen, vol_samples, left_surf, right_surf)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error: {e}")