import numpy as np
import os
import sys
import random
import re
import argparse
import traceback
import pyvista as pv
import matplotlib.colors as mcolors # [新增] 用于生成渐变色

from occwl.compound import Compound
from occwl.entity_mapper import EntityMapper
from occwl.solid import Solid

# OCC Core Imports
from OCC.Core.Bnd import Bnd_OBB
from OCC.Core.BRepBndLib import brepbndlib_AddOBB
from OCC.Core.gp import gp_Pnt, gp_Pnt2d
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepTools import breptools_Clean, breptools_UVBounds
from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf

# ==========================================
# 辅助函数：三角化 Face
# ==========================================
def triangulate_face(face, deflection=1e-3):
    breptools_Clean(face.topods_shape())
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
        self.left_geom_surf = None
        self.right_geom_surf = None
        self.obb_object = None 
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
        if self.left_id not in index_to_face: raise ValueError(f"Left ID {self.left_id} not found.")
        if self.right_id not in index_to_face: raise ValueError(f"Right ID {self.right_id} not found.")
        self.left_face = index_to_face[self.left_id]
        self.right_face = index_to_face[self.right_id]
        self.left_geom_surf = BRep_Tool.Surface(self.left_face.topods_shape())
        self.right_geom_surf = BRep_Tool.Surface(self.right_face.topods_shape())

    def compute_obb(self):
        obb = Bnd_OBB()
        brepbndlib_AddOBB(self.left_face.topods_shape(), obb)
        brepbndlib_AddOBB(self.right_face.topods_shape(), obb)
        self.obb_object = obb
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

    def _get_obb_corners(self):
        corners = []
        for i in [0, 1]:
            for j in [0, 1]:
                for k in [0, 1]:
                    pt_np = self.obb_origin + (i * self.obb_x_vec) + (j * self.obb_y_vec) + (k * self.obb_z_vec)
                    corners.append(gp_Pnt(float(pt_np[0]), float(pt_np[1]), float(pt_np[2])))
        return corners

    def to_world(self, points_os):
        points_ws = []
        for p in points_os:
            ws = (self.obb_origin + p[0]*self.obb_x_vec + p[1]*self.obb_y_vec + p[2]*self.obb_z_vec)
            points_ws.append(ws)
        return np.array(points_ws)

    def sample_in_obb(self):
        samples_data = [] 
        random_points_os = np.random.rand(self.sample_num, 3)
        proj_l = GeomAPI_ProjectPointOnSurf(gp_Pnt(0,0,0), self.left_geom_surf)
        proj_r = GeomAPI_ProjectPointOnSurf(gp_Pnt(0,0,0), self.right_geom_surf)
        
        for i in range(self.sample_num):
            pt_os = random_points_os[i]
            pt_ws_np = (self.obb_origin + pt_os[0]*self.obb_x_vec + pt_os[1]*self.obb_y_vec + pt_os[2]*self.obb_z_vec)
            pt_occ = gp_Pnt(float(pt_ws_np[0]), float(pt_ws_np[1]), float(pt_ws_np[2]))

            proj_l.Perform(pt_occ)
            if not proj_l.IsDone(): continue 
            dist_l = proj_l.LowerDistance()
            closest_pt_l = proj_l.NearestPoint()
            
            proj_r.Perform(pt_occ)
            if not proj_r.IsDone(): continue
            dist_r = proj_r.LowerDistance()
            closest_pt_r = proj_r.NearestPoint()
            
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

    def sample_on_surface_in_obb(self, face, geom_surf, count):
        points_os = [] 
        u_min, u_max, v_min, v_max = breptools_UVBounds(face.topods_shape())
        corners = self._get_obb_corners()
        proj = GeomAPI_ProjectPointOnSurf(gp_Pnt(0,0,0), geom_surf)
        
        for corner_pt in corners:
            try:
                proj.Perform(corner_pt)
                if proj.IsDone():
                    u, v = proj.LowerDistanceParameters()
                    if u < u_min: u_min = u
                    if u > u_max: u_max = u
                    if v < v_min: v_min = v
                    if v > v_max: v_max = v
            except: pass
        
        u_span = u_max - u_min
        v_span = v_max - v_min
        if u_span < 1e-6: u_span = 1.0
        if v_span < 1e-6: v_span = 1.0
        
        padding = 0.1 
        u_min -= u_span * padding
        u_max += u_span * padding
        v_min -= v_span * padding
        v_max += v_span * padding
        
        len_x = np.linalg.norm(self.obb_x_vec)
        len_y = np.linalg.norm(self.obb_y_vec)
        len_z = np.linalg.norm(self.obb_z_vec)
        axis_x = self.obb_x_vec / len_x if len_x > 0 else np.array([1,0,0])
        axis_y = self.obb_y_vec / len_y if len_y > 0 else np.array([0,1,0])
        axis_z = self.obb_z_vec / len_z if len_z > 0 else np.array([0,0,1])

        valid_count = 0
        attempts = 0
        max_attempts = count * 50 

        while valid_count < count and attempts < max_attempts:
            attempts += 1
            u = random.uniform(u_min, u_max)
            v = random.uniform(v_min, v_max)
            pnt = geom_surf.Value(u, v)
            if not self.obb_object.IsOut(pnt):
                p_rel = np.array([pnt.X(), pnt.Y(), pnt.Z()]) - self.obb_origin
                x_norm = np.dot(p_rel, axis_x) / len_x
                y_norm = np.dot(p_rel, axis_y) / len_y
                z_norm = np.dot(p_rel, axis_z) / len_z
                points_os.append([x_norm, y_norm, z_norm])
                valid_count += 1
        return np.array(points_os)

    def export_data(self, output_path, samples_data, left_pts, right_pts, relative_root=None):
        try:
            if relative_root:
                model_path_str = os.path.relpath(self.step_path, relative_root)
            else:
                model_path_str = self.step_path

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"ModelPath: {model_path_str}\n")
                f.write(f"LeftFaceID: {self.left_id}\n")
                f.write(f"RightFaceID: {self.right_id}\n")
                f.write(f"SampleCount: {self.sample_num}\n\n")
                f.write("SAMPLES_DATA (Format: x y z o v)\n")
                for d in samples_data: f.write(f"{d[0]:.6f} {d[1]:.6f} {d[2]:.6f} {d[3]:.6f} {d[4]:.6f}\n")
                f.write("\nLEFT_POINTS (Format: x y z)\n")
                for p in left_pts: f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
                f.write("\nRIGHT_POINTS (Format: x y z)\n")
                for p in right_pts: f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
            print(f"[Success] Data saved to {output_path}")
        except Exception as e:
            print(f"[Error] Failed to save file: {e}")

# ==========================================
# 批量处理逻辑 (保持不变)
# ==========================================
def parse_face_pairs(txt_path):
    pairs = []
    if not os.path.exists(txt_path): return pairs
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if not line: continue
        parts = re.split(r'[,\s]+', line)
        parts = [p for p in parts if p]
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            pairs.append((int(parts[0]), int(parts[1])))
    return pairs

def run_batch_process(root_dir, sample_num):
    print(f"=== Starting Batch Process (Surface+OBB Smart Mode) ===")
    total_files = 0
    total_pairs = 0
    for dirpath, dirnames, filenames in os.walk(root_dir):
        step_files = [f for f in filenames if f.lower().endswith(('.step', '.stp'))]
        for step_file in step_files:
            base_name = os.path.splitext(step_file)[0]
            txt_file = base_name + ".txt"
            if txt_file in filenames:
                step_full_path = os.path.join(dirpath, step_file)
                txt_full_path = os.path.join(dirpath, txt_file)
                print(f"\nProcessing: {step_full_path}")
                pairs = parse_face_pairs(txt_full_path)
                if not pairs: continue
                output_dir = os.path.join(dirpath, "pointCloudResult")
                if not os.path.exists(output_dir): os.makedirs(output_dir)
                try:
                    loaded_obj = Compound.load_from_step(step_full_path)
                    if isinstance(loaded_obj, Compound):
                        solids = list(loaded_obj.solids())
                        if not solids and isinstance(loaded_obj, Solid): solids = [loaded_obj]
                    elif isinstance(loaded_obj, Solid): solids = [loaded_obj]
                    else: solids = list(loaded_obj)
                    if not solids: continue
                    solid = solids[0]
                    mapper = EntityMapper(solid)
                    index_to_face = {}
                    for face in solid.faces():
                        idx = mapper.face_index(face)
                        index_to_face[idx] = face
                    for left_id, right_id in pairs:
                        if left_id not in index_to_face or right_id not in index_to_face: continue
                        out_fname = f"{base_name}_{left_id}_{right_id}.txt"
                        out_path = os.path.join(output_dir, out_fname)
                        gen = PointCloudGenerator(step_full_path, left_id, right_id, sample_num)
                        gen.solid = solid
                        gen.mapper = mapper
                        gen.left_face = index_to_face[left_id]
                        gen.right_face = index_to_face[right_id]
                        gen.left_geom_surf = BRep_Tool.Surface(gen.left_face.topods_shape())
                        gen.right_geom_surf = BRep_Tool.Surface(gen.right_face.topods_shape())
                        try:
                            gen.compute_obb()
                            vol = gen.sample_in_obb()
                            ls = gen.sample_on_surface_in_obb(gen.left_face, gen.left_geom_surf, sample_num)
                            rs = gen.sample_on_surface_in_obb(gen.right_face, gen.right_geom_surf, sample_num)
                            gen.export_data(out_path, vol, ls, rs, relative_root=root_dir)
                            total_pairs += 1
                        except Exception as e: print(f"  [Error] Pair {left_id}-{right_id}: {e}")
                    total_files += 1
                except Exception as e: print(f"  [Error] Loading STEP: {e}")
    print(f"\nBatch Finished. Files: {total_files}, Outputs: {total_pairs}")

# ==========================================
# [新增] 交互式可视化类
# ==========================================
class InteractiveVisualizer:
    def __init__(self, generator, vol_samples, left_samples, right_samples):
        self.gen = generator
        # 数据准备
        self.vol_ws = generator.to_world(vol_samples[:, :3])
        self.offsets = vol_samples[:, 3]
        self.v_vals = vol_samples[:, 4]
        self.left_ws = generator.to_world(left_samples)
        self.right_ws = generator.to_world(right_samples)
        
        # 状态
        self.scalar_mode = "offset" # 'offset' or 'v'
        self.color_theme_idx = 0
        
        # 预设颜色主题 [LeftColor, RightColor, VolColormapStart, VolColormapEnd]
        self.themes = [
            ["green", "yellow", "blue", "red"],      # 默认: Coolwarm风格
            ["cyan", "magenta", "black", "white"],   # 黑白风格
            ["blue", "red", "green", "purple"]       # 强对比风格
        ]
        
        # 初始化 Plotter
        self.plotter = pv.Plotter(title="Interactive Generator")
        
        # 1. 绘制背景模型
        self.draw_model()
        
        # 2. 绘制 OBB
        self.draw_obb()
        
        # 3. 绘制点云
        # 体素点 (Actor需要保存以便更新)
        self.vol_actor = self.plotter.add_points(
            self.vol_ws, 
            scalars=self.offsets, 
            cmap=self._get_custom_cmap(),
            point_size=6, 
            render_points_as_spheres=True,
            scalar_bar_args={'title': 'Offset'},
            lighting=False
        )
        
        # 表面点
        self.left_actor = None
        if len(self.left_ws) > 0:
            self.left_actor = self.plotter.add_points(
                self.left_ws, color=self.themes[0][0], 
                point_size=4, 
                render_points_as_spheres=True,
                lighting=False
                )
            
        self.right_actor = None
        if len(self.right_ws) > 0:
            self.right_actor = self.plotter.add_points(self.right_ws, color=self.themes[0][1], point_size=4, render_points_as_spheres=True)

        # 4. 添加 UI 控件
        self.add_ui_widgets()
        
        self.plotter.show()

    def _get_custom_cmap(self):
        """根据当前主题生成 matplotlib LinearSegmentedColormap"""
        c_start = self.themes[self.color_theme_idx][2]
        c_end = self.themes[self.color_theme_idx][3]
        return mcolors.LinearSegmentedColormap.from_list("custom", [c_start, c_end])

    def draw_model(self):
        for face in self.gen.solid.faces():
            mesh = triangulate_face(face)
            if mesh:
                idx = self.gen.mapper.face_index(face)
                color, opacity = "lightgrey", 0.3
                if idx == self.gen.left_id: color, opacity = "blue", 0.3
                elif idx == self.gen.right_id: color, opacity = "red", 0.3
                self.plotter.add_mesh(mesh, color=color, opacity=opacity, show_edges=False)

    def draw_obb(self):
        origin, vx, vy, vz = self.gen.obb_origin, self.gen.obb_x_vec, self.gen.obb_y_vec, self.gen.obb_z_vec
        def draw_line(p1, p2): self.plotter.add_mesh(pv.Line(p1, p2), color="black", line_width=2)
        pts = [origin, origin+vx, origin+vy, origin+vz, origin+vx+vy, origin+vx+vz, origin+vy+vz, origin+vx+vy+vz]
        draw_line(pts[0], pts[1]); draw_line(pts[0], pts[2]); draw_line(pts[0], pts[3])
        draw_line(pts[1], pts[4]); draw_line(pts[1], pts[5]); draw_line(pts[2], pts[4])
        draw_line(pts[2], pts[6]); draw_line(pts[3], pts[5]); draw_line(pts[3], pts[6])
        draw_line(pts[4], pts[7]); draw_line(pts[5], pts[7]); draw_line(pts[6], pts[7])

    def add_ui_widgets(self):
        # [滑条] 体素点半径
        def update_vol_size(val):
            if self.vol_actor: self.vol_actor.GetProperty().SetPointSize(val)
        self.plotter.add_slider_widget(update_vol_size, [1, 15], value=6, title="Vol Radius", pointa=(0.8, 0.9), pointb=(0.95, 0.9))
        
        # [滑条] 表面点半径
        def update_surf_size(val):
            if self.left_actor: self.left_actor.GetProperty().SetPointSize(val)
            if self.right_actor: self.right_actor.GetProperty().SetPointSize(val)
        self.plotter.add_slider_widget(update_surf_size, [1, 15], value=4, title="Surf Radius", pointa=(0.8, 0.75), pointb=(0.95, 0.75))
        
        # [按钮] 切换着色模式 (Offset / V)
        def toggle_mode(state):
            self.scalar_mode = "v" if state else "offset"
            
            # 1. 获取新数据
            new_scalars = self.v_vals if self.scalar_mode == "v" else self.offsets
            title = "V Value" if self.scalar_mode == "v" else "Offset"
            
            # 2. 更新网格上的数据
            self.plotter.update_scalars(new_scalars, mesh=self.vol_actor.mapper.dataset)
            
            # 3. [新增] 动态计算新数据的上下界
            new_min = np.min(new_scalars)
            new_max = np.max(new_scalars)
            
            # 4. [新增] 强制更新 Mapper 的标尺范围 (Clim)
            self.vol_actor.mapper.scalar_range = (new_min, new_max)
            
            # 5. 更新标题并重绘
            self.plotter.scalar_bar.GetTitleTextProperty().SetVerticalJustificationToTop()
            self.plotter.scalar_bar.SetTitle(title)
            self.plotter.render()
            
        self.plotter.add_checkbox_button_widget(toggle_mode, position=(10, 700), size=30, border_size=2)
        self.plotter.add_text("Toggle Offset / V", position=(50, 705), font_size=12, color="black")

        # [按钮] 切换颜色主题 (简单模拟调色盘)
        # 既然无法轻易在纯VTK里做全功能调色盘，我们提供一套预设循环
        # 同时也满足了“左右面颜色”和“两端插值”的调整需求
        def cycle_colors(state):
            if not state: return
            self.color_theme_idx = (self.color_theme_idx + 1) % len(self.themes)
            theme = self.themes[self.color_theme_idx]
            
            # [修复] 使用 mcolors.to_rgb，它稳定返回 (r, g, b) 浮点元组
            if self.left_actor: 
                # theme[0] 是字符串 "green" 等
                c_left = mcolors.to_rgb(theme[0]) 
                self.left_actor.GetProperty().SetColor(c_left)
            
            if self.right_actor: 
                c_right = mcolors.to_rgb(theme[1])
                self.right_actor.GetProperty().SetColor(c_right)
            
            # 更新体素渐变色
            cmap = self._get_custom_cmap()
            self.vol_actor.mapper.lookup_table.cmap = cmap
            
            self.plotter.render()
            
        self.plotter.add_checkbox_button_widget(cycle_colors, position=(10, 650), size=30, border_size=2, color_on="blue", color_off="blue")
        self.plotter.add_text("Cycle Color Theme", position=(50, 655), font_size=12, color="black")


# ==========================================
# 主入口
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Point Cloud Generator.")
    parser.add_argument("--mode", choices=["single", "batch"], default="single", help="Operation mode.")
    parser.add_argument("--samples", type=int, default=1024, help="Number of samples.")
    
    # Batch 模式参数
    parser.add_argument("--root", type=str, help="Root directory for batch processing.")
    
    # [需求 1] Single 模式可选参数
    parser.add_argument("--step", type=str, help="STEP file path (Single Mode).")
    parser.add_argument("--left", type=int, help="Left Face ID.")
    parser.add_argument("--right", type=int, help="Right Face ID.")
    
    args = parser.parse_args()
    
    if args.mode == "batch":
        if not args.root:
            print("Error: --root is required for batch mode.")
        else:
            run_batch_process(args.root, args.samples)
            
    else: # Single Mode
        # [需求 1 实现] 参数逻辑检查
        # 默认值
        default_step = r"C:\Users\27800\Desktop\picker\model_data\90.step"
        default_left = 4
        default_right = 127
        
        target_step = default_step
        target_left = default_left
        target_right = default_right
        
        # 如果传入了 step，则必须传入 left 和 right
        if args.step:
            target_step = args.step
            if args.left is None or args.right is None:
                print("Error: If --step is provided, --left and --right are mandatory.")
                sys.exit(1)
            target_left = args.left
            target_right = args.right
        
        print(f"=== Running Single Mode ===")
        print(f"File: {target_step}")
        print(f"IDs: {target_left}, {target_right}")
        
        try:
            gen = PointCloudGenerator(target_step, target_left, target_right, sample_num=args.samples)
            
            print("1. Loading...")
            gen.load_model()
            
            print("2. Computing OBB...")
            gen.compute_obb()
            
            print("3. Sampling In OBB...")
            vol = gen.sample_in_obb()
            
            print("4. Sampling On Surface...")
            ls = gen.sample_on_surface_in_obb(gen.left_face, gen.left_geom_surf, args.samples)
            rs = gen.sample_on_surface_in_obb(gen.right_face, gen.right_geom_surf, args.samples)
            
            # Single 模式下不强制要求导出，但可保留
            OUTPUT_FILE = "testData_single_out.txt"
            gen.export_data(OUTPUT_FILE, vol, ls, rs)
            
            print("5. Visualizing (Interactive)...")
            # [需求 2,3,4 实现] 使用新的交互式可视化类
            InteractiveVisualizer(gen, vol, ls, rs)
            
        except Exception as e:
            traceback.print_exc()
            print(f"Error: {e}")