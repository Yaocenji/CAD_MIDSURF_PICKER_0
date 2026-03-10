import numpy as np
import os
import sys
import random
import re
import argparse
import traceback
import pyvista as pv

from occwl.compound import Compound
from occwl.entity_mapper import EntityMapper
from occwl.solid import Solid

# OCC Core Imports
from OCC.Core.Bnd import Bnd_OBB
from OCC.Core.BRepBndLib import brepbndlib_AddOBB
from OCC.Core.gp import gp_Pnt, gp_Pnt2d, gp_Dir, gp_Vec, gp_Ax2, gp_Trsf
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepTools import breptools_Clean, breptools_UVBounds
# [新增] 用于计算点到无限曲面的投影
from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf

# ==========================================
# 辅助函数：三角化 Face (用于可视化)
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
# 核心生成器类 (混合模式：Face OBB + Surface Distance)
# ==========================================
class PointCloudGenerator:
    def __init__(self, step_path, left_id, right_id, sample_num=1024):
        self.step_path = step_path
        self.left_id = int(left_id)
        self.right_id = int(right_id)
        self.sample_num = sample_num
        self.solid = None
        self.mapper = None
        
        # 原始 Face (用于计算 OBB 和 获取初始 UV 范围)
        self.left_face = None
        self.right_face = None
        
        # 底层 Geometry Surface (用于计算距离投影)
        self.left_geom_surf = None
        self.right_geom_surf = None
        
        # OBB 数据
        self.obb_object = None # 保存 OCC Bnd_OBB 对象用于后续判定
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
            
        if self.left_id not in index_to_face:
            raise ValueError(f"Left Face ID {self.left_id} not found.")
        if self.right_id not in index_to_face:
            raise ValueError(f"Right Face ID {self.right_id} not found.")
            
        self.left_face = index_to_face[self.left_id]
        self.right_face = index_to_face[self.right_id]

        # [逻辑 2] 提取底层几何 Surface
        self.left_geom_surf = BRep_Tool.Surface(self.left_face.topods_shape())
        self.right_geom_surf = BRep_Tool.Surface(self.right_face.topods_shape())

    def compute_obb(self):
        """
        [逻辑 1] 回退到使用原始 Face 计算 OBB
        这样保证 OBB 是有限的，且紧包围原本的两个面。
        """
        obb = Bnd_OBB()
        brepbndlib_AddOBB(self.left_face.topods_shape(), obb)
        brepbndlib_AddOBB(self.right_face.topods_shape(), obb)
        
        self.obb_object = obb # 保存对象用于 IsOut 判断
        
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
        points_ws = []
        for p in points_os:
            ws = (self.obb_origin + 
                  p[0] * self.obb_x_vec + 
                  p[1] * self.obb_y_vec + 
                  p[2] * self.obb_z_vec)
            points_ws.append(ws)
        return np.array(points_ws)

    def sample_in_obb(self):
        """
        [逻辑 2] 在 OBB 内体素采样，但计算到 Surface (无限曲面) 的距离
        """
        samples_data = [] 
        random_points_os = np.random.rand(self.sample_num, 3)
        
        # 初始化投影工具 (GeomAPI_ProjectPointOnSurf)
        # 这比 BRepExtrema 更适合计算到无限曲面的距离
        proj_l = GeomAPI_ProjectPointOnSurf(gp_Pnt(0,0,0), self.left_geom_surf)
        proj_r = GeomAPI_ProjectPointOnSurf(gp_Pnt(0,0,0), self.right_geom_surf)
        
        for i in range(self.sample_num):
            pt_os = random_points_os[i]
            pt_ws_np = (self.obb_origin + pt_os[0]*self.obb_x_vec + pt_os[1]*self.obb_y_vec + pt_os[2]*self.obb_z_vec)
            pt_occ = gp_Pnt(float(pt_ws_np[0]), float(pt_ws_np[1]), float(pt_ws_np[2]))

            # 计算到左曲面的投影
            proj_l.Perform(pt_occ)
            if not proj_l.IsDone(): continue # 极少情况投影失败
            dist_l = proj_l.LowerDistance()
            closest_pt_l = proj_l.NearestPoint()
            
            # 计算到右曲面的投影
            proj_r.Perform(pt_occ)
            if not proj_r.IsDone(): continue
            dist_r = proj_r.LowerDistance()
            closest_pt_r = proj_r.NearestPoint()
            
            # 计算 Offset
            total_dist = dist_l + dist_r
            offset = 0.5 if total_dist < 1e-9 else dist_l / total_dist
                
            # 计算 V (方向一致性)
            vec_l = np.array([closest_pt_l.X()-pt_ws_np[0], closest_pt_l.Y()-pt_ws_np[1], closest_pt_l.Z()-pt_ws_np[2]])
            vec_r = np.array([closest_pt_r.X()-pt_ws_np[0], closest_pt_r.Y()-pt_ws_np[1], closest_pt_r.Z()-pt_ws_np[2]])
            
            norm_l = np.linalg.norm(vec_l)
            norm_r = np.linalg.norm(vec_r)
            if norm_l > 1e-9: vec_l /= norm_l
            if norm_r > 1e-9: vec_r /= norm_r
            v_val = np.dot(vec_l, vec_r)
            
            samples_data.append([pt_os[0], pt_os[1], pt_os[2], offset, v_val])
            
        return np.array(samples_data)


    def _get_obb_corners(self):
            """
            辅助函数：计算 OBB 的 8 个角点 (世界坐标)
            """
            # OBB 中心和方向向量 (Halflength)
            center = self.obb_origin + 0.5 * self.obb_x_vec + 0.5 * self.obb_y_vec + 0.5 * self.obb_z_vec
            
            # 半长向量
            hx = 0.5 * self.obb_x_vec
            hy = 0.5 * self.obb_y_vec
            hz = 0.5 * self.obb_z_vec
            
            # 8个角点的组合 (+/-)
            corners = []
            for i in [-1, 1]:
                for j in [-1, 1]:
                    for k in [-1, 1]:
                        # 向量加法
                        pt = center + (i * hx) + (j * hy) + (k * hz)
                        corners.append(gp_Pnt(float(pt[0]), float(pt[1]), float(pt[2])))
            return corners


    def sample_on_surface_in_obb(self, face, geom_surf, count):
        """
        [逻辑升级] 
        1. 获取原始 Face 的 UV 范围。
        2. 获取 OBB 8 个角点，投影到 Surface 得到 UV。
        3. 取二者的并集范围，并给予少量 Padding (10%)。
        4. 在该范围内随机采样 -> OBB 过滤。
        """
        points_os = [] 
        
        # 1. 获取原始 Face 的 UV Bounds (作为基础范围)
        u_min, u_max, v_min, v_max = breptools_UVBounds(face.topods_shape())
        
        # 2. 计算 OBB 角点并投影
        corners = self._get_obb_corners()
        proj = GeomAPI_ProjectPointOnSurf(gp_Pnt(0,0,0), geom_surf)
        
        for corner_pt in corners:
            try:
                # 投影角点到曲面
                proj.Perform(corner_pt)
                if proj.IsDone():
                    u, v = proj.LowerDistanceParameters()
                    # 动态更新范围：取并集
                    if u < u_min: u_min = u
                    if u > u_max: u_max = u
                    if v < v_min: v_min = v
                    if v > v_max: v_max = v
            except:
                pass # 忽略投影失败的点（极少情况）
        
        # 3. 添加少量安全 Padding (10%) 以应对曲面在角点之间凸起的情况
        u_span = u_max - u_min
        v_span = v_max - v_min
        
        # 防止退化
        if u_span < 1e-6: u_span = 1.0
        if v_span < 1e-6: v_span = 1.0
        
        padding = 0.1 # 10%
        u_min -= u_span * padding
        u_max += u_span * padding
        v_min -= v_span * padding
        v_max += v_span * padding
        
        # 4. 采样循环
        # 预计算向量长度
        len_x = np.linalg.norm(self.obb_x_vec)
        len_y = np.linalg.norm(self.obb_y_vec)
        len_z = np.linalg.norm(self.obb_z_vec)
        axis_x = self.obb_x_vec / len_x if len_x > 0 else np.array([1,0,0])
        axis_y = self.obb_y_vec / len_y if len_y > 0 else np.array([0,1,0])
        axis_z = self.obb_z_vec / len_z if len_z > 0 else np.array([0,0,1])

        valid_count = 0
        attempts = 0
        # 因为现在的 UV 范围很精准，采样效率会很高，max_attempts 可以设小一点
        max_attempts = count * 50 

        while valid_count < count and attempts < max_attempts:
            attempts += 1
            
            # 随机 UV
            u = random.uniform(u_min, u_max)
            v = random.uniform(v_min, v_max)
            
            # 计算 3D 点
            pnt = geom_surf.Value(u, v)
            
            # OBB 过滤
            if not self.obb_object.IsOut(pnt):
                # 转回 OBB 归一化空间
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
# 批量处理逻辑
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
    print(f"=== Starting Batch Process (Surface+OBB Mode) ===")
    print(f"Root Directory: {root_dir}")
    
    total_files_processed = 0
    total_pairs_generated = 0
    
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
                if not pairs:
                    print(f"  No valid pairs found in {txt_file}")
                    continue
                
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
                        if left_id not in index_to_face or right_id not in index_to_face:
                            continue
                            
                        out_fname = f"{base_name}_{left_id}_{right_id}.txt"
                        out_path = os.path.join(output_dir, out_fname)
                        
                        gen = PointCloudGenerator(step_full_path, left_id, right_id, sample_num)
                        gen.solid = solid
                        gen.mapper = mapper
                        gen.left_face = index_to_face[left_id]
                        gen.right_face = index_to_face[right_id]
                        # 提取 Geom Surface
                        gen.left_geom_surf = BRep_Tool.Surface(gen.left_face.topods_shape())
                        gen.right_geom_surf = BRep_Tool.Surface(gen.right_face.topods_shape())

                        try:
                            gen.compute_obb()
                            vol = gen.sample_in_obb()
                            # [逻辑 3] 使用新采样函数
                            ls = gen.sample_on_surface_in_obb(gen.left_face, gen.left_geom_surf, sample_num)
                            rs = gen.sample_on_surface_in_obb(gen.right_face, gen.right_geom_surf, sample_num)
                            gen.export_data(out_path, vol, ls, rs, relative_root=root_dir)
                            total_pairs_generated += 1
                        except Exception as e:
                            print(f"  [Error] Pair {left_id}-{right_id}: {e}")
                            
                    total_files_processed += 1
                except Exception as e:
                    print(f"  [Error] Loading STEP: {e}")

    print("\n=== Batch Process Finished ===")
    print(f"Files Processed: {total_files_processed}")
    print(f"Total Outputs: {total_pairs_generated}")

# ==========================================
# 可视化功能
# ==========================================
def visualize_result(generator, vol_samples, left_samples, right_samples):
    print("Preparing visualization...")
    p = pv.Plotter(window_size=[1200, 800], title="Surface+OBB Mode Visualization")
    
    # Render Original Faces
    for face in generator.solid.faces():
        mesh = triangulate_face(face)
        if mesh:
            idx = generator.mapper.face_index(face)
            color, opacity = "lightgrey", 0.3
            if idx == generator.left_id: color, opacity = "blue", 0.5
            elif idx == generator.right_id: color, opacity = "red", 0.5
            p.add_mesh(mesh, color=color, opacity=opacity, smooth_shading=True, show_edges=False)

    vol_ws = generator.to_world(vol_samples[:, :3])
    offsets = vol_samples[:, 3]
    left_ws = generator.to_world(left_samples)
    right_ws = generator.to_world(right_samples)
    
    p.add_points(vol_ws, scalars=offsets, cmap="coolwarm", point_size=6, render_points_as_spheres=True)
    
    # 这里的点可能会超出原本 Face 的边界，但在 OBB 范围内，这是预期的
    if len(left_ws) > 0: p.add_points(left_ws, color="green", point_size=4, render_points_as_spheres=True, label="Left Surf")
    if len(right_ws) > 0: p.add_points(right_ws, color="yellow", point_size=4, render_points_as_spheres=True, label="Right Surf")

    origin, vx, vy, vz = generator.obb_origin, generator.obb_x_vec, generator.obb_y_vec, generator.obb_z_vec
    def draw_line(p1, p2): p.add_mesh(pv.Line(p1, p2), color="black", line_width=2)
    pts = [origin, origin+vx, origin+vy, origin+vz, origin+vx+vy, origin+vx+vz, origin+vy+vz, origin+vx+vy+vz]
    draw_line(pts[0], pts[1]); draw_line(pts[0], pts[2]); draw_line(pts[0], pts[3])
    draw_line(pts[1], pts[4]); draw_line(pts[1], pts[5]); draw_line(pts[2], pts[4])
    draw_line(pts[2], pts[6]); draw_line(pts[3], pts[5]); draw_line(pts[3], pts[6])
    draw_line(pts[4], pts[7]); draw_line(pts[5], pts[7]); draw_line(pts[6], pts[7])

    p.add_axes()
    p.add_legend()
    p.show()

# ==========================================
# 主入口
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Point Cloud Generator (Surface Mode).")
    parser.add_argument("--mode", choices=["single", "batch"], default="single", help="Operation mode.")
    parser.add_argument("--samples", type=int, default=1024, help="Number of samples.")
    parser.add_argument("--root", type=str, help="Root directory for batch processing.")
    
    args = parser.parse_args()
    
    if args.mode == "batch":
        if not args.root:
            print("Error: --root is required for batch mode.")
        else:
            run_batch_process(args.root, args.samples)
            
    else: # Single Mode
        # 硬编码测试参数
        STEP_FILE = r"C:\Users\27800\Desktop\picker\model_data\90.step"   # r"C:\Users\27800\Desktop\picker\model_data\0.step"  
        OUTPUT_FILE = "testData_surface_obb.txt"
        LEFT_ID = 4 #8
        RIGHT_ID = 127 #3
        
        try:
            print(f"=== Running Single Mode (Surface+OBB) ===")
            gen = PointCloudGenerator(STEP_FILE, LEFT_ID, RIGHT_ID, sample_num=args.samples)
            
            print("1. Loading Model...")
            gen.load_model()
            
            print("2. Computing OBB (Using Faces)...")
            gen.compute_obb()
            
            print("3. Sampling In OBB (Dist to Surface)...")
            vol = gen.sample_in_obb()
            
            print("4. Sampling On Surface (Clipped by OBB)...")
            ls = gen.sample_on_surface_in_obb(gen.left_face, gen.left_geom_surf, args.samples)
            rs = gen.sample_on_surface_in_obb(gen.right_face, gen.right_geom_surf, args.samples)
            
            print("5. Exporting...")
            gen.export_data(OUTPUT_FILE, vol, ls, rs)
            
            print("6. Visualizing...")
            visualize_result(gen, vol, ls, rs)
            
        except Exception as e:
            traceback.print_exc()
            print(f"Error: {e}")