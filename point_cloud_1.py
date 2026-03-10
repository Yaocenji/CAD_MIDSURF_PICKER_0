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
from OCC.Core.gp import gp_Pnt, gp_Pnt2d
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
# 核心生成器类
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
        
        # 加载 STEP
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
        
        # 建立 ID 映射
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

    def export_data(self, output_path, samples_data, left_pts, right_pts, relative_root=None):
        """
        [修改] 增加了 relative_root 参数。
        如果传入了 relative_root，ModelPath 将写入相对路径。
        """
        try:
            # 计算路径
            if relative_root:
                # 获取 step_path 相对于 relative_root 的路径
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
    """
    解析 TXT 文件，提取有效行中的 (id1, id2)
    规则：
    1. 包含两个非负整数
    2. 分隔符可以是：逗号(,), 空格( ), 逗号+空格(, )
    3. 忽略空行、中文逗号行、非纯数字对行
    """
    pairs = []
    if not os.path.exists(txt_path):
        return pairs

    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for line_num, line in enumerate(lines):
        line = line.strip()
        if not line: continue
        
        # 使用正则分割：匹配逗号或空白字符
        # [,\s]+ 表示匹配一个或多个逗号或空白
        parts = re.split(r'[,\s]+', line)
        
        # 过滤空字符串（例如行首行尾有分隔符导致的情况）
        parts = [p for p in parts if p]
        
        # 检查是否只有两个元素，且都是数字
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            pairs.append((int(parts[0]), int(parts[1])))
        # else:
        #     print(f"  [Ignore Line {line_num+1}] Invalid format: {line}")
            
    return pairs

def run_batch_process(root_dir, sample_num):
    """
    递归遍历根目录，处理 step 和 txt
    """
    print(f"=== Starting Batch Process ===")
    print(f"Root Directory: {root_dir}")
    print(f"Sample Num: {sample_num}")
    
    total_files_processed = 0
    total_pairs_generated = 0
    
    # os.walk 递归遍历
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 查找所有 .step / .stp 文件
        step_files = [f for f in filenames if f.lower().endswith(('.step', '.stp'))]
        
        for step_file in step_files:
            base_name = os.path.splitext(step_file)[0]
            txt_file = base_name + ".txt"
            
            # 检查同级目录下是否有同名 txt
            if txt_file in filenames:
                step_full_path = os.path.join(dirpath, step_file)
                txt_full_path = os.path.join(dirpath, txt_file)
                
                print(f"\nProcessing: {step_full_path}")
                
                # 1. 解析面 ID 对
                pairs = parse_face_pairs(txt_full_path)
                if not pairs:
                    print(f"  No valid pairs found in {txt_file}")
                    continue
                
                # 2. 准备输出目录: 当前目录/pointCloudResult
                output_dir = os.path.join(dirpath, "pointCloudResult")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                    
                # 3. 逐个生成
                # 加载模型（只加载一次，提高效率）
                try:
                    # 这里的 left_id/right_id 先随便传一个占位，后面会单独检查
                    # 我们需要修改 Generator 逻辑使其支持懒加载或者重置 ID
                    # 但为了不破坏原来的类结构，我们在这里每次实例化一个新的，
                    # 或者，我们在循环里实例化。虽然加载模型会重复，但为了稳健性先这样。
                    # 优化方案：实例化一次，load_model一次，然后循环修改 ID 并 compute。
                    
                    # 预加载模型对象检查
                    loaded_obj = Compound.load_from_step(step_full_path)
                    if isinstance(loaded_obj, Compound):
                        solids = list(loaded_obj.solids())
                        if not solids and isinstance(loaded_obj, Solid): solids = [loaded_obj]
                    elif isinstance(loaded_obj, Solid): solids = [loaded_obj]
                    else: solids = list(loaded_obj)
                    
                    if not solids:
                        print("  [Error] No solids in STEP file.")
                        continue
                    
                    solid = solids[0]
                    mapper = EntityMapper(solid)
                    
                    # 建立 ID 索引 (一次性)
                    index_to_face = {}
                    for face in solid.faces():
                        idx = mapper.face_index(face)
                        index_to_face[idx] = face
                    
                    # 遍历 Face Pairs
                    for left_id, right_id in pairs:
                        # 检查 ID 是否存在
                        if left_id not in index_to_face or right_id not in index_to_face:
                            print(f"  [Skip] IDs {left_id},{right_id} not found in model.")
                            continue
                            
                        # 构造输出文件名: 原文件名_左id_右id.txt
                        out_fname = f"{base_name}_{left_id}_{right_id}.txt"
                        out_path = os.path.join(output_dir, out_fname)
                        
                        # 实例化生成器并注入数据 (复用已加载的数据以优化性能)
                        gen = PointCloudGenerator(step_full_path, left_id, right_id, sample_num)
                        # 手动注入已加载的 solid 和 mapper，避免重复 IO
                        gen.solid = solid
                        gen.mapper = mapper
                        gen.left_face = index_to_face[left_id]
                        gen.right_face = index_to_face[right_id]
                        
                        # 计算与导出
                        try:
                            gen.compute_obb()
                            vol = gen.sample_in_obb()
                            ls = gen.sample_on_face(gen.left_face, sample_num) # 表面采样数通常和体素一致或更少
                            rs = gen.sample_on_face(gen.right_face, sample_num)
                            
                            # 导出，传入 root_dir 以生成相对路径
                            gen.export_data(out_path, vol, ls, rs, relative_root=root_dir)
                            total_pairs_generated += 1
                        except Exception as e:
                            print(f"  [Error] Processing pair {left_id}-{right_id}: {e}")
                            
                    total_files_processed += 1
                    
                except Exception as e:
                    print(f"  [Error] Loading STEP file: {e}")

    print("\n=== Batch Process Finished ===")
    print(f"Files Processed: {total_files_processed}")
    print(f"Total Outputs: {total_pairs_generated}")


# ==========================================
# 可视化功能 (仅单文件模式使用)
# ==========================================
def visualize_result(generator, vol_samples, left_samples, right_samples):
    print("Preparing visualization...")
    p = pv.Plotter(window_size=[1200, 800], title="Single Mode Visualization")
    
    for face in generator.solid.faces():
        mesh = triangulate_face(face)
        if mesh:
            idx = generator.mapper.face_index(face)
            color, opacity = "lightgrey", 0.3
            if idx == generator.left_id: color, opacity = "blue", 0.6
            elif idx == generator.right_id: color, opacity = "red", 0.6
            p.add_mesh(mesh, color=color, opacity=opacity, smooth_shading=True, show_edges=False)

    vol_ws = generator.to_world(vol_samples[:, :3])
    offsets = vol_samples[:, 3]
    left_ws = generator.to_world(left_samples)
    right_ws = generator.to_world(right_samples)
    
    p.add_points(vol_ws, scalars=offsets, cmap="coolwarm", point_size=6, render_points_as_spheres=True)
    if len(left_ws) > 0: p.add_points(left_ws, color="green", point_size=4, render_points_as_spheres=True)
    if len(right_ws) > 0: p.add_points(right_ws, color="yellow", point_size=4, render_points_as_spheres=True)

    # Draw OBB Wireframe
    origin, vx, vy, vz = generator.obb_origin, generator.obb_x_vec, generator.obb_y_vec, generator.obb_z_vec
    def draw_line(p1, p2): p.add_mesh(pv.Line(p1, p2), color="black", line_width=2)
    pts = [origin, origin+vx, origin+vy, origin+vz, origin+vx+vy, origin+vx+vz, origin+vy+vz, origin+vx+vy+vz]
    # Connect
    draw_line(pts[0], pts[1]); draw_line(pts[0], pts[2]); draw_line(pts[0], pts[3])
    draw_line(pts[1], pts[4]); draw_line(pts[1], pts[5]); draw_line(pts[2], pts[4])
    draw_line(pts[2], pts[6]); draw_line(pts[3], pts[5]); draw_line(pts[3], pts[6])
    draw_line(pts[4], pts[7]); draw_line(pts[5], pts[7]); draw_line(pts[6], pts[7])

    p.add_axes()
    p.show()

# ==========================================
# 主入口
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Point Cloud Generator for STEP models.")
    
    # 模式选择
    parser.add_argument("--mode", choices=["single", "batch"], default="single", 
                        help="Operation mode: 'single' for one file with viz, 'batch' for directory scan.")
    
    # 通用参数
    parser.add_argument("--samples", type=int, default=1024, help="Number of samples (SampleCount).")
    
    # Batch 模式参数
    parser.add_argument("--root", type=str, help="Root directory for batch processing.")
    
    # Single 模式参数 (为了兼容旧用法，虽然也可以通过参数传，这里为了方便直接硬编码或保留原有逻辑)
    # 你可以修改下面的默认值为你常用的测试路径
    # 如果想通过命令行传参，可以取消注释下面的行
    # parser.add_argument("--step", type=str, help="STEP file path for single mode.")
    # parser.add_argument("--left", type=int, help="Left Face ID.")
    # parser.add_argument("--right", type=int, help="Right Face ID.")
    
    args = parser.parse_args()
    
    if args.mode == "batch":
        if not args.root:
            print("Error: --root argument is required for batch mode.")
        else:
            run_batch_process(args.root, args.samples)
            
    else: # Single Mode
        # 这里你可以修改为你原本的测试数据，或者接收命令行参数
        # 示例硬编码路径 (同你之前的文件)
        STEP_FILE = r"C:\Users\27800\Desktop\picker\test_clouder\0.step"  
        OUTPUT_FILE = "testData_visualized.txt"
        LEFT_ID = 8
        RIGHT_ID = 3
        
        try:
            print(f"=== Running Single Mode ===")
            gen = PointCloudGenerator(STEP_FILE, LEFT_ID, RIGHT_ID, sample_num=args.samples)
            
            print("1. Loading Model...")
            gen.load_model()
            print("2. Computing OBB...")
            gen.compute_obb()
            print("3. Sampling...")
            vol = gen.sample_in_obb()
            ls = gen.sample_on_face(gen.left_face, args.samples)
            rs = gen.sample_on_face(gen.right_face, args.samples)
            
            print("4. Exporting...")
            # 单文件模式下，不传 relative_root，保持绝对路径 (或根据需要修改)
            gen.export_data(OUTPUT_FILE, vol, ls, rs)
            
            print("5. Visualizing...")
            visualize_result(gen, vol, ls, rs)
            
        except Exception as e:
            traceback.print_exc()
            print(f"Error: {e}")