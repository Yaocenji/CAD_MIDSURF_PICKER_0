import os
import sys
from OCC.Core.STEPControl import STEPControl_Reader, STEPControl_Writer, STEPControl_AsIs
from OCC.Core.IGESControl import IGESControl_Reader
from OCC.Core.Interface import Interface_Static_SetCVal
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_SOLID, TopAbs_FACE, TopAbs_SHELL
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Sewing, BRepBuilderAPI_MakeSolid
from OCC.Core.TopoDS import topods_Shell

# ================= 配置区域 =================

# 1. 输入路径 (可以是文件夹，也可以是单个文件路径)
INPUT_PATH = r"D:\MidSurf\Dataset_new\GrabCAD_Dataset\falcon-f-16c-variant--1\Raw"
OUTPUT_FOLDER = r"D:\MidSurf\Dataset_new\GrabCAD_Dataset\falcon-f-16c-variant--1\Processed"

# 2. 去重开关
USE_DEDUPLICATION = True 

# 3. 日志模式: "FULL" (详细) 或 "SHORT" (简洁)
LOG_MODE = "SHORT"

# 4. 缝合容差 (单位: mm)
# 如果曲面之间有 0.1mm 以内的缝隙，尝试自动修补。如果模型很大且缝隙大，可适当调大到 1.0
SEWING_TOLERANCE = 0.1 

# ===========================================

def log(msg, level="FULL"):
    """自定义日志输出函数"""
    if LOG_MODE == "FULL":
        print(msg)
    elif LOG_MODE == "SHORT" and level == "SHORT":
        print(msg)

def load_file(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext in ['.stp', '.step']:
        reader = STEPControl_Reader()
    elif ext in ['.igs', '.iges']:
        reader = IGESControl_Reader()
    else:
        return None

    status = reader.ReadFile(filepath)
    if status == IFSelect_RetDone:
        reader.TransferRoots()
        return reader.OneShape()
    return None

def get_fingerprint(shape):
    """计算体积和表面积作为指纹"""
    try:
        v_props = GProp_GProps()
        brepgprop.VolumeProperties(shape, v_props)
        vol = v_props.Mass()

        s_props = GProp_GProps()
        brepgprop.SurfaceProperties(shape, s_props)
        area = s_props.Mass()
        return (round(vol, 5), round(area, 5))
    except:
        return (0, 0)

def write_step_ap214(shape, output_path):
    writer = STEPControl_Writer()
    Interface_Static_SetCVal("write.step.schema", "AP214") 
    status = writer.Transfer(shape, STEPControl_AsIs)
    if status == IFSelect_RetDone:
        writer.Write(output_path)
        return True
    return False

def attempt_sewing_to_solid(shape):
    """
    [新增功能] 尝试将散乱的曲面缝合成实体
    """
    log(f"  [尝试缝合] 正在收集面并进行缝合 (Tolerance={SEWING_TOLERANCE})...")
    
    # 1. 收集所有的 Face
    sewing_tool = BRepBuilderAPI_Sewing(SEWING_TOLERANCE)
    
    face_exp = TopExp_Explorer(shape, TopAbs_FACE)
    count = 0
    while face_exp.More():
        sewing_tool.Add(face_exp.Current())
        face_exp.Next()
        count += 1
    
    log(f"  [尝试缝合] 找到 {count} 个面，开始计算...")
    
    # 2. 执行缝合
    sewing_tool.Perform()
    sewed_shape = sewing_tool.SewedShape()
    
    # 3. 检查缝合结果
    # 缝合结果可能是一个 Shell，也可能是 Compound
    # 我们尝试遍历里面的 Shell，并把 Shell 变成 Solid
    
    new_solids = []
    shell_exp = TopExp_Explorer(sewed_shape, TopAbs_SHELL)
    
    while shell_exp.More():
        shell = topods_Shell(shell_exp.Current())
        # 尝试由壳生成实体
        maker = BRepBuilderAPI_MakeSolid(shell)
        if maker.IsDone():
            new_solids.append(maker.Solid())
        else:
            # 如果造不出实体（说明壳没闭合），我们至少保留这个壳，
            # 这样导出的 STEP 虽然不是实体，但在 CAD 软件里看起来是完整的曲面模型
            # 这是一个妥协方案
            log("  [缝合结果] 壳未完全闭合，保留为 Shell 导出。")
            new_solids.append(shell)
            
        shell_exp.Next()
        
    return new_solids

def main():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # 智能处理输入路径
    cad_files = []
    if os.path.isdir(INPUT_PATH):
        all_files = os.listdir(INPUT_PATH)
        supported_exts = ('.stp', '.step', '.igs', '.iges')
        cad_files = [os.path.join(INPUT_PATH, f) for f in all_files if f.lower().endswith(supported_exts)]
    elif os.path.isfile(INPUT_PATH):
        cad_files = [INPUT_PATH]
    else:
        print(f"错误：路径无效 -> {INPUT_PATH}")
        return

    log(f"找到 {len(cad_files)} 个源文件，准备处理...", "SHORT")
    
    global_counter = 0
    seen_geometries = set() 
    
    for full_path in cad_files:
        filename = os.path.basename(full_path)
        log(f"\n[{global_counter}] 正在处理文件: {filename}", "SHORT")
        
        main_shape = load_file(full_path)
        if not main_shape:
            log("  -> 读取失败", "SHORT")
            continue
            
        # 1. 优先寻找现成的实体 (Solid)
        solids_to_process = []
        explorer = TopExp_Explorer(main_shape, TopAbs_SOLID)
        while explorer.More():
            solids_to_process.append(explorer.Current())
            explorer.Next()
            
        # 2. 如果没找到实体，启动“曲面缝合”逻辑
        if not solids_to_process:
            log("  -> 未找到实体，尝试从曲面缝合...", "SHORT")
            sewed_solids = attempt_sewing_to_solid(main_shape)
            if sewed_solids:
                log(f"  -> 缝合成功，生成 {len(sewed_solids)} 个对象", "SHORT")
                solids_to_process.extend(sewed_solids)
            else:
                log("  -> 缝合失败，无法生成有效几何", "SHORT")

        # 3. 遍历列表进行导出
        exported_count = 0
        for current_shape in solids_to_process:
            # 指纹去重
            is_duplicate = False
            if USE_DEDUPLICATION:
                fp = get_fingerprint(current_shape)
                # 体积极小（可能是未闭合的片体）时，我们看表面积
                # 如果体积和表面积都接近0，可能是坏数据
                if fp[1] > 0.0001: 
                    if fp in seen_geometries:
                        is_duplicate = True
                        log(f"  [跳过重复] Vol:{fp[0]} Area:{fp[1]}")
                    else:
                        seen_geometries.add(fp)
            
            if not is_duplicate:
                out_name = f"{global_counter}.step"
                out_path = os.path.join(OUTPUT_FOLDER, out_name)
                
                if write_step_ap214(current_shape, out_path):
                    log(f"  -> 导出: {out_name}")
                    global_counter += 1
                    exported_count += 1
                else:
                    log("  -> 写入失败", "SHORT")

        log(f"  -> 本文件共产出 {exported_count} 个零件", "SHORT")

    print(f"\n全部完成！共生成 {global_counter} 个文件。", "SHORT")

if __name__ == "__main__":
    main()