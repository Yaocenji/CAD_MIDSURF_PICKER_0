import os
import argparse
from OCC.Core.STEPControl import STEPControl_Reader, STEPControl_Writer, STEPControl_AsIs
from OCC.Core.IGESControl import IGESControl_Reader
from OCC.Core.Interface import Interface_Static_SetCVal
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_SOLID
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop_VolumeProperties, brepgprop_SurfaceProperties

# ================= 配置区域 =================
INPUT_FOLDER = r"D:\MidSurf\Dataset_new\TracePart_Dataset\物料搬运\raw1"
OUTPUT_FOLDER = r"D:\MidSurf\Dataset_new\TracePart_Dataset\物料搬运\processed1"

# 是否开启去重？
# True: 几何完全相同（体积+表面积）的零件只导出一份
# False: 所有零件都导出（即使是重复的螺丝）
USE_DEDUPLICATION = True 
# ===========================================

def load_file(filepath):
    """读取文件返回 Shape"""
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
    """
    [本地去重核心] 计算实体的体积和表面积
    返回: (Volume, Area) 元组
    """
    # 1. 计算体积
    v_props = GProp_GProps()
    brepgprop_VolumeProperties(shape, v_props)
    vol = v_props.Mass()

    # 2. 计算表面积
    s_props = GProp_GProps()
    brepgprop_SurfaceProperties(shape, s_props)
    area = s_props.Mass()

    # 保留5位小数，避免浮点数微小误差
    return (round(vol, 5), round(area, 5))

def write_step_ap214(shape, output_path):
    """写入单个实体为 STEP AP214"""
    writer = STEPControl_Writer()
    Interface_Static_SetCVal("write.step.schema", "AP214") 
    status = writer.Transfer(shape, STEPControl_AsIs)
    if status == IFSelect_RetDone:
        writer.Write(output_path)
        return True
    return False

def main():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        
    all_files = os.listdir(INPUT_FOLDER)
    supported_exts = ('.stp', '.step', '.igs', '.iges')
    cad_files = [f for f in all_files if f.lower().endswith(supported_exts)]
    
    print(f"找到 {len(cad_files)} 个源文件，准备拆分处理...")
    
    # === [新增] 参数解析逻辑 ===
    parser = argparse.ArgumentParser(description="CAD 批量拆分工具")
    # 添加 --start 参数，默认值为 0，类型为整数
    parser.add_argument('--start', type=int, default=0, help="起始编号 (例如: 87)")
    # 解析参数 (如果有未知参数也不报错，方便在某些IDE中运行)
    args, unknown = parser.parse_known_args()
    
    global_counter = args.start
    print(f"当前起始编号设置: {global_counter}")
    # ==========================

    # 全局指纹库 (如果希望跨文件去重，放在这里；如果只希望单文件内去重，移入循环)
    seen_geometries = set() 
    
    for filename in cad_files:
        full_path = os.path.join(INPUT_FOLDER, filename)
        print(f"\n正在读取: {filename}")
        
        main_shape = load_file(full_path)
        if not main_shape:
            print("  -> 读取失败或文件为空")
            continue
            
        # === 核心逻辑：遍历拓扑结构 ===
        # TopExp_Explorer 会深入挖掘，找到所有的 SOLID (实体)
        # 不管它们是打散的还是组合在一个 Compound 里的
        explorer = TopExp_Explorer(main_shape, TopAbs_SOLID)
        
        parts_in_file = 0
        
        while explorer.More():
            current_solid = explorer.Current()
            
            # 1. 指纹检查 (去重)
            is_duplicate = False
            if USE_DEDUPLICATION:
                fingerprint = get_fingerprint(current_solid)
                # 只有体积大于微小值才计算(避免噪点)
                if fingerprint[0] > 0.0001:
                    if fingerprint in seen_geometries:
                        is_duplicate = True
                        print(f"  [跳过重复] Vol: {fingerprint[0]}")
                    else:
                        seen_geometries.add(fingerprint)
            
            # 2. 导出
            if not is_duplicate:
                out_name = f"{global_counter}.step"
                out_path = os.path.join(OUTPUT_FOLDER, out_name)
                
                if write_step_ap214(current_solid, out_path):
                    print(f"  -> 导出零件 {global_counter}: {out_name}")
                    global_counter += 1
                    parts_in_file += 1
                else:
                    print(f"  -> 写入失败")
            
            # 移动到下一个实体
            explorer.Next()
            
        if parts_in_file == 0:
            print("  -> 未在该文件中找到实体 (可能仅包含曲面或线框)")
        else:
            print(f"  -> 从该文件拆分出 {parts_in_file} 个零件")

    print("\n=========================")
    print(f"处理完成！共生成 {global_counter} 个独立零件文件。")

if __name__ == "__main__":
    main()