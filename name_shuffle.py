import os
import random
import re
import uuid
import sys

def batch_rename_shuffle(target_dir, range_start, range_end, offset):
    """
    Args:
        target_dir (str): 目标文件夹路径
        range_start (int): 闭区间开始
        range_end (int): 闭区间结束
        offset (int): 新命名的起始偏移量
    """
    
    # 1. 检查路径是否存在
    if not os.path.exists(target_dir):
        print(f"错误：文件夹 '{target_dir}' 不存在。")
        return

    # 获取文件夹内所有文件
    all_files = os.listdir(target_dir)
    
    # 2. 筛选符合条件的文件
    # 结构: {'original_full_name': '1.txt', 'num': 1, 'ext': '.txt', 'path': ...}
    target_files = []
    
    for filename in all_files:
        # 分离文件名和后缀
        name_part, ext_part = os.path.splitext(filename)
        
        # 尝试将文件名转为整数
        if name_part.isdigit():
            num = int(name_part)
            # 检查是否在闭区间 [range_start, range_end]
            if range_start <= num <= range_end:
                target_files.append({
                    'original_name': filename,
                    'num': num,
                    'ext': ext_part,
                    'old_path': os.path.join(target_dir, filename)
                })

    if not target_files:
        print(f"在区间 [{range_start}, {range_end}] 内没有找到符合条件的文件。")
        return

    print(f"共找到 {len(target_files)} 个文件符合条件，准备处理...")

    # 3. 随机打乱
    # 为了满足“尽量让原本相邻的文件不再相邻”，标准的 random.shuffle 对于小样本可能不够完美，
    # 但对于一般用途，random.shuffle 配合多次洗牌通常足够。
    # 这里我们简单使用 shuffle。如果需要极端的“非相邻”算法，逻辑会复杂很多，
    # 但通常随机性本身就会打破邻接关系。
    random.shuffle(target_files)

    # 4. 执行重命名
    # 策略：为了防止冲突（例如把 5.txt 改成 4.txt，但 4.txt 也在列表中），
    # 我们采用两步法：
    # 第一步：原名 -> 临时随机名 (UUID)
    # 第二步：临时随机名 -> 目标名 (Offset)

    temp_info_list = []
    mapping_record = [] # 用于保存映射记录
    
    print("步骤 1/2: 重命名为临时文件...")
    for item in target_files:
        # 生成唯一的临时文件名
        temp_name = f"temp_{uuid.uuid4().hex}{item['ext']}"
        temp_path = os.path.join(target_dir, temp_name)
        
        # 重命名
        os.rename(item['old_path'], temp_path)
        
        # 记录中间状态
        temp_info_list.append({
            'original_name': item['original_name'],
            'temp_path': temp_path,
            'ext': item['ext']
        })

    print("步骤 2/2: 重命名为目标顺序文件...")
    current_num = offset
    
    for item in temp_info_list:
        new_name = f"{current_num}{item['ext']}"
        new_path = os.path.join(target_dir, new_name)
        
        # 检查最终目标是否存在（防止覆盖了不在区间内的其他已有文件）
        if os.path.exists(new_path):
            print(f"警告: 目标文件 {new_name} 已存在（可能不在处理区间内），跳过该文件的最终重命名。")
            # 这里可以选择报错停止，或者保留临时文件名，视需求而定。
            # 为安全起见，这里我们不仅打印，还把原映射记下来方便手动恢复
            mapping_record.append(f"{item['original_name']} -> [冲突] 保持为临时文件: {os.path.basename(item['temp_path'])}")
            continue
        
        os.rename(item['temp_path'], new_path)
        
        # 记录映射
        mapping_record.append(f"{item['original_name']} -> {new_name}")
        current_num += 1

    # 5. 保存映射文件
    mapping_file = os.path.join(target_dir, "mapping_log.txt")
    with open(mapping_file, 'w', encoding='utf-8') as f:
        f.write(f"处理时间:区间[{range_start}, {range_end}], Offset: {offset}\n")
        f.write("-" * 40 + "\n")
        for line in mapping_record:
            f.write(line + "\n")
            
    print(f"处理完成！\n映射日志已保存至: {mapping_file}")

# --- 配置区域 ---
if __name__ == "__main__":
    # 请在这里修改你的配置
    
    # 1. 文件夹路径 (请修改为你实际的路径)
    # Windows 示例: r"C:\Users\Name\Desktop\files"
    # Mac/Linux 示例: "/Users/name/files"
    DIRECTORY = r"C:\Users\27800\Desktop\Filtered_data_lean_version\train_shuffled" 
    
    # 2. 筛选区间 [min, max]
    RANGE_START = 29
    RANGE_END = 48999
    
    # 3. 起始 Offset
    OFFSET = 29

    # 为了安全，建议先在一个测试文件夹运行
    try:
        batch_rename_shuffle(DIRECTORY, RANGE_START, RANGE_END, OFFSET)
    except Exception as e:
        print(f"发生意外错误: {e}")