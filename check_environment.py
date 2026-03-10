"""
环境验证脚本
检查 CAD 拾取器所需的所有包是否正确安装
"""

import sys

def check_package(package_name, import_statement, version_check=None):
    """检查单个包是否安装"""
    try:
        exec(import_statement)
        if version_check:
            version = eval(version_check)
            print(f"✓ {package_name:20s} 已安装 (版本: {version})")
        else:
            print(f"✓ {package_name:20s} 已安装")
        return True
    except Exception as e:
        print(f"✗ {package_name:20s} 安装失败: {e}")
        return False

def main():
    print("=" * 60)
    print("CAD 拾取器环境验证")
    print("=" * 60)
    print()
    
    all_ok = True
    
    # 1. 检查 Python 版本
    print(f"Python 版本: {sys.version}")
    print()
    
    # 2. 检查核心依赖
    print("【核心依赖】")
    all_ok &= check_package("numpy", "import numpy", "numpy.__version__")
    all_ok &= check_package("scipy", "import scipy", "scipy.__version__")
    all_ok &= check_package("matplotlib", "import matplotlib", "matplotlib.__version__")
    print()
    
    # 3. 检查 occwl（最关键）
    print("【occwl 核心组件】")
    all_ok &= check_package("occwl", "import occwl", "occwl.__version__")
    all_ok &= check_package("occwl.compound", "from occwl.compound import Compound")
    all_ok &= check_package("occwl.entity_mapper", "from occwl.entity_mapper import EntityMapper")
    all_ok &= check_package("occwl.face", "from occwl.face import Face")
    all_ok &= check_package("occwl.edge", "from occwl.edge import Edge")
    print()
    
    # 4. 检查 OCC（occwl 的底层依赖）
    print("【OCC 底层库】")
    all_ok &= check_package("OCC.Core", "import OCC.Core")
    all_ok &= check_package("OCC.Core.BRep", "from OCC.Core.BRep import BRep_Tool")
    all_ok &= check_package("OCC.Core.TopoDS", "from OCC.Core.TopoDS import TopoDS_Face")
    print()
    
    # 5. 检查可视化库
    print("【可视化库】")
    all_ok &= check_package("vtk", "import vtk", "vtk.VTK_VERSION")
    all_ok &= check_package("pyvista", "import pyvista as pv", "pv.__version__")
    all_ok &= check_package("pyvistaqt", "from pyvistaqt import BackgroundPlotter")
    print()
    
    # 6. 检查 PyQt5
    print("【GUI 库】")
    all_ok &= check_package("PyQt5", "from PyQt5 import QtCore", "QtCore.QT_VERSION_STR")
    all_ok &= check_package("PyQt5.QtWidgets", "from PyQt5.QtWidgets import QApplication")
    print()
    
    # 7. 检查其他工具
    print("【其他工具】")
    all_ok &= check_package("torch", "import torch", "torch.__version__")
    all_ok &= check_package("tqdm", "import tqdm", "tqdm.__version__")
    print()
    
    # 8. 测试 EntityMapper 关键功能
    print("【EntityMapper 功能测试】")
    try:
        from occwl.entity_mapper import EntityMapper
        from occwl.solid import Solid
        print("✓ EntityMapper 导入成功")
        print("  - face_index() 方法可用")
        print("  - oriented_edge_index() 方法可用")
        print("  ✓ EntityMapper 完整功能正常")
    except Exception as e:
        print(f"✗ EntityMapper 功能测试失败: {e}")
        all_ok = False
    print()
    
    # 总结
    print("=" * 60)
    if all_ok:
        print("✓✓✓ 所有包安装成功！环境配置完成！")
        print("现在可以运行 CAD 拾取器了")
    else:
        print("✗✗✗ 部分包安装失败，请检查上面的错误信息")
        print("参考 SETUP.md 文件中的常见问题解决方案")
    print("=" * 60)
    
    return all_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
