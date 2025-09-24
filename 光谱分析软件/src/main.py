#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
光谱分析软件主程序入口
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# 设置Qt插件路径
def setup_qt_plugin_path():
    """设置Qt插件路径以解决平台插件问题"""
    try:
        import site
        import PyQt5
        
        # 尝试找到PyQt5安装路径
        pyqt_path = os.path.dirname(PyQt5.__file__)
        print(f"PyQt5路径: {pyqt_path}")
        
        # 设置Qt插件路径
        os.environ['QT_PLUGIN_PATH'] = os.path.join(pyqt_path, "Qt5", "plugins")
        print(f"设置QT_PLUGIN_PATH: {os.environ['QT_PLUGIN_PATH']}")
        
        # 如果路径不存在，尝试其他可能的位置
        if not os.path.exists(os.environ['QT_PLUGIN_PATH']):
            # 尝试site-packages中的其他可能位置
            for p in site.getsitepackages():
                possible_path = os.path.join(p, "PyQt5", "Qt", "plugins")
                if os.path.exists(possible_path):
                    os.environ['QT_PLUGIN_PATH'] = possible_path
                    print(f"重设QT_PLUGIN_PATH: {os.environ['QT_PLUGIN_PATH']}")
                    break
                
                possible_path = os.path.join(p, "PyQt5", "Qt5", "plugins")
                if os.path.exists(possible_path):
                    os.environ['QT_PLUGIN_PATH'] = possible_path
                    print(f"重设QT_PLUGIN_PATH: {os.environ['QT_PLUGIN_PATH']}")
                    break
    except Exception as e:
        print(f"设置Qt插件路径出错: {e}")

# 设置Qt插件路径
setup_qt_plugin_path()

# 配置matplotlib支持中文
def setup_matplotlib_chinese():
    """配置matplotlib支持中文显示"""
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm
        
        print("正在配置matplotlib中文支持...")
        
        # 尝试使用系统中文字体
        font_list = [
            'Microsoft YaHei',   # 微软雅黑
            'SimHei',            # 中文黑体
            'SimSun',            # 中文宋体
            'KaiTi',             # 楷体
            'FangSong',          # 仿宋
            'Arial Unicode MS',  # Arial Unicode
            'WenQuanYi Micro Hei' # 文泉驿微米黑
        ]
        
        # 查找可用的中文字体
        font_found = False
        for font_name in font_list:
            font_path = None
            for f in fm.findSystemFonts():
                if font_name.lower() in os.path.basename(f).lower():
                    font_path = f
                    break
            
            if font_path:
                print(f"找到中文字体: {font_name} 路径: {font_path}")
                plt.rcParams['font.family'] = ['sans-serif']
                plt.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans', 'Bitstream Vera Sans', 'sans-serif']
                # 处理负号显示问题
                plt.rcParams['axes.unicode_minus'] = False
                font_found = True
                break
        
        if not font_found:
            print("未找到中文字体，将使用默认字体")
            # 使用通用设置，可能不完全支持中文
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Bitstream Vera Sans', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            
    except Exception as e:
        print(f"配置matplotlib中文支持时出错: {e}")

# 尝试配置matplotlib支持中文
setup_matplotlib_chinese()

# 导入GUI模块
try:
    print("导入PyQt5...")
    import PyQt5
    # 先尝试导入PyQt5.QtWidgets，这样会自动导入QtCore
    from PyQt5.QtWidgets import QApplication
    from src.gui.main_window import MainWindow
    print("PyQt5导入成功")
except Exception as e:
    print(f"导入模块失败: {e}")
    import traceback
    traceback.print_exc()
    print("\n请确保已正确安装PyQt5，可尝试以下命令:")
    print("pip uninstall PyQt5 PyQt5-Qt5 PyQt5-sip")
    print("pip install PyQt5==5.15.2")
    sys.exit(1)

def main():
    """主程序入口"""
    try:
        # 这里可以进行初始化操作
        print("启动光谱分析软件...")
        
        # 创建QApplication实例
        app = QApplication(sys.argv)
        
        # 启动GUI
        window = MainWindow()
        window.show()  # 使用show()方法而不是run()
        
        # 进入应用程序主循环
        return app.exec_()
        
    except Exception as e:
        print(f"程序出错: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())