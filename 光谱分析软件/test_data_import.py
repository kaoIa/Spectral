#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试新的数据导入功能
"""

import sys
import os
from pathlib import Path

# 添加项目路径到系统路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.manager import SpectrumDataset

def test_corn_data_import():
    """测试玉米数据导入功能"""
    print("=== 测试玉米数据导入功能 ===")

    # 测试包含成分数据的文件
    corn_file = r"D:\a-gsh-study-app-space\Near-Infrared Spectroscopy\data\csv数据\玉米数据集\m5数据.csv"

    if not os.path.exists(corn_file):
        print(f"文件不存在: {corn_file}")
        return False

    try:
        # 创建数据集对象
        dataset = SpectrumDataset("玉米数据测试")

        # 加载数据
        print(f"尝试加载文件: {corn_file}")
        success = dataset.load_file(corn_file, 'csv')

        if success:
            print(f"[OK] 成功加载玉米数据文件: {os.path.basename(corn_file)}")
            print(f"  - 样本数量: {dataset.get_sample_count()}")
            print(f"  - 波长范围: {dataset.get_wavelength_range()}")
            print(f"  - 数据形状: {dataset.get_data_shape()}")

            # 检查目标变量
            target = dataset.get_target()
            target_name = dataset.get_target_name()
            if target is not None:
                print(f"  - 目标变量: {target_name}")
                print(f"  - 目标变量形状: {target.shape}")
                print(f"  - 目标变量范围: {target.min():.3f} - {target.max():.3f}")

            # 检查额外目标变量
            additional_targets = dataset.get_additional_targets()
            if additional_targets:
                print(f"  - 额外目标变量: {list(additional_targets.keys())}")
                for name, values in additional_targets.items():
                    print(f"    - {name}: {len(values)} 个值")

            return True
        else:
            print("[ERROR] 加载玉米数据文件失败")
            return False

    except Exception as e:
        print(f"[ERROR] 测试玉米数据导入时出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_spectrum_data_import():
    """测试纯光谱数据导入功能"""
    print("\n=== 测试纯光谱数据导入功能 ===")

    # 测试纯光谱数据文件
    spectrum_file = r"D:\a-gsh-study-app-space\Near-Infrared Spectroscopy\data\csv数据\玉米数据集\m5_spectra.csv"

    if not os.path.exists(spectrum_file):
        print(f"文件不存在: {spectrum_file}")
        return False

    try:
        # 创建数据集对象
        dataset = SpectrumDataset("光谱数据测试")

        # 加载数据
        success = dataset.load_file(spectrum_file, 'csv')

        if success:
            print(f"[OK] 成功加载光谱数据文件: {os.path.basename(spectrum_file)}")
            print(f"  - 样本数量: {dataset.get_sample_count()}")
            print(f"  - 波长范围: {dataset.get_wavelength_range()}")
            print(f"  - 数据形状: {dataset.get_data_shape()}")

            # 检查波长数据
            wavelengths = dataset.get_wavelengths()
            if wavelengths:
                print(f"  - 波长数量: {len(wavelengths)}")
                print(f"  - 前5个波长: {wavelengths[:5]}")

            # 检查是否有目标变量
            target = dataset.get_target()
            if target is None:
                print("  - 无目标变量（符合预期）")

            return True
        else:
            print("[ERROR] 加载光谱数据文件失败")
            return False

    except Exception as e:
        print(f"[ERROR] 测试光谱数据导入时出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("开始测试新的数据导入功能...\n")

    # 测试结果
    results = []

    # 测试玉米数据导入
    results.append(("玉米数据导入", test_corn_data_import()))

    # 测试纯光谱数据导入
    results.append(("纯光谱数据导入", test_spectrum_data_import()))

    # 显示测试结果
    print("\n" + "="*50)
    print("测试结果汇总:")
    print("="*50)

    all_passed = True
    for test_name, passed in results:
        status = "[OK] 通过" if passed else "[ERROR] 失败"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False

    print("="*50)
    if all_passed:
        print("所有测试通过！新的数据导入功能工作正常。")
    else:
        print("部分测试失败，请检查错误信息。")

    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)