#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
光谱预处理模块测试
"""

import sys
import os
import unittest
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

from src.core.preprocessing import (normalize, msc, snv, savitzky_golay, 
                                  detrend, baseline_correction)


class TestPreprocessingMethods(unittest.TestCase):
    """测试光谱预处理方法"""
    
    def setUp(self):
        """测试前准备工作"""
        # 创建一些合成测试数据
        self.wavelengths = np.linspace(900, 1700, 200)
        
        # 创建带有基线和噪声的合成光谱
        n_samples = 5
        n_wavelengths = len(self.wavelengths)
        self.spectra = np.zeros((n_samples, n_wavelengths))
        
        # 添加基本峰值
        peak_positions = [1000, 1200, 1400, 1600]
        peak_heights = [0.2, 0.5, 0.3, 0.4]
        peak_widths = [30, 40, 20, 25]
        
        for i in range(n_samples):
            # 添加基线 (线性或二次)
            baseline = 0.1 * (i + 1) + 0.0002 * (i + 1) * (self.wavelengths - 900)**2
            
            # 添加峰值
            for pos, height, width in zip(peak_positions, peak_heights, peak_widths):
                # 不同样本峰值位置略有差异
                pos_shift = pos + np.random.normal(0, 5)
                # 不同样本峰值高度略有差异
                height_shift = height * (0.8 + 0.4 * np.random.random())
                
                # 添加高斯峰
                self.spectra[i, :] += height_shift * np.exp(-(self.wavelengths - pos_shift)**2 / (2 * width**2))
            
            # 添加基线和噪声
            self.spectra[i, :] += baseline + 0.02 * np.random.randn(n_wavelengths)
    
    def test_normalize(self):
        """测试标准化方法"""
        # 测试min-max标准化
        norm_minmax = normalize(self.spectra, method='minmax')
        self.assertEqual(norm_minmax.shape, self.spectra.shape)
        
        # 每个样本的最小值应该接近0，最大值应该接近1
        for i in range(norm_minmax.shape[0]):
            self.assertAlmostEqual(np.min(norm_minmax[i, :]), 0, delta=1e-10)
            self.assertAlmostEqual(np.max(norm_minmax[i, :]), 1, delta=1e-10)
        
        # 测试均值标准化
        norm_mean = normalize(self.spectra, method='mean')
        self.assertEqual(norm_mean.shape, self.spectra.shape)
        
        # 每个样本的均值应该接近0，标准差应该接近1
        for i in range(norm_mean.shape[0]):
            self.assertAlmostEqual(np.mean(norm_mean[i, :]), 0, delta=1e-10)
            self.assertAlmostEqual(np.std(norm_mean[i, :]), 1, delta=1e-10)
    
    def test_msc(self):
        """测试多元散射校正"""
        msc_spectra = msc(self.spectra)
        self.assertEqual(msc_spectra.shape, self.spectra.shape)
        
        # 检查MSC是否减少了样本间的变异性
        orig_var = np.var(self.spectra, axis=0).mean()
        msc_var = np.var(msc_spectra, axis=0).mean()
        
        # MSC后的变异性应该减小
        self.assertLess(msc_var, orig_var)
    
    def test_snv(self):
        """测试标准正态变量转换"""
        snv_spectra = snv(self.spectra)
        self.assertEqual(snv_spectra.shape, self.spectra.shape)
        
        # 每个样本的均值应该接近0，标准差应该接近1
        for i in range(snv_spectra.shape[0]):
            self.assertAlmostEqual(np.mean(snv_spectra[i, :]), 0, delta=1e-10)
            self.assertAlmostEqual(np.std(snv_spectra[i, :]), 1, delta=1e-10)
    
    def test_savitzky_golay(self):
        """测试Savitzky-Golay平滑"""
        smoothed = savitzky_golay(self.spectra, window_length=11, polyorder=2)
        self.assertEqual(smoothed.shape, self.spectra.shape)
        
        # 平滑后的曲线应该比原始数据更平滑（方差更小）
        for i in range(self.spectra.shape[0]):
            # 计算差分的均方差作为粗糙度度量
            orig_roughness = np.mean(np.diff(self.spectra[i, :])**2)
            smoothed_roughness = np.mean(np.diff(smoothed[i, :])**2)
            
            # 平滑后的粗糙度应该更小
            self.assertLess(smoothed_roughness, orig_roughness)
        
        # 测试求导
        sg_deriv = savitzky_golay(self.spectra, window_length=11, polyorder=2, deriv=1)
        self.assertEqual(sg_deriv.shape, self.spectra.shape)
    
    def test_detrend(self):
        """测试去趋势"""
        detrended = detrend(self.spectra, degree=1)
        self.assertEqual(detrended.shape, self.spectra.shape)
        
        # 去趋势后，线性拟合的斜率应该接近0
        for i in range(detrended.shape[0]):
            x = np.arange(detrended.shape[1])
            slope, _ = np.polyfit(x, detrended[i, :], 1)
            self.assertAlmostEqual(slope, 0, delta=1e-10)
    
    def test_baseline_correction(self):
        """测试基线校正"""
        corrected = baseline_correction(self.spectra)
        self.assertEqual(corrected.shape, self.spectra.shape)
        
        # 基线校正后，光谱的最小值应该接近0
        for i in range(corrected.shape[0]):
            self.assertLess(np.min(corrected[i, :]), 0.1)


if __name__ == "__main__":
    unittest.main() 