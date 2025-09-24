#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
光谱分析工具函数模块
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import interp1d


def find_spectral_peaks(spectrum, prominence=0.1, width=None, distance=5, height=None):
    """
    寻找光谱中的峰值
    
    参数:
        spectrum: 输入光谱数据，一维数组
        prominence: 峰的突出度
        width: 峰的宽度约束
        distance: 相邻峰之间的最小距离
        height: 峰的高度约束
        
    返回:
        peak_indices: 峰值对应的索引
        peak_properties: 峰的属性（高度、宽度等）
    """
    peak_indices, peak_properties = find_peaks(
        spectrum, 
        prominence=prominence,
        width=width,
        distance=distance,
        height=height
    )
    
    return peak_indices, peak_properties


def interpolate_spectrum(wavelengths, intensities, new_wavelengths, method='linear'):
    """
    光谱插值
    
    参数:
        wavelengths: 原始波长
        intensities: 原始光谱强度
        new_wavelengths: 新的波长点
        method: 插值方法，可选 'linear', 'cubic', 'nearest'
        
    返回:
        new_intensities: 插值后的光谱强度
    """
    f = interp1d(wavelengths, intensities, kind=method, bounds_error=False, fill_value="extrapolate")
    new_intensities = f(new_wavelengths)
    
    return new_intensities


def wavelength_to_wavenumber(wavelength_nm):
    """
    波长转波数 (nm -> cm^-1)
    
    参数:
        wavelength_nm: 波长 (纳米)
        
    返回:
        wavenumber: 波数 (cm^-1)
    """
    # 波长（nm）转波数（cm^-1）
    return 1e7 / wavelength_nm


def wavenumber_to_wavelength(wavenumber_cm1):
    """
    波数转波长 (cm^-1 -> nm)
    
    参数:
        wavenumber_cm1: 波数 (cm^-1)
        
    返回:
        wavelength: 波长 (nm)
    """
    # 波数（cm^-1）转波长（nm）
    return 1e7 / wavenumber_cm1


def rmse(y_true, y_pred):
    """
    计算均方根误差 (Root Mean Square Error)
    
    参数:
        y_true: 真实值
        y_pred: 预测值
        
    返回:
        RMSE值
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def plot_spectrum(wavelengths, intensities, title="光谱图", xlabel="波长 (nm)", ylabel="吸光度", 
                  peak_indices=None, legend=None, ax=None, **kwargs):
    """
    绘制光谱图
    
    参数:
        wavelengths: 波长数据
        intensities: 光谱强度数据 (可以是单个光谱或者多个光谱)
        title: 图标题
        xlabel: x轴标签
        ylabel: y轴标签
        peak_indices: 峰值索引，用于在图上标记峰值
        legend: 图例标签
        ax: matplotlib轴对象，如果提供则在该轴上绘图
        **kwargs: 传递给plot函数的其他参数
    
    返回:
        ax: 绘图使用的轴对象
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # 检查intensities是否为二维数组（多个光谱）
    if intensities.ndim > 1:
        for i, intensity in enumerate(intensities):
            label = legend[i] if legend is not None and i < len(legend) else f"光谱 {i+1}"
            ax.plot(wavelengths, intensity, label=label, **kwargs)
        
        if legend is not None:
            ax.legend()
    else:
        # 单个光谱
        ax.plot(wavelengths, intensities, **kwargs)
        
        # 如果提供了峰值索引，标记峰值
        if peak_indices is not None:
            ax.plot(wavelengths[peak_indices], intensities[peak_indices], 'ro', label='峰值')
            ax.legend()
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    return ax


def plot_spectra_comparison(wavelengths, original, processed, title="光谱对比图", 
                            xlabel="波长 (nm)", ylabel="吸光度", labels=None, ax=None):
    """
    绘制原始光谱和处理后光谱的对比图
    
    参数:
        wavelengths: 波长数据
        original: 原始光谱数据
        processed: 处理后的光谱数据
        title: 图标题
        xlabel: x轴标签
        ylabel: y轴标签
        labels: 图例标签 [原始标签, 处理后标签]
        ax: matplotlib轴对象
        
    返回:
        ax: 绘图使用的轴对象
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    if labels is None:
        labels = ["原始光谱", "处理后光谱"]
    
    ax.plot(wavelengths, original, 'b-', label=labels[0])
    ax.plot(wavelengths, processed, 'r-', label=labels[1])
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    return ax


def generate_synthetic_spectrum(wavelengths, peaks=None, noise_level=0.02, baseline_type='linear'):
    """
    生成合成光谱数据（用于测试）
    
    参数:
        wavelengths: 波长数组
        peaks: 峰值列表，每个元素为 (位置, 高度, 宽度)
        noise_level: 噪声水平
        baseline_type: 基线类型 ('linear', 'quadratic', 'exponential')
        
    返回:
        spectrum: 合成的光谱数据
    """
    if peaks is None:
        # 默认峰值
        peaks = [
            (wavelengths[len(wavelengths) // 4], 1.0, len(wavelengths) // 20),
            (wavelengths[len(wavelengths) // 2], 0.8, len(wavelengths) // 25),
            (wavelengths[3 * len(wavelengths) // 4], 1.2, len(wavelengths) // 15)
        ]
    
    # 初始化光谱
    spectrum = np.zeros_like(wavelengths, dtype=float)
    
    # 添加高斯峰
    for position, height, width in peaks:
        # 计算与位置的距离
        idx = np.argmin(np.abs(wavelengths - position))
        for i in range(len(wavelengths)):
            # 高斯函数
            spectrum[i] += height * np.exp(-((wavelengths[i] - wavelengths[idx]) ** 2) / (2 * width ** 2))
    
    # 添加基线
    x_norm = (wavelengths - wavelengths.min()) / (wavelengths.max() - wavelengths.min())
    
    if baseline_type == 'linear':
        baseline = 0.2 * x_norm
    elif baseline_type == 'quadratic':
        baseline = 0.2 * x_norm**2
    elif baseline_type == 'exponential':
        baseline = 0.1 * np.exp(x_norm) - 0.1
    else:
        baseline = np.zeros_like(wavelengths)
    
    spectrum += baseline
    
    # 添加随机噪声
    noise = np.random.normal(0, noise_level, size=wavelengths.shape)
    spectrum += noise
    
    return spectrum 