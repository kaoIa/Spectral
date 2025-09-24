#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
光谱预处理算法模块
包含常用的光谱预处理方法，如标准化、平滑、导数等
"""

import numpy as np
from scipy import signal
from scipy.signal import savgol_filter


def normalize(spectrum, method='minmax'):
    """
    光谱数据标准化
    
    参数:
        spectrum: 输入光谱数据, shape=(n_samples, n_features)
        method: 标准化方法, 可选 'minmax', 'mean', 'unit'
        
    返回:
        标准化后的光谱数据
    """
    if method == 'minmax':
        # 最小-最大规范化
        min_vals = np.min(spectrum, axis=1, keepdims=True)
        max_vals = np.max(spectrum, axis=1, keepdims=True)
        return (spectrum - min_vals) / (max_vals - min_vals)
    
    elif method == 'mean':
        # 均值标准化
        mean = np.mean(spectrum, axis=1, keepdims=True)
        std = np.std(spectrum, axis=1, keepdims=True)
        return (spectrum - mean) / std
    
    elif method == 'unit':
        # 单位向量标准化
        norm = np.linalg.norm(spectrum, axis=1, keepdims=True)
        return spectrum / norm
    
    else:
        raise ValueError(f"不支持的标准化方法: {method}")


def msc(spectrum, reference=None):
    """
    多元散射校正 (Multiplicative Scatter Correction)
    
    参数:
        spectrum: 输入光谱数据, shape=(n_samples, n_features)
        reference: 参考光谱, 默认使用平均光谱
        
    返回:
        校正后的光谱数据
    """
    if reference is None:
        reference = np.mean(spectrum, axis=0)
    
    # 对每个样本进行线性回归与scatter校正
    n_samples = spectrum.shape[0]
    corrected = np.zeros_like(spectrum)
    
    for i in range(n_samples):
        # 计算回归系数 (截距和斜率)
        coef = np.polyfit(reference, spectrum[i, :], 1)
        # 校正
        corrected[i, :] = (spectrum[i, :] - coef[1]) / coef[0]
    
    return corrected


def snv(spectrum):
    """
    标准正态变量转换 (Standard Normal Variate)
    
    参数:
        spectrum: 输入光谱数据, shape=(n_samples, n_features)
        
    返回:
        校正后的光谱数据
    """
    # 对每个样本进行均值中心化和标准差归一化
    mean = np.mean(spectrum, axis=1, keepdims=True)
    std = np.std(spectrum, axis=1, keepdims=True)
    return (spectrum - mean) / std


def savitzky_golay(spectrum, window_length=11, polyorder=2, deriv=0):
    """
    Savitzky-Golay平滑滤波
    
    参数:
        spectrum: 输入光谱数据, shape=(n_samples, n_features)
        window_length: 窗口长度，必须为奇数
        polyorder: 多项式阶数
        deriv: 求导阶数
        
    返回:
        平滑后的光谱数据
    """
    # 确保窗口长度为奇数
    if window_length % 2 == 0:
        window_length += 1
    
    # 应用Savitzky-Golay滤波器
    return savgol_filter(spectrum, window_length, polyorder, deriv=deriv, axis=1)


def detrend(spectrum, degree=1):
    """
    去趋势 (Detrending)
    
    参数:
        spectrum: 输入光谱数据, shape=(n_samples, n_features)
        degree: 多项式阶数
        
    返回:
        去趋势后的光谱数据
    """
    n_samples, n_features = spectrum.shape
    x = np.arange(n_features)
    corrected = np.zeros_like(spectrum)
    
    for i in range(n_samples):
        # 拟合多项式
        coeffs = np.polyfit(x, spectrum[i, :], degree)
        # 计算趋势
        trend = np.polyval(coeffs, x)
        # 去除趋势
        corrected[i, :] = spectrum[i, :] - trend
    
    return corrected


def baseline_correction(spectrum, asymmetry_param=0.05, smoothness_param=1000000):
    """
    基线校正，使用非对称最小二乘法
    
    参数:
        spectrum: 输入光谱数据, shape=(n_samples, n_features)
        asymmetry_param: 非对称参数
        smoothness_param: 平滑参数
        
    返回:
        基线校正后的光谱数据
    """
    n_samples, n_features = spectrum.shape
    corrected = np.zeros_like(spectrum)
    
    for i in range(n_samples):
        y = spectrum[i, :]
        # 非对称最小二乘法基线校正
        baseline = als_baseline(y, asymmetry_param, smoothness_param)
        corrected[i, :] = y - baseline
    
    return corrected


def als_baseline(y, lam=1e5, p=0.05, niter=10):
    """
    渐进线性最小二乘算法基线校正
    
    参数:
        y: 光谱数据
        lam: 平滑参数
        p: 非对称参数
        niter: 迭代次数
        
    返回:
        估计的基线
    """
    L = len(y)
    # 构建二阶差分矩阵
    D = np.diff(np.eye(L), 2)
    w = np.ones(L)
    
    for i in range(niter):
        # 计算此次迭代的基线
        W = np.diag(w)
        Z = W + lam * np.dot(D.T, D)
        z = np.linalg.solve(Z, w * y)
        
        # 更新权重
        w = p * (y > z) + (1-p) * (y <= z)
    
    return z


def moving_average(spectrum, window_size=5):
    """
    移动平均滤波
    
    参数:
        spectrum: 输入光谱数据, shape=(n_samples, n_features)
        window_size: 窗口大小，必须为奇数
        
    返回:
        滤波后的光谱数据
    """
    # 确保窗口长度为奇数
    if window_size % 2 == 0:
        window_size += 1
    
    # 创建输出数组
    n_samples, n_features = spectrum.shape
    filtered = np.zeros_like(spectrum)
    
    # 定义卷积窗口
    window = np.ones(window_size) / window_size
    
    # 对每个样本应用移动平均
    for i in range(n_samples):
        # 使用卷积实现移动平均
        filtered[i, :] = np.convolve(spectrum[i, :], window, mode='same')
        
        # 处理边缘效应
        half_win = window_size // 2
        # 左边
        for j in range(half_win):
            filtered[i, j] = np.mean(spectrum[i, :j+half_win+1])
        # 右边
        for j in range(n_features - half_win, n_features):
            filtered[i, j] = np.mean(spectrum[i, j-half_win:])
    
    return filtered


def wavelet_denoise(spectrum, wavelet='db8', level=3, threshold='soft'):
    """
    小波变换降噪
    
    参数:
        spectrum: 输入光谱数据, shape=(n_samples, n_features)
        wavelet: 小波基函数
        level: 分解层数
        threshold: 阈值策略 'soft' 或 'hard'
        
    返回:
        降噪后的光谱数据
    """
    try:
        import pywt
    except ImportError:
        raise ImportError("请安装PyWavelets库: pip install PyWavelets")
    
    n_samples, n_features = spectrum.shape
    denoised = np.zeros_like(spectrum)
    
    for i in range(n_samples):
        # 小波分解
        coeffs = pywt.wavedec(spectrum[i, :], wavelet, level=level)
        
        # 应用阈值
        for j in range(1, len(coeffs)):
            # 计算VisuShrink阈值
            sigma = np.median(np.abs(coeffs[j])) / 0.6745
            thresh = sigma * np.sqrt(2 * np.log(n_features))
            
            # 应用阈值
            if threshold == 'soft':
                coeffs[j] = pywt.threshold(coeffs[j], thresh, mode='soft')
            else:
                coeffs[j] = pywt.threshold(coeffs[j], thresh, mode='hard')
        
        # 小波重构
        denoised[i, :] = pywt.waverec(coeffs, wavelet)
    
    return denoised 