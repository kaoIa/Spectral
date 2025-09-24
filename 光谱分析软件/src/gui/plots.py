#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
光谱分析软件绘图模块
包含用于在GUI中显示各种光谱分析图表的类
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

from PyQt5.QtWidgets import QSizePolicy


class SpectrumPlotCanvas(FigureCanvas):
    """光谱绘图画布类"""
    
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        """
        初始化光谱绘图画布
        
        参数:
            parent: 父级窗口部件
            width: 画布宽度（英寸）
            height: 画布高度（英寸）
            dpi: 分辨率
        """
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        
        # 初始化画布
        super(SpectrumPlotCanvas, self).__init__(self.fig)
        self.setParent(parent)
        
        # 设置画布大小策略
        FigureCanvas.setSizePolicy(self,
                               QSizePolicy.Expanding,
                               QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        
        # 初始化绘图
        self.clear_plot()
    
    def clear_plot(self):
        """清除绘图"""
        self.axes.clear()
        self.axes.set_title("光谱图")
        self.axes.set_xlabel("波长 (nm)")
        self.axes.set_ylabel("吸光度")
        self.axes.grid(True, linestyle='--', alpha=0.7)
        self.draw()
    
    def plot_spectrum(self, wavelengths, intensities, title="光谱图", 
                     xlabel="波长 (nm)", ylabel="吸光度", 
                     peak_indices=None, sample_name=None):
        """
        绘制单个光谱
        
        参数:
            wavelengths: 波长数组
            intensities: 光谱强度
            title: 图标题
            xlabel: x轴标签
            ylabel: y轴标签
            peak_indices: 峰值索引
            sample_name: 样本名称
        """
        self.axes.clear()
        
        # 设置标题和轴标签
        self.axes.set_title(title)
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        
        # 绘制光谱线
        label = sample_name if sample_name else "光谱"
        self.axes.plot(wavelengths, intensities, 'b-', linewidth=1.5, label=label)
        
        # 如果提供了峰值索引，标记峰值
        if peak_indices is not None:
            self.axes.plot(wavelengths[peak_indices], intensities[peak_indices], 
                         'ro', markersize=4, label='峰值')
        
        self.axes.grid(True, linestyle='--', alpha=0.7)
        self.axes.legend()
        
        # 更新画布
        self.fig.tight_layout()
        self.draw()
    
    def plot_multiple_spectra(self, wavelengths, spectra, sample_names=None, 
                             title="多光谱对比", xlabel="波长 (nm)", 
                             ylabel="吸光度"):
        """
        绘制多个光谱进行对比
        
        参数:
            wavelengths: 波长数组
            spectra: 多个光谱强度数组，shape为(n_samples, n_wavelengths)
            sample_names: 样本名称列表
            title: 图标题
            xlabel: x轴标签
            ylabel: y轴标签
        """
        self.axes.clear()
        
        # 设置标题和轴标签
        self.axes.set_title(title)
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        
        # 绘制每个光谱
        n_samples = spectra.shape[0]
        for i in range(n_samples):
            label = sample_names[i] if sample_names and i < len(sample_names) else f"光谱 {i+1}"
            self.axes.plot(wavelengths, spectra[i], linewidth=1.5, label=label)
        
        self.axes.grid(True, linestyle='--', alpha=0.7)
        self.axes.legend()
        
        # 更新画布
        self.fig.tight_layout()
        self.draw()
    
    def plot_spectra_comparison(self, wavelengths, original, processed, 
                              title="预处理前后对比", xlabel="波长 (nm)", 
                              ylabel="吸光度", labels=None):
        """
        绘制预处理前后的光谱对比图
        
        参数:
            wavelengths: 波长数组
            original: 原始光谱
            processed: 处理后的光谱
            title: 图标题
            xlabel: x轴标签
            ylabel: y轴标签
            labels: 图例标签 [原始标签, 处理后标签]
        """
        self.axes.clear()
        
        # 设置标题和轴标签
        self.axes.set_title(title)
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        
        # 默认标签
        if labels is None:
            labels = ["原始光谱", "处理后光谱"]
        
        # 绘制原始和处理后的光谱
        self.axes.plot(wavelengths, original, 'b-', linewidth=1.5, label=labels[0])
        self.axes.plot(wavelengths, processed, 'r-', linewidth=1.5, label=labels[1])
        
        self.axes.grid(True, linestyle='--', alpha=0.7)
        self.axes.legend()
        
        # 更新画布
        self.fig.tight_layout()
        self.draw()


class PCAPlotCanvas(FigureCanvas):
    """PCA绘图画布类"""
    
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        """
        初始化PCA绘图画布
        
        参数:
            parent: 父级窗口部件
            width: 画布宽度（英寸）
            height: 画布高度（英寸）
            dpi: 分辨率
        """
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        
        # 初始化画布
        super(PCAPlotCanvas, self).__init__(self.fig)
        self.setParent(parent)
        
        # 设置画布大小策略
        FigureCanvas.setSizePolicy(self,
                               QSizePolicy.Expanding,
                               QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        
        # 创建子图
        self.axes_scores = self.fig.add_subplot(121)
        self.axes_loadings = self.fig.add_subplot(122)
        
        # 初始化绘图
        self.clear_plot()
    
    def clear_plot(self):
        """清除绘图"""
        self.axes_scores.clear()
        self.axes_loadings.clear()
        
        self.axes_scores.set_title("得分图")
        self.axes_scores.set_xlabel("PC1")
        self.axes_scores.set_ylabel("PC2")
        self.axes_scores.grid(True, linestyle='--', alpha=0.7)
        
        self.axes_loadings.set_title("载荷图")
        self.axes_loadings.set_xlabel("波长 (nm)")
        self.axes_loadings.set_ylabel("载荷")
        self.axes_loadings.grid(True, linestyle='--', alpha=0.7)
        
        self.fig.tight_layout()
        self.draw()
    
    def plot_pca_results(self, scores, loadings, wavelengths=None, sample_names=None, 
                        explained_variance=None, pc_x=0, pc_y=1):
        """
        绘制PCA分析结果
        
        参数:
            scores: PCA得分
            loadings: PCA载荷
            wavelengths: 波长数组（用于载荷图）
            sample_names: 样本名称（用于得分图标记）
            explained_variance: 解释方差比例
            pc_x: x轴主成分索引
            pc_y: y轴主成分索引
        """
        self.axes_scores.clear()
        self.axes_loadings.clear()
        
        # 设置轴标签，包含解释方差比例（如果提供）
        pc_x_label = f"PC{pc_x+1}"
        pc_y_label = f"PC{pc_y+1}"
        
        if explained_variance is not None:
            pc_x_label += f" ({explained_variance[pc_x]:.1%})"
            pc_y_label += f" ({explained_variance[pc_y]:.1%})"
        
        # 绘制得分图
        self.axes_scores.set_title("得分图")
        self.axes_scores.set_xlabel(pc_x_label)
        self.axes_scores.set_ylabel(pc_y_label)
        
        # 绘制得分散点图
        self.axes_scores.scatter(scores[:, pc_x], scores[:, pc_y], 
                              marker='o', s=50, c='b', alpha=0.7)
        
        # 如果提供了样本名称，添加文本标记
        if sample_names:
            for i, name in enumerate(sample_names):
                self.axes_scores.annotate(name, (scores[i, pc_x], scores[i, pc_y]), 
                                      fontsize=8, ha='right', va='bottom')
        
        # 绘制原点十字线
        self.axes_scores.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        self.axes_scores.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        self.axes_scores.grid(True, linestyle='--', alpha=0.7)
        
        # 绘制载荷图
        self.axes_loadings.set_title("载荷图")
        
        if wavelengths is not None:
            self.axes_loadings.set_xlabel("波长 (nm)")
            x_values = wavelengths
        else:
            self.axes_loadings.set_xlabel("变量")
            x_values = np.arange(loadings.shape[1])
        
        self.axes_loadings.set_ylabel("载荷")
        
        # 绘制前两个主成分的载荷
        self.axes_loadings.plot(x_values, loadings[pc_x], 'b-', 
                             label=pc_x_label, linewidth=1.5)
        self.axes_loadings.plot(x_values, loadings[pc_y], 'r-', 
                             label=pc_y_label, linewidth=1.5)
        
        self.axes_loadings.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        self.axes_loadings.grid(True, linestyle='--', alpha=0.7)
        self.axes_loadings.legend()
        
        # 更新画布
        self.fig.tight_layout()
        self.draw()


class PredictionPlotCanvas(FigureCanvas):
    """预测结果绘图画布类"""
    
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        """
        初始化预测结果绘图画布
        
        参数:
            parent: 父级窗口部件
            width: 画布宽度（英寸）
            height: 画布高度（英寸）
            dpi: 分辨率
        """
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        
        # 初始化画布
        super(PredictionPlotCanvas, self).__init__(self.fig)
        self.setParent(parent)
        
        # 设置画布大小策略
        FigureCanvas.setSizePolicy(self,
                               QSizePolicy.Expanding,
                               QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        
        # 初始化绘图
        self.clear_plot()
    
    def clear_plot(self):
        """清除绘图"""
        self.axes.clear()
        self.axes.set_title("预测结果")
        self.axes.set_xlabel("实际值")
        self.axes.set_ylabel("预测值")
        self.axes.grid(True, linestyle='--', alpha=0.7)
        self.draw()
    
    def plot_prediction_results(self, y_true, y_pred, title="预测结果", 
                              xlabel="实际值", ylabel="预测值", 
                              sample_names=None, metrics=None):
        """
        绘制预测结果散点图
        
        参数:
            y_true: 实际值
            y_pred: 预测值
            title: 图标题
            xlabel: x轴标签
            ylabel: y轴标签
            sample_names: 样本名称（用于标记点）
            metrics: 性能指标字典，如 {'R²': 0.95, 'RMSE': 0.123}
        """
        self.axes.clear()
        
        # 设置标题和轴标签
        self.axes.set_title(title)
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        
        # 绘制散点图
        self.axes.scatter(y_true, y_pred, c='b', marker='o', alpha=0.7)
        
        # 如果提供了样本名称，添加文本标记
        if sample_names:
            for i, name in enumerate(sample_names):
                self.axes.annotate(name, (y_true[i], y_pred[i]), 
                                fontsize=8, ha='right', va='bottom')
        
        # 绘制理想线 (y=x)
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        padding = (max_val - min_val) * 0.05  # 添加5%的边距
        line_x = np.array([min_val - padding, max_val + padding])
        self.axes.plot(line_x, line_x, 'r--', alpha=0.7, label='理想线 (y=x)')
        
        # 设置轴范围
        self.axes.set_xlim(min_val - padding, max_val + padding)
        self.axes.set_ylim(min_val - padding, max_val + padding)
        
        # 如果提供了性能指标，添加到图中
        if metrics:
            metrics_text = "\n".join([f"{key}: {value:.4f}" for key, value in metrics.items()])
            self.axes.text(0.05, 0.95, metrics_text, transform=self.axes.transAxes,
                        verticalalignment='top', horizontalalignment='left',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        self.axes.grid(True, linestyle='--', alpha=0.7)
        self.axes.legend()
        
        # 更新画布
        self.fig.tight_layout()
        self.draw() 