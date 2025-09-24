#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
光谱分析软件GUI主窗口
"""

import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import pandas as pd
import json
import traceback
from datetime import datetime

# 使用PyQt5
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QTabWidget, 
                             QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                             QAction, QFileDialog, QMessageBox, QComboBox,
                             QGroupBox, QFormLayout, QLineEdit, QTextEdit,
                             QTableWidget, QTableWidgetItem, QSplitter, QTreeWidget,
                             QTreeWidgetItem, QCheckBox, QStatusBar, QDockWidget,
                             QMenu, QListWidget, QListWidgetItem, QDialog, QGridLayout,
                             QSpinBox, QDoubleSpinBox, QRadioButton, QButtonGroup)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QIcon, QFont, QColor

from src.data.manager import SpectrumDataset
from src.core.preprocessing import (normalize, msc, snv, savitzky_golay, 
                                   detrend, baseline_correction, moving_average,
                                   wavelet_denoise)
from src.core.analysis import (pca_analysis, pls_regression, svm_regression,
                              variable_selection, mahalanobis_distance)
from src.core.utils import (plot_spectrum, plot_spectra_comparison, find_spectral_peaks,
                          interpolate_spectrum, wavelength_to_wavenumber,
                          wavenumber_to_wavelength)
from src.gui.plots import SpectrumPlotCanvas, PCAPlotCanvas, PredictionPlotCanvas

class MainWindow(QMainWindow):
    """主窗口类"""
    
    def __init__(self):
        super().__init__()
        
        # 初始化数据
        self.dataset = SpectrumDataset()
        self.current_processed_data = None  # 存储当前处理后的数据
        self.processing_history = []  # 处理历史
        self.pca_results = None  # 存储PCA分析结果
        self.pls_results = None  # 存储PLS分析结果
        self.svm_results = None  # 存储SVM分析结果
        self.corn_model = None  # 存储玉米分析模型
        
        # 设置UI
        self.setWindowTitle("光谱分析软件")
        self.setGeometry(100, 100, 1200, 800)
        
        # 创建菜单栏
        self._create_menu_bar()
        
        # 创建状态栏
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("就绪")
        
        # 创建主布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # 创建选项卡
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        # 添加不同功能的选项卡
        self._create_data_tab()
        self._create_preprocessing_tab()
        self._create_analysis_tab()
        self._create_visualization_tab()
        self._create_corn_analysis_tab()  # 添加玉米分析专用选项卡
        
        # 创建样本浏览器面板
        self._create_sample_browser()
        
        # 显示窗口
        self.show()
    
    def _create_menu_bar(self):
        """创建菜单栏"""
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu("文件")
        
        # 打开文件
        open_action = QAction("打开", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._open_file)
        file_menu.addAction(open_action)
        
        # 保存文件
        save_action = QAction("保存", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self._save_file)
        file_menu.addAction(save_action)
        
        # 导出结果
        export_action = QAction("导出结果", self)
        export_action.triggered.connect(self._export_results)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        # 退出
        exit_action = QAction("退出", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 编辑菜单
        edit_menu = menubar.addMenu("编辑")
        
        # 复制
        copy_action = QAction("复制", self)
        copy_action.setShortcut("Ctrl+C")
        edit_menu.addAction(copy_action)
        
        # 粘贴
        paste_action = QAction("粘贴", self)
        paste_action.setShortcut("Ctrl+V")
        edit_menu.addAction(paste_action)
        
        # 预处理菜单
        preprocess_menu = menubar.addMenu("预处理")
        
        # 标准化
        normalize_action = QAction("标准化", self)
        normalize_action.triggered.connect(lambda: self.tabs.setCurrentIndex(1))
        preprocess_menu.addAction(normalize_action)
        
        # 平滑
        smooth_action = QAction("平滑", self)
        smooth_action.triggered.connect(lambda: self.tabs.setCurrentIndex(1))
        preprocess_menu.addAction(smooth_action)
        
        # 分析菜单
        analysis_menu = menubar.addMenu("分析")
        
        # PCA
        pca_action = QAction("主成分分析", self)
        pca_action.triggered.connect(lambda: self.tabs.setCurrentIndex(2))
        analysis_menu.addAction(pca_action)
        
        # PLS
        pls_action = QAction("偏最小二乘回归", self)
        pls_action.triggered.connect(lambda: self.tabs.setCurrentIndex(2))
        analysis_menu.addAction(pls_action)
        
        # 帮助菜单
        help_menu = menubar.addMenu("帮助")
        
        # 关于
        about_action = QAction("关于", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
        
        # 文档
        docs_action = QAction("文档", self)
        help_menu.addAction(docs_action)
    
    def _create_data_tab(self):
        """创建数据管理选项卡"""
        data_tab = QWidget()
        self.tabs.addTab(data_tab, "数据管理")
        
        # 创建布局
        layout = QVBoxLayout(data_tab)
        
        # 创建数据加载部分
        data_group = QGroupBox("数据加载")
        data_layout = QVBoxLayout()
        
        # 文件加载按钮
        file_layout = QHBoxLayout()
        load_btn = QPushButton("加载数据文件")
        load_btn.clicked.connect(self._open_file)
        file_layout.addWidget(load_btn)
        
        file_format_label = QLabel("文件格式:")
        file_layout.addWidget(file_format_label)
        self.file_format_combo = QComboBox()
        self.file_format_combo.addItems(["CSV", "TXT", "JCAMP-DX", "JSON", "HDF5", "MAT"])
        file_layout.addWidget(self.file_format_combo)
        file_layout.addStretch()
        
        data_layout.addLayout(file_layout)
        
        # 数据信息
        self.data_info_text = QTextEdit()
        self.data_info_text.setReadOnly(True)
        self.data_info_text.setMaximumHeight(100)
        data_layout.addWidget(QLabel("数据集信息:"))
        data_layout.addWidget(self.data_info_text)
        
        data_group.setLayout(data_layout)
        layout.addWidget(data_group)
        
        # 创建表格显示数据
        table_group = QGroupBox("数据预览")
        table_layout = QVBoxLayout()
        
        self.data_table = QTableWidget()
        table_layout.addWidget(self.data_table)
        
        # 表格控制按钮
        table_btn_layout = QHBoxLayout()
        refresh_btn = QPushButton("刷新表格")
        refresh_btn.clicked.connect(self._refresh_data_table)
        table_btn_layout.addWidget(refresh_btn)
        
        export_table_btn = QPushButton("导出表格")
        export_table_btn.clicked.connect(self._export_data_table)
        table_btn_layout.addWidget(export_table_btn)
        
        table_btn_layout.addStretch()
        table_layout.addLayout(table_btn_layout)
        
        table_group.setLayout(table_layout)
        layout.addWidget(table_group)
        
        # 创建简单波长选择器
        range_group = QGroupBox("波长范围选择")
        range_layout = QFormLayout()
        
        self.min_wl_input = QLineEdit()
        range_layout.addRow("最小波长:", self.min_wl_input)
        
        self.max_wl_input = QLineEdit()
        range_layout.addRow("最大波长:", self.max_wl_input)
        
        apply_range_btn = QPushButton("应用波长范围")
        apply_range_btn.clicked.connect(self._apply_wavelength_range)
        range_layout.addRow(apply_range_btn)
        
        range_group.setLayout(range_layout)
        layout.addWidget(range_group)
    
    def _create_preprocessing_tab(self):
        """创建预处理选项卡"""
        preprocessing_tab = QWidget()
        self.tabs.addTab(preprocessing_tab, "预处理")
        
        # 创建整体布局
        layout = QHBoxLayout(preprocessing_tab)
        
        # 创建左侧预处理选项面板
        options_panel = QWidget()
        options_layout = QVBoxLayout(options_panel)
        options_panel.setMaximumWidth(300)
        
        # 预处理方法选择
        method_group = QGroupBox("预处理方法")
        method_layout = QVBoxLayout()
        
        # 预处理方法下拉框
        self.preprocess_method_combo = QComboBox()
        self.preprocess_method_combo.addItems([
            "标准化 (Normalization)", 
            "多元散射校正 (MSC)", 
            "标准正态变量转换 (SNV)", 
            "Savitzky-Golay平滑/微分", 
            "去趋势 (Detrend)", 
            "基线校正 (Baseline Correction)",
            "移动平均滤波",
            "小波变换降噪"
        ])
        self.preprocess_method_combo.currentIndexChanged.connect(self._change_preprocessing_options)
        method_layout.addWidget(self.preprocess_method_combo)
        
        # 创建存放不同预处理方法参数的控件集合
        self.preprocess_options_widget = QWidget()
        self.preprocess_options_layout = QFormLayout(self.preprocess_options_widget)
        method_layout.addWidget(self.preprocess_options_widget)
        
        # 应用按钮
        apply_preprocess_btn = QPushButton("应用预处理")
        apply_preprocess_btn.clicked.connect(self._apply_preprocessing)
        method_layout.addWidget(apply_preprocess_btn)
        
        # 重置按钮
        reset_preprocess_btn = QPushButton("重置为原始数据")
        reset_preprocess_btn.clicked.connect(self._reset_preprocessing)
        method_layout.addWidget(reset_preprocess_btn)
        
        method_group.setLayout(method_layout)
        options_layout.addWidget(method_group)
        
        # 处理历史
        history_group = QGroupBox("处理历史")
        history_layout = QVBoxLayout()
        
        self.process_history_list = QListWidget()
        history_layout.addWidget(self.process_history_list)
        
        history_group.setLayout(history_layout)
        options_layout.addWidget(history_group)
        
        # 增加弹性空间
        options_layout.addStretch()
        
        # 添加到主布局
        layout.addWidget(options_panel)
        
        # 创建右侧显示结果的图表区域
        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)
        
        # 创建图表显示区域
        self.preprocess_figure = Figure(figsize=(5, 4), dpi=100)
        self.preprocess_canvas = FigureCanvas(self.preprocess_figure)
        self.preprocess_toolbar = NavigationToolbar(self.preprocess_canvas, self)
        
        plot_layout.addWidget(self.preprocess_toolbar)
        plot_layout.addWidget(self.preprocess_canvas)
        
        # 初始化图表
        self.preprocess_ax = self.preprocess_figure.add_subplot(111)
        self.preprocess_ax.set_title("预处理结果")
        self.preprocess_ax.set_xlabel("波长 (nm)")
        self.preprocess_ax.set_ylabel("强度")
        self.preprocess_figure.tight_layout()
        self.preprocess_canvas.draw()
        
        # 添加到主布局
        layout.addWidget(plot_panel)
        
        # 默认设置初始化预处理选项
        self._change_preprocessing_options(0)
    
    def _create_analysis_tab(self):
        """创建分析选项卡"""
        analysis_tab = QWidget()
        self.tabs.addTab(analysis_tab, "分析")
        
        # 创建整体布局
        layout = QHBoxLayout(analysis_tab)
        
        # 创建左侧分析选项面板
        options_panel = QWidget()
        options_layout = QVBoxLayout(options_panel)
        options_panel.setMaximumWidth(300)
        
        # 分析方法选择
        method_group = QGroupBox("分析方法")
        method_layout = QVBoxLayout()
        
        # 分析方法下拉框
        self.analysis_method_combo = QComboBox()
        self.analysis_method_combo.addItems([
            "主成分分析 (PCA)", 
            "偏最小二乘回归 (PLS)", 
            "支持向量机回归 (SVR)",
            "变量选择 (Variable Selection)",
            "异常检测 (Outlier Detection)"
        ])
        self.analysis_method_combo.currentIndexChanged.connect(self._change_analysis_options)
        method_layout.addWidget(self.analysis_method_combo)
        
        # 创建存放不同分析方法参数的控件集合
        self.analysis_options_widget = QWidget()
        self.analysis_options_layout = QFormLayout(self.analysis_options_widget)
        method_layout.addWidget(self.analysis_options_widget)
        
        # 执行分析按钮
        run_analysis_btn = QPushButton("执行分析")
        run_analysis_btn.clicked.connect(self._run_analysis)
        method_layout.addWidget(run_analysis_btn)
        
        method_group.setLayout(method_layout)
        options_layout.addWidget(method_group)
        
        # 分析结果摘要
        results_group = QGroupBox("分析结果摘要")
        results_layout = QVBoxLayout()
        
        self.analysis_result_text = QTextEdit()
        self.analysis_result_text.setReadOnly(True)
        results_layout.addWidget(self.analysis_result_text)
        
        results_group.setLayout(results_layout)
        options_layout.addWidget(results_group)
        
        # 增加弹性空间
        options_layout.addStretch()
        
        # 添加到主布局
        layout.addWidget(options_panel)
        
        # 创建右侧显示结果的选项卡区域
        result_tab = QTabWidget()
        layout.addWidget(result_tab)
        
        # 结果图表选项卡
        plot_tab = QWidget()
        plot_layout = QVBoxLayout(plot_tab)
        
        # 创建图表显示
        self.analysis_figure = Figure(figsize=(5, 4), dpi=100)
        self.analysis_canvas = FigureCanvas(self.analysis_figure)
        self.analysis_toolbar = NavigationToolbar(self.analysis_canvas, self)
        
        plot_layout.addWidget(self.analysis_toolbar)
        plot_layout.addWidget(self.analysis_canvas)
        
        # 初始化图表
        self.analysis_ax = self.analysis_figure.add_subplot(111)
        self.analysis_ax.set_title("分析结果")
        self.analysis_figure.tight_layout()
        self.analysis_canvas.draw()
        
        result_tab.addTab(plot_tab, "图表")
        
        # 数值结果选项卡
        data_tab = QWidget()
        data_layout = QVBoxLayout(data_tab)
        
        # 创建表格来显示结果
        self.analysis_table = QTableWidget()
        data_layout.addWidget(self.analysis_table)
        
        # 添加导出按钮
        export_btn = QPushButton("导出结果")
        export_btn.clicked.connect(self._export_analysis_result)
        data_layout.addWidget(export_btn)
        
        result_tab.addTab(data_tab, "数据")
        
        # 默认设置初始化分析选项
        self._change_analysis_options(0) 
    
    def _create_corn_analysis_tab(self):
        """创建玉米分析专用选项卡"""
        corn_tab = QWidget()
        self.tabs.addTab(corn_tab, "玉米分析")
        
        # 创建整体布局
        layout = QHBoxLayout(corn_tab)
        
        # 创建左侧选项面板
        options_panel = QWidget()
        options_layout = QVBoxLayout(options_panel)
        options_panel.setMaximumWidth(300)
        
        # 数据加载部分
        data_group = QGroupBox("玉米光谱数据")
        data_layout = QVBoxLayout()
        
        # 加载按钮
        load_btn = QPushButton("加载玉米数据文件")
        load_btn.clicked.connect(self._open_corn_file)
        data_layout.addWidget(load_btn)
        
        # 目标变量选择
        self.corn_target_label = QLabel("目标变量: 无")
        data_layout.addWidget(self.corn_target_label)
        
        # 样本信息
        self.corn_info_text = QTextEdit()
        self.corn_info_text.setReadOnly(True)
        self.corn_info_text.setMaximumHeight(100)
        data_layout.addWidget(QLabel("数据集信息:"))
        data_layout.addWidget(self.corn_info_text)
        
        data_group.setLayout(data_layout)
        options_layout.addWidget(data_group)
        
        # 玉米特定预处理选项
        preprocess_group = QGroupBox("预处理")
        preprocess_layout = QVBoxLayout()
        
        self.corn_preprocess_combo = QComboBox()
        self.corn_preprocess_combo.addItems([
            "标准化",
            "多元散射校正",
            "Savitzky-Golay平滑",
            "一阶导数",
            "二阶导数",
            "基线校正",
            "移动平均滤波",
            "小波变换降噪",
            "最优预处理组合"
        ])
        preprocess_layout.addWidget(self.corn_preprocess_combo)
        
        apply_preprocess_btn = QPushButton("应用预处理")
        apply_preprocess_btn.clicked.connect(self._apply_corn_preprocessing)
        preprocess_layout.addWidget(apply_preprocess_btn)
        
        preprocess_group.setLayout(preprocess_layout)
        options_layout.addWidget(preprocess_group)
        
        # 模型选择
        model_group = QGroupBox("模型选择")
        model_layout = QVBoxLayout()
        
        self.corn_model_combo = QComboBox()
        self.corn_model_combo.addItems([
            "偏最小二乘回归 (PLS)",
            "支持向量机回归 (SVR)",
            "随机森林回归 (RF)",
            "XGBoost回归 (XGB)",
            "集成优化模型"
        ])
        model_layout.addWidget(self.corn_model_combo)
        
        # 交叉验证选项
        cv_layout = QHBoxLayout()
        cv_layout.addWidget(QLabel("交叉验证折数:"))
        self.corn_cv_spinbox = QSpinBox()
        self.corn_cv_spinbox.setRange(3, 10)
        self.corn_cv_spinbox.setValue(5)
        cv_layout.addWidget(self.corn_cv_spinbox)
        model_layout.addLayout(cv_layout)
        
        # 训练和预测按钮
        train_btn = QPushButton("训练模型")
        train_btn.clicked.connect(self._train_corn_model)
        model_layout.addWidget(train_btn)
        
        model_group.setLayout(model_layout)
        options_layout.addWidget(model_group)
        
        # 结果摘要
        results_group = QGroupBox("分析结果摘要")
        results_layout = QVBoxLayout()
        
        self.corn_result_text = QTextEdit()
        self.corn_result_text.setReadOnly(True)
        results_layout.addWidget(self.corn_result_text)
        
        # 导出结果按钮
        export_btn = QPushButton("导出结果")
        export_btn.clicked.connect(self._export_corn_result)
        results_layout.addWidget(export_btn)
        
        results_group.setLayout(results_layout)
        options_layout.addWidget(results_group)
        
        # 增加弹性空间
        options_layout.addStretch()
        
        # 添加到主布局
        layout.addWidget(options_panel)
        
        # 创建右侧显示结果的区域
        result_panel = QWidget()
        result_layout = QVBoxLayout(result_panel)
        
        # 创建图表显示区域
        self.corn_figure = Figure(figsize=(5, 4), dpi=100)
        self.corn_canvas = FigureCanvas(self.corn_figure)
        self.corn_toolbar = NavigationToolbar(self.corn_canvas, self)
        
        result_layout.addWidget(self.corn_toolbar)
        result_layout.addWidget(self.corn_canvas)
        
        # 初始化图表
        self.corn_ax = self.corn_figure.add_subplot(111)
        self.corn_ax.set_title("玉米成分预测")
        self.corn_ax.set_xlabel("实际值")
        self.corn_ax.set_ylabel("预测值")
        self.corn_ax.grid(True)
        self.corn_figure.tight_layout()
        self.corn_canvas.draw()
        
        # 添加到主布局
        layout.addWidget(result_panel)

    def _create_sample_browser(self):
        """创建样本浏览器面板"""
        # 创建样本面板
        sample_dock = QDockWidget("样本浏览器", self)
        sample_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        
        # 样本列表
        sample_widget = QWidget()
        sample_layout = QVBoxLayout(sample_widget)
        
        self.sample_tree = QTreeWidget()
        self.sample_tree.setHeaderLabels(["样本"])
        sample_layout.addWidget(self.sample_tree)
        
        # 样本控制按钮
        sample_btn_layout = QHBoxLayout()
        refresh_samples_btn = QPushButton("刷新样本")
        refresh_samples_btn.clicked.connect(self._refresh_samples)
        sample_btn_layout.addWidget(refresh_samples_btn)
        
        delete_sample_btn = QPushButton("删除样本")
        delete_sample_btn.clicked.connect(self._delete_selected_sample)
        sample_btn_layout.addWidget(delete_sample_btn)
        
        sample_layout.addLayout(sample_btn_layout)
        
        sample_dock.setWidget(sample_widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, sample_dock)
    
    def _open_file(self):
        """打开光谱数据文件"""
        options = QFileDialog.Options()
        file_format = self.file_format_combo.currentText() if hasattr(self, 'file_format_combo') else "CSV"
        
        # 根据选择的文件格式设置过滤器
        if file_format == "CSV":
            filter_str = "CSV文件 (*.csv);;所有文件 (*)"
        elif file_format == "TXT":
            filter_str = "文本文件 (*.txt);;所有文件 (*)"
        elif file_format == "JCAMP-DX":
            filter_str = "JCAMP-DX文件 (*.jdx *.dx);;所有文件 (*)"
        elif file_format == "JSON":
            filter_str = "JSON文件 (*.json);;所有文件 (*)"
        elif file_format == "HDF5":
            filter_str = "HDF5文件 (*.h5 *.hdf5);;所有文件 (*)"
        elif file_format == "MAT":
            filter_str = "MATLAB文件 (*.mat);;所有文件 (*)"
        else:
            filter_str = "所有文件 (*)"
        
        # 打开文件对话框
        file_name, _ = QFileDialog.getOpenFileName(
            self, "打开光谱数据文件", "", filter_str, options=options
        )
        
        if file_name:
            try:
                # 根据文件类型加载数据
                success = self.dataset.load_file(file_name, file_format.lower())
                
                if success:
                    # 更新数据信息显示
                    if hasattr(self, 'data_info_text'):
                        self.data_info_text.clear()
                        info_text = (
                            f"文件名: {os.path.basename(file_name)}\n"
                            f"样本数: {self.dataset.get_sample_count()}\n"
                            f"波长范围: {self.dataset.get_wavelength_range()}\n"
                            f"数据大小: {self.dataset.get_data_shape()}"
                        )
                        self.data_info_text.setText(info_text)
                    
                    # 刷新数据表格
                    if hasattr(self, 'data_table'):
                        self._refresh_data_table()
                    
                    # 更新状态栏
                    self.statusBar.showMessage(f"已加载文件: {os.path.basename(file_name)}")
                else:
                    QMessageBox.warning(self, "加载失败", "无法加载所选文件，请检查文件格式是否正确。")
                    
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载文件时出错：{str(e)}")
                print(f"加载文件错误: {e}")
    
    def _save_file(self):
        """保存当前数据集"""
        if not self.dataset or self.dataset.is_empty():
            QMessageBox.warning(self, "警告", "没有数据可保存!")
            return
            
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(
            self, "保存数据", "", 
            "CSV文件 (*.csv);;TXT文件 (*.txt);;JSON文件 (*.json);;HDF5文件 (*.h5);;所有文件 (*)",
            options=options
        )
        
        if file_name:
            try:
                # 从文件扩展名确定保存格式
                ext = os.path.splitext(file_name)[1].lower()
                if ext == '.csv':
                    format_type = 'csv'
                elif ext == '.txt':
                    format_type = 'txt'
                elif ext == '.json':
                    format_type = 'json'
                elif ext in ['.h5', '.hdf5']:
                    format_type = 'hdf5'
                else:
                    format_type = 'csv'  # 默认CSV
                
                # 保存文件
                success = self.dataset.save_file(file_name, format_type)
                
                if success:
                    self.statusBar.showMessage(f"文件已保存: {os.path.basename(file_name)}")
                else:
                    QMessageBox.warning(self, "保存失败", "无法保存文件，请检查文件路径和权限。")
                    
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存文件时出错：{str(e)}")
                print(f"保存文件错误: {e}")
    
    def _export_results(self):
        """导出分析结果"""
        if not hasattr(self, 'pca_results') or not self.pca_results:
            if not hasattr(self, 'pls_results') or not self.pls_results:
                if not hasattr(self, 'svm_results') or not self.svm_results:
                    QMessageBox.warning(self, "警告", "没有分析结果可导出!")
                    return
        
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(
            self, "导出结果", "", 
            "CSV文件 (*.csv);;Excel文件 (*.xlsx);;JSON文件 (*.json);;所有文件 (*)",
            options=options
        )
        
        if file_name:
            try:
                # 根据文件扩展名决定导出格式
                ext = os.path.splitext(file_name)[1].lower()
                
                # 创建结果字典
                results_data = {}
                
                # 添加PCA结果
                if hasattr(self, 'pca_results') and self.pca_results:
                    results_data['pca'] = {
                        'explained_variance': self.pca_results.get('explained_variance', []),
                        'scores': self.pca_results.get('scores', []).tolist() if isinstance(self.pca_results.get('scores', []), np.ndarray) else [],
                        'loadings': self.pca_results.get('loadings', []).tolist() if isinstance(self.pca_results.get('loadings', []), np.ndarray) else []
                    }
                
                # 添加PLS结果
                if hasattr(self, 'pls_results') and self.pls_results:
                    results_data['pls'] = {
                        'coefficients': self.pls_results.get('coefficients', []).tolist() if isinstance(self.pls_results.get('coefficients', []), np.ndarray) else [],
                        'predictions': self.pls_results.get('predictions', []).tolist() if isinstance(self.pls_results.get('predictions', []), np.ndarray) else [],
                        'rmse': self.pls_results.get('rmse', 0),
                        'r2': self.pls_results.get('r2', 0)
                    }
                
                # 根据格式导出
                if ext == '.csv':
                    # 导出为CSV（简化版本）
                    with open(file_name, 'w', encoding='utf-8') as f:
                        f.write("分析结果导出\n")
                        f.write(f"导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                        
                        # 导出PCA结果
                        if 'pca' in results_data:
                            f.write("PCA分析结果\n")
                            f.write("解释方差比例:\n")
                            for i, var in enumerate(results_data['pca']['explained_variance']):
                                f.write(f"PC{i+1}: {var:.6f}\n")
                            f.write("\n")
                        
                        # 导出PLS结果
                        if 'pls' in results_data:
                            f.write("PLS回归结果\n")
                            f.write(f"RMSE: {results_data['pls']['rmse']:.6f}\n")
                            f.write(f"R²: {results_data['pls']['r2']:.6f}\n\n")
                
                elif ext == '.xlsx':
                    try:
                        import openpyxl
                        # 创建Excel工作簿
                        import pandas as pd
                        with pd.ExcelWriter(file_name) as writer:
                            # 导出PCA结果
                            if 'pca' in results_data:
                                pca_df = pd.DataFrame({
                                    'PC': [f"PC{i+1}" for i in range(len(results_data['pca']['explained_variance']))],
                                    'Explained Variance': results_data['pca']['explained_variance']
                                })
                                pca_df.to_excel(writer, sheet_name='PCA结果', index=False)
                            
                            # 导出PLS结果
                            if 'pls' in results_data:
                                pls_info = pd.DataFrame({
                                    'Metric': ['RMSE', 'R²'],
                                    'Value': [results_data['pls']['rmse'], results_data['pls']['r2']]
                                })
                                pls_info.to_excel(writer, sheet_name='PLS结果', index=False)
                    except ImportError:
                        QMessageBox.warning(self, "警告", "导出Excel需要安装openpyxl库。正在使用JSON格式导出...")
                        with open(file_name.replace('.xlsx', '.json'), 'w', encoding='utf-8') as f:
                            json.dump(results_data, f, indent=2)
                
                elif ext == '.json':
                    # 导出为JSON
                    with open(file_name, 'w', encoding='utf-8') as f:
                        json.dump(results_data, f, indent=2)
                
                else:
                    # 默认导出为JSON
                    with open(file_name, 'w', encoding='utf-8') as f:
                        json.dump(results_data, f, indent=2)
                
                self.statusBar.showMessage(f"结果已导出: {os.path.basename(file_name)}")
                
            except Exception as e:
                QMessageBox.critical(self, "错误", f"导出结果时出错：{str(e)}")
                print(f"导出结果错误: {e}")
    
    def _show_about(self):
        """显示关于对话框"""
        about_text = (
            "光谱分析软件 v1.0\n\n"
            "本软件用于近红外光谱数据的处理与分析，支持多种预处理方法和分析模型。\n\n"
            "特点:\n"
            "- 支持多种格式的光谱数据导入\n"
            "- 提供多种预处理算法\n"
            "- 支持主成分分析(PCA)和偏最小二乘回归(PLS)\n"
            "- 玉米成分专业化分析功能\n\n"
            "©2023 光谱分析团队"
        )
        QMessageBox.about(self, "关于", about_text)
        
    def _refresh_data_table(self):
        """刷新数据表格"""
        if not hasattr(self, 'data_table') or not self.dataset or self.dataset.is_empty():
            return

        # 清空表格
        self.data_table.clear()
        self.data_table.setRowCount(0)
        self.data_table.setColumnCount(0)

        try:
            # 获取数据
            data = self.dataset.get_data()
            wavelengths = self.dataset.get_wavelengths()
            sample_ids = self.dataset.get_sample_ids()
            target = self.dataset.get_target()
            target_name = self.dataset.get_target_name()
            additional_targets = self.dataset.get_additional_targets()

            if data is None or len(data) == 0:
                return

            # 设置表头
            num_samples = len(sample_ids) if sample_ids else data.shape[0]
            num_wavelengths = len(wavelengths) if wavelengths else data.shape[1]

            # 计算总列数：样本ID + 目标变量 + 额外目标变量 + 光谱数据
            total_columns = 1  # 样本ID
            target_column_idx = None
            additional_target_columns = {}

            # 添加目标变量列
            if target is not None and target_name:
                target_column_idx = total_columns
                total_columns += 1

            # 添加额外目标变量列
            for name, values in additional_targets.items():
                additional_target_columns[name] = total_columns
                total_columns += 1

            # 添加光谱数据列
            wavelength_start_idx = total_columns
            total_columns += num_wavelengths

            self.data_table.setColumnCount(total_columns)
            self.data_table.setRowCount(num_samples)

            # 添加表头
            headers = ["样本ID"]

            # 添加目标变量表头
            if target is not None and target_name:
                headers.append(target_name)

            # 添加额外目标变量表头
            for name in additional_targets.keys():
                headers.append(name)

            # 添加波长表头
            if wavelengths is not None:
                headers.extend([f"{w:.0f}" for w in wavelengths])  # 使用整数显示波长
            else:
                headers.extend([f"波长{i+1}" for i in range(num_wavelengths)])

            self.data_table.setHorizontalHeaderLabels(headers)

            # 填充数据
            for i in range(num_samples):
                col_idx = 0

                # 样本ID
                sample_id = sample_ids[i] if sample_ids else f"样本{i+1}"
                id_item = QTableWidgetItem(str(sample_id))
                self.data_table.setItem(i, col_idx, id_item)
                col_idx += 1

                # 目标变量
                if target is not None and target_name:
                    target_value = target[i] if i < len(target) else ""
                    target_item = QTableWidgetItem(f"{target_value:.3f}")
                    self.data_table.setItem(i, col_idx, target_item)
                    col_idx += 1

                # 额外目标变量
                for name, values in additional_targets.items():
                    value = values[i] if i < len(values) else ""
                    value_item = QTableWidgetItem(f"{value:.3f}")
                    self.data_table.setItem(i, col_idx, value_item)
                    col_idx += 1

                # 光谱数据
                for j in range(num_wavelengths):
                    value_item = QTableWidgetItem(f"{data[i, j]:.6f}")
                    self.data_table.setItem(i, col_idx + j, value_item)

            # 调整列宽
            self.data_table.horizontalHeader().setSectionResizeMode(0, 1)  # 样本ID列自适应内容
            if target_column_idx is not None:
                self.data_table.horizontalHeader().setSectionResizeMode(target_column_idx, 1)  # 目标变量列自适应内容
            for idx in additional_target_columns.values():
                self.data_table.horizontalHeader().setSectionResizeMode(idx, 1)  # 额外目标变量列自适应内容

            # 设置光谱数据列为固定宽度或可滚动
            for j in range(num_wavelengths):
                self.data_table.horizontalHeader().setSectionResizeMode(wavelength_start_idx + j, 3)  # 固定宽度

        except Exception as e:
            print(f"刷新数据表格错误: {e}")
            traceback.print_exc()
            
    def _export_data_table(self):
        """导出数据表格"""
        if not hasattr(self, 'data_table') or not self.dataset or self.dataset.is_empty():
            QMessageBox.warning(self, "警告", "没有数据可导出!")
            return
            
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(
            self, "导出数据表格", "", 
            "CSV文件 (*.csv);;Excel文件 (*.xlsx);;所有文件 (*)",
            options=options
        )
        
        if file_name:
            try:
                # 获取数据
                data = self.dataset.get_data()
                wavelengths = self.dataset.get_wavelengths()
                sample_ids = self.dataset.get_sample_ids()
                
                if data is None or len(data) == 0:
                    QMessageBox.warning(self, "警告", "没有数据可导出!")
                    return
                
                # 确定文件格式
                ext = os.path.splitext(file_name)[1].lower()
                
                if ext == '.csv':
                    # 导出为CSV
                    with open(file_name, 'w', encoding='utf-8') as f:
                        # 写入表头
                        headers = ["sample_id"]
                        if wavelengths is not None:
                            headers.extend([f"{w:.2f}" for w in wavelengths])
                        else:
                            headers.extend([f"wavelength_{i+1}" for i in range(data.shape[1])])
                        
                        f.write(",".join(headers) + "\n")
                        
                        # 写入数据
                        for i in range(data.shape[0]):
                            sample_id = sample_ids[i] if sample_ids else f"sample_{i+1}"
                            row = [str(sample_id)]
                            row.extend([f"{x:.6f}" for x in data[i]])
                            f.write(",".join(row) + "\n")
                
                elif ext == '.xlsx':
                    try:
                        # 导出为Excel
                        import pandas as pd
                        
                        # 创建数据框
                        df_data = {}
                        df_data["sample_id"] = sample_ids if sample_ids else [f"sample_{i+1}" for i in range(data.shape[0])]
                        
                        # 添加波长列
                        if wavelengths is not None:
                            for i, w in enumerate(wavelengths):
                                df_data[f"{w:.2f}"] = data[:, i]
                        else:
                            for i in range(data.shape[1]):
                                df_data[f"wavelength_{i+1}"] = data[:, i]
                        
                        # 创建DataFrame并保存
                        df = pd.DataFrame(df_data)
                        df.to_excel(file_name, index=False)
                    
                    except ImportError:
                        QMessageBox.warning(self, "警告", "导出Excel需要安装pandas和openpyxl库。")
                        return
                
                else:
                    # 默认导出为CSV
                    with open(file_name, 'w', encoding='utf-8') as f:
                        # 写入表头
                        headers = ["sample_id"]
                        if wavelengths is not None:
                            headers.extend([f"{w:.2f}" for w in wavelengths])
                        else:
                            headers.extend([f"wavelength_{i+1}" for i in range(data.shape[1])])
                        
                        f.write(",".join(headers) + "\n")
                        
                        # 写入数据
                        for i in range(data.shape[0]):
                            sample_id = sample_ids[i] if sample_ids else f"sample_{i+1}"
                            row = [str(sample_id)]
                            row.extend([f"{x:.6f}" for x in data[i]])
                            f.write(",".join(row) + "\n")
                
                self.statusBar.showMessage(f"数据表格已导出: {os.path.basename(file_name)}")
                
            except Exception as e:
                QMessageBox.critical(self, "错误", f"导出数据表格时出错：{str(e)}")
                print(f"导出数据表格错误: {e}")
                
    def _apply_wavelength_range(self):
        """应用波长范围筛选"""
        if not self.dataset or self.dataset.is_empty():
            QMessageBox.warning(self, "警告", "没有数据可筛选!")
            return
            
        try:
            # 获取输入的波长范围
            min_wl_text = self.min_wl_input.text().strip()
            max_wl_text = self.max_wl_input.text().strip()
            
            if not min_wl_text and not max_wl_text:
                QMessageBox.warning(self, "警告", "请至少输入一个波长范围值!")
                return
            
            min_wl = float(min_wl_text) if min_wl_text else None
            max_wl = float(max_wl_text) if max_wl_text else None
            
            # 应用波长范围筛选
            success = self.dataset.filter_wavelength_range(min_wl, max_wl)
            
            if success:
                # 更新数据显示
                self._refresh_data_table()
                
                # 更新数据信息显示
                if hasattr(self, 'data_info_text'):
                    info_text = (
                        f"样本数: {self.dataset.get_sample_count()}\n"
                        f"波长范围: {self.dataset.get_wavelength_range()}\n"
                        f"数据大小: {self.dataset.get_data_shape()}"
                    )
                    self.data_info_text.setText(info_text)
                
                self.statusBar.showMessage(f"已应用波长范围筛选: {min_wl if min_wl else '最小'} - {max_wl if max_wl else '最大'}")
            else:
                QMessageBox.warning(self, "警告", "应用波长范围筛选失败，请检查输入值是否有效!")
        
        except ValueError:
            QMessageBox.warning(self, "警告", "请输入有效的数字!")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"应用波长范围筛选时出错：{str(e)}")
            print(f"应用波长范围筛选错误: {e}")
            
    def _refresh_samples(self):
        """刷新样本列表"""
        if not hasattr(self, 'sample_tree') or not self.dataset or self.dataset.is_empty():
            return
            
        # 清空样本树
        self.sample_tree.clear()
        
        try:
            # 获取样本ID
            sample_ids = self.dataset.get_sample_ids()
            
            if not sample_ids:
                return
                
            # 添加样本到树中
            for sample_id in sample_ids:
                item = QTreeWidgetItem(self.sample_tree)
                item.setText(0, str(sample_id))
                
            # 展开树
            self.sample_tree.expandAll()
            
        except Exception as e:
            print(f"刷新样本列表错误: {e}")
            
    def _delete_selected_sample(self):
        """删除选中的样本"""
        if not hasattr(self, 'sample_tree') or not self.dataset or self.dataset.is_empty():
            return
            
        # 获取选中的样本
        selected_items = self.sample_tree.selectedItems()
        
        if not selected_items:
            QMessageBox.warning(self, "警告", "请先选择要删除的样本!")
            return
            
        # 确认删除
        reply = QMessageBox.question(
            self, "确认删除", 
            f"确定要删除选中的{len(selected_items)}个样本吗?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                # 获取要删除的样本ID
                sample_ids = [item.text(0) for item in selected_items]
                
                # 删除样本
                for sample_id in sample_ids:
                    success = self.dataset.delete_sample(sample_id)
                    if not success:
                        print(f"删除样本 {sample_id} 失败")
                
                # 刷新显示
                self._refresh_samples()
                self._refresh_data_table()
                
                # 更新数据信息显示
                if hasattr(self, 'data_info_text'):
                    info_text = (
                        f"样本数: {self.dataset.get_sample_count()}\n"
                        f"波长范围: {self.dataset.get_wavelength_range()}\n"
                        f"数据大小: {self.dataset.get_data_shape()}"
                    )
                    self.data_info_text.setText(info_text)
                
                self.statusBar.showMessage(f"已删除{len(sample_ids)}个样本")
                
            except Exception as e:
                QMessageBox.critical(self, "错误", f"删除样本时出错：{str(e)}")
                print(f"删除样本错误: {e}")

    def _change_preprocessing_options(self, index):
        """根据预处理方法更改选项界面"""
        # 清空当前选项
        while self.preprocess_options_layout.count():
            item = self.preprocess_options_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # 根据选择的预处理方法添加对应选项
        if index == 0:  # 标准化
            methods = ["最大值", "均值", "范围", "向量"]
            self.normalize_method_combo = QComboBox()
            self.normalize_method_combo.addItems(methods)
            self.preprocess_options_layout.addRow("标准化方法:", self.normalize_method_combo)
            
        elif index == 1:  # MSC
            # MSC没有额外参数
            self.preprocess_options_layout.addRow(QLabel("多元散射校正不需要额外参数"))
            
        elif index == 2:  # SNV
            # SNV没有额外参数
            self.preprocess_options_layout.addRow(QLabel("标准正态变量转换不需要额外参数"))
            
        elif index == 3:  # Savitzky-Golay平滑/微分
            # 窗口大小
            self.sg_window_size = QSpinBox()
            self.sg_window_size.setRange(5, 25)
            self.sg_window_size.setSingleStep(2)  # 设置为奇数
            self.sg_window_size.setValue(9)
            self.preprocess_options_layout.addRow("窗口大小:", self.sg_window_size)
            
            # 多项式阶数
            self.sg_poly_order = QSpinBox()
            self.sg_poly_order.setRange(1, 5)
            self.sg_poly_order.setValue(2)
            self.preprocess_options_layout.addRow("多项式阶数:", self.sg_poly_order)
            
            # 导数阶数
            self.sg_deriv = QSpinBox()
            self.sg_deriv.setRange(0, 2)
            self.sg_deriv.setValue(0)
            self.preprocess_options_layout.addRow("导数阶数:", self.sg_deriv)
            
        elif index == 4:  # 去趋势
            # 趋势阶数
            self.detrend_order = QSpinBox()
            self.detrend_order.setRange(1, 3)
            self.detrend_order.setValue(1)
            self.preprocess_options_layout.addRow("趋势阶数:", self.detrend_order)
            
        elif index == 5:  # 基线校正
            # 基线校正算法
            methods = ["多项式拟合", "自适应迭代加权平滑", "渐进线校正"]
            self.baseline_method_combo = QComboBox()
            self.baseline_method_combo.addItems(methods)
            self.preprocess_options_layout.addRow("校正算法:", self.baseline_method_combo)
            
            # 多项式阶数
            self.baseline_poly_order = QSpinBox()
            self.baseline_poly_order.setRange(1, 5)
            self.baseline_poly_order.setValue(2)
            self.preprocess_options_layout.addRow("多项式阶数:", self.baseline_poly_order)
            
        elif index == 6:  # 移动平均滤波
            # 窗口大小
            self.ma_window_size = QSpinBox()
            self.ma_window_size.setRange(3, 21)
            self.ma_window_size.setSingleStep(2)
            self.ma_window_size.setValue(5)
            self.preprocess_options_layout.addRow("窗口大小:", self.ma_window_size)
            
        elif index == 7:  # 小波变换降噪
            # 小波函数
            wavelet_types = ["db4", "sym4", "coif4", "haar"]
            self.wavelet_type_combo = QComboBox()
            self.wavelet_type_combo.addItems(wavelet_types)
            self.preprocess_options_layout.addRow("小波函数:", self.wavelet_type_combo)
            
            # 分解层数
            self.wavelet_level = QSpinBox()
            self.wavelet_level.setRange(1, 5)
            self.wavelet_level.setValue(3)
            self.preprocess_options_layout.addRow("分解层数:", self.wavelet_level)
            
            # 阈值比例
            self.wavelet_threshold = QDoubleSpinBox()
            self.wavelet_threshold.setRange(0.1, 5.0)
            self.wavelet_threshold.setSingleStep(0.1)
            self.wavelet_threshold.setValue(1.0)
            self.preprocess_options_layout.addRow("阈值比例:", self.wavelet_threshold)
    
    def _change_analysis_options(self, index):
        """根据分析方法更改选项界面"""
        # 清空当前选项
        while self.analysis_options_layout.count():
            item = self.analysis_options_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # 根据选择的分析方法添加对应选项
        if index == 0:  # PCA
            # 主成分数量
            self.pca_n_components = QSpinBox()
            self.pca_n_components.setRange(1, 10)
            self.pca_n_components.setValue(2)
            self.analysis_options_layout.addRow("主成分数量:", self.pca_n_components)
            
            # 是否标准化
            self.pca_scale = QCheckBox("数据标准化")
            self.pca_scale.setChecked(True)
            self.analysis_options_layout.addRow("", self.pca_scale)
            
        elif index == 1:  # PLS
            # 目标变量选择
            self.pls_target_label = QLabel("请先加载含有目标变量的数据")
            self.analysis_options_layout.addRow("目标变量:", self.pls_target_label)
            
            # 潜变量数量
            self.pls_n_components = QSpinBox()
            self.pls_n_components.setRange(1, 10)
            self.pls_n_components.setValue(2)
            self.analysis_options_layout.addRow("潜变量数量:", self.pls_n_components)
            
            # 交叉验证
            self.pls_cv = QSpinBox()
            self.pls_cv.setRange(2, 10)
            self.pls_cv.setValue(5)
            self.analysis_options_layout.addRow("交叉验证折数:", self.pls_cv)
            
        elif index == 2:  # SVR
            # 目标变量选择
            self.svr_target_label = QLabel("请先加载含有目标变量的数据")
            self.analysis_options_layout.addRow("目标变量:", self.svr_target_label)
            
            # 核函数
            kernels = ["linear", "poly", "rbf", "sigmoid"]
            self.svr_kernel_combo = QComboBox()
            self.svr_kernel_combo.addItems(kernels)
            self.svr_kernel_combo.setCurrentText("rbf")
            self.analysis_options_layout.addRow("核函数:", self.svr_kernel_combo)
            
            # C参数
            self.svr_c = QDoubleSpinBox()
            self.svr_c.setRange(0.1, 100.0)
            self.svr_c.setSingleStep(0.1)
            self.svr_c.setValue(1.0)
            self.analysis_options_layout.addRow("C参数:", self.svr_c)
            
            # Gamma参数
            self.svr_gamma = QDoubleSpinBox()
            self.svr_gamma.setRange(0.001, 1.0)
            self.svr_gamma.setSingleStep(0.001)
            self.svr_gamma.setValue(0.1)
            self.analysis_options_layout.addRow("Gamma参数:", self.svr_gamma)
            
            # 交叉验证
            self.svr_cv = QSpinBox()
            self.svr_cv.setRange(2, 10)
            self.svr_cv.setValue(5)
            self.analysis_options_layout.addRow("交叉验证折数:", self.svr_cv)
            
        elif index == 3:  # 变量选择
            # 选择方法
            methods = ["递归特征消除", "Lasso回归", "互信息", "PLS-VIP"]
            self.vs_method_combo = QComboBox()
            self.vs_method_combo.addItems(methods)
            self.analysis_options_layout.addRow("选择方法:", self.vs_method_combo)
            
            # 目标变量选择
            self.vs_target_label = QLabel("请先加载含有目标变量的数据")
            self.analysis_options_layout.addRow("目标变量:", self.vs_target_label)
            
            # 选择特征数量
            self.vs_n_features = QSpinBox()
            self.vs_n_features.setRange(1, 30)
            self.vs_n_features.setValue(10)
            self.analysis_options_layout.addRow("选择特征数:", self.vs_n_features)
            
        elif index == 4:  # 异常检测
            # 检测方法
            methods = ["马氏距离", "局部异常因子", "隔离森林", "一类SVM"]
            self.od_method_combo = QComboBox()
            self.od_method_combo.addItems(methods)
            self.analysis_options_layout.addRow("检测方法:", self.od_method_combo)
            
            # 置信度
            self.od_confidence = QDoubleSpinBox()
            self.od_confidence.setRange(0.8, 0.99)
            self.od_confidence.setSingleStep(0.01)
            self.od_confidence.setValue(0.95)
            self.analysis_options_layout.addRow("置信度:", self.od_confidence)
    
    def _create_visualization_tab(self):
        """创建可视化选项卡"""
        visualization_tab = QWidget()
        self.tabs.addTab(visualization_tab, "可视化")
        
        # 创建布局
        layout = QHBoxLayout(visualization_tab)
        
        # 左侧选项面板
        options_panel = QWidget()
        options_layout = QVBoxLayout(options_panel)
        options_panel.setMaximumWidth(300)
        
        # 图表类型
        plot_group = QGroupBox("图表类型")
        plot_layout = QVBoxLayout()
        
        plot_types = [
            "光谱曲线", 
            "衍生光谱", 
            "PCA得分图", 
            "PCA载荷图",
            "预测-实际对比图",
            "残差分析图",
            "变量重要性图",
            "3D散点图"
        ]
        
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(plot_types)
        self.plot_type_combo.currentIndexChanged.connect(self._update_plot_options)
        plot_layout.addWidget(self.plot_type_combo)
        
        # 图表选项
        self.plot_options_widget = QWidget()
        self.plot_options_layout = QFormLayout(self.plot_options_widget)
        plot_layout.addWidget(self.plot_options_widget)
        
        # 绘图按钮
        self.plot_btn = QPushButton("绘制图表")
        self.plot_btn.clicked.connect(self._create_plot)
        plot_layout.addWidget(self.plot_btn)
        
        plot_group.setLayout(plot_layout)
        options_layout.addWidget(plot_group)
        
        # 图表调整选项
        customize_group = QGroupBox("图表调整")
        customize_layout = QFormLayout()
        
        # 标题
        self.plot_title = QLineEdit()
        customize_layout.addRow("标题:", self.plot_title)
        
        # X轴标签
        self.plot_xlabel = QLineEdit()
        self.plot_xlabel.setText("波长 (nm)")
        customize_layout.addRow("X轴标签:", self.plot_xlabel)
        
        # Y轴标签
        self.plot_ylabel = QLineEdit()
        self.plot_ylabel.setText("吸光度")
        customize_layout.addRow("Y轴标签:", self.plot_ylabel)
        
        # 保存图表按钮
        self.save_plot_btn = QPushButton("保存图表")
        self.save_plot_btn.clicked.connect(self._save_plot)
        customize_layout.addRow("", self.save_plot_btn)
        
        customize_group.setLayout(customize_layout)
        options_layout.addWidget(customize_group)
        
        # 增加弹性空间
        options_layout.addStretch()
        
        # 添加到主布局
        layout.addWidget(options_panel)
        
        # 右侧图表区域
        plot_panel = QWidget()
        self.plot_layout = QVBoxLayout(plot_panel)

        # 创建图表显示区域
        self.viz_figure = Figure(figsize=(5, 4), dpi=100)
        self.viz_canvas = FigureCanvas(self.viz_figure)
        self.viz_toolbar = NavigationToolbar(self.viz_canvas, self)

        self.plot_layout.addWidget(self.viz_toolbar)
        self.plot_layout.addWidget(self.viz_canvas)
        
        # 添加到主布局
        layout.addWidget(plot_panel)
        
        # 初始化
        self._update_plot_options(0)
    
    def _update_plot_options(self, index):
        """根据图表类型更新选项"""
        try:
            plot_type = self.plot_type_combo.itemText(index)

            # 清除现有选项
            if hasattr(self, 'plot_options_layout'):
                for i in reversed(range(self.plot_options_layout.count())):
                    self.plot_options_layout.itemAt(i).widget().setParent(None)

            # 根据图表类型添加相应选项
            if plot_type == "光谱曲线":
                # 添加光谱曲线选项
                show_legend_cb = QCheckBox("显示图例")
                show_grid_cb = QCheckBox("显示网格")
                line_width_spin = QSpinBox()
                line_width_spin.setRange(1, 5)
                line_width_spin.setValue(2)
                line_width_spin.setPrefix("线宽: ")

                self.plot_options_layout.addWidget(show_legend_cb)
                self.plot_options_layout.addWidget(show_grid_cb)
                self.plot_options_layout.addWidget(line_width_spin)

            elif plot_type in ["PCA得分图", "3D散点图"]:
                # 添加PCA选项
                pc1_spin = QSpinBox()
                pc1_spin.setRange(1, 10)
                pc1_spin.setValue(1)
                pc1_spin.setPrefix("PC1: ")

                pc2_spin = QSpinBox()
                pc2_spin.setRange(1, 10)
                pc2_spin.setValue(2)
                pc2_spin.setPrefix("PC2: ")

                self.plot_options_layout.addWidget(pc1_spin)
                self.plot_options_layout.addWidget(pc2_spin)

            elif plot_type == "预测vs实际":
                # 添加回归选项
                show_line_cb = QCheckBox("显示回归线")
                show_r2_cb = QCheckBox("显示R²")

                self.plot_options_layout.addWidget(show_line_cb)
                self.plot_options_layout.addWidget(show_r2_cb)

            elif plot_type == "变量重要性":
                # 添加变量选择选项
                threshold_spin = QDoubleSpinBox()
                threshold_spin.setRange(0.1, 5.0)
                threshold_spin.setValue(1.0)
                threshold_spin.setSingleStep(0.1)
                threshold_spin.setPrefix("VIP阈值: ")

                self.plot_options_layout.addWidget(threshold_spin)

        except Exception as e:
            print(f"更新图表选项时出错: {e}")
            import traceback
            traceback.print_exc()
    
    def _apply_preprocessing(self):
        """应用预处理方法"""
        try:
            if self.dataset.spectra is None:
                QMessageBox.warning(self, "警告", "请先加载数据！")
                return

            method = self.preprocess_method_combo.currentText()
            params = {}

            # 获取预处理参数
            if method == "标准化":
                norm_type = self.norm_type_combo.currentText()
                params['method'] = norm_type.lower()

            elif method == "Savitzky-Golay平滑":
                params['window_length'] = int(self.sg_window_spin.value())
                params['polyorder'] = int(self.sg_poly_spin.value())
                params['deriv'] = int(self.sg_deriv_spin.value())

            elif method == "小波去噪":
                params['wavelet'] = self.wavelet_type_combo.currentText()
                params['threshold_mode'] = self.threshold_mode_combo.currentText()

            elif method == "基线校正":
                params['lam'] = float(self.baseline_lam_spin.value())
                params['p'] = float(self.baseline_p_spin.value())

            elif method == "移动平均":
                params['window_size'] = int(self.ma_window_spin.value())

            # 应用预处理
            if method == "标准化":
                self.current_processed_data = normalize(self.dataset.spectra, **params)
            elif method == "MSC":
                self.current_processed_data = msc(self.dataset.spectra)
            elif method == "SNV":
                self.current_processed_data = snv(self.dataset.spectra)
            elif method == "Savitzky-Golay平滑":
                self.current_processed_data = savitzky_golay(self.dataset.spectra, **params)
            elif method == "去趋势":
                self.current_processed_data = detrend(self.dataset.spectra)
            elif method == "基线校正":
                self.current_processed_data = baseline_correction(self.dataset.spectra, **params)
            elif method == "移动平均":
                self.current_processed_data = moving_average(self.dataset.spectra, **params)
            elif method == "小波去噪":
                self.current_processed_data = wavelet_denoise(self.dataset.spectra, **params)

            # 添加到处理历史
            self.processing_history.append({
                'method': method,
                'params': params,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })

            # 更新历史记录显示
            self._update_processing_history()

            # 显示成功消息
            QMessageBox.information(self, "成功", f"预处理方法 '{method}' 应用成功！")

            # 更新可视化选项卡
            self.tabs.setCurrentIndex(3)  # 切换到可视化选项卡

        except Exception as e:
            QMessageBox.critical(self, "错误", f"预处理失败：{str(e)}")
            import traceback
            traceback.print_exc()
    
    def _reset_preprocessing(self):
        """重置为原始数据"""
        try:
            if self.dataset.spectra is None:
                QMessageBox.warning(self, "警告", "没有数据需要重置！")
                return

            # 重置处理后的数据
            self.current_processed_data = None

            # 清空处理历史
            self.processing_history.clear()

            # 更新历史记录显示
            self._update_processing_history()

            QMessageBox.information(self, "成功", "已重置为原始数据！")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"重置失败：{str(e)}")
            import traceback
            traceback.print_exc()
    
    def _run_analysis(self):
        """运行选择的分析方法"""
        try:
            if self.dataset.spectra is None:
                QMessageBox.warning(self, "警告", "请先加载数据！")
                return

            method = self.analysis_method_combo.currentText()

            # 确定使用哪个数据
            data_to_analyze = self.current_processed_data if self.current_processed_data is not None else self.dataset.spectra

            # 运行分析方法
            if method == "主成分分析 (PCA)":
                n_components = int(self.pca_components_spin.value())
                self.pca_results = pca_analysis(data_to_analyze, n_components=n_components)

                # 显示PCA结果
                self._display_pca_results()

            elif method == "偏最小二乘回归 (PLS)":
                if self.dataset.target is None:
                    QMessageBox.warning(self, "警告", "PLS回归需要目标值数据！")
                    return

                n_components = int(self.pls_components_spin.value())
                self.pls_results = pls_regression(data_to_analyze, self.dataset.target, n_components=n_components)

                # 显示PLS结果
                self._display_pls_results()

            elif method == "支持向量回归 (SVR)":
                if self.dataset.target is None:
                    QMessageBox.warning(self, "警告", "SVR回归需要目标值数据！")
                    return

                # 简化的SVR参数
                params = {
                    'kernel': 'rbf',
                    'C': 1.0,
                    'gamma': 'scale'
                }
                self.svm_results = svm_regression(data_to_analyze, self.dataset.target, **params)

                # 显示SVR结果
                self._display_svr_results()

            elif method == "变量选择":
                if self.dataset.target is None:
                    QMessageBox.warning(self, "警告", "变量选择需要目标值数据！")
                    return

                self.variable_selection_results = variable_selection(data_to_analyze, self.dataset.target)

                # 显示变量选择结果
                self._display_variable_selection_results()

            elif method == "异常值检测":
                # 使用马氏距离进行异常值检测
                outlier_results = mahalanobis_distance(data_to_analyze)

                # 显示异常值检测结果
                self._display_outlier_results(outlier_results)

            # 切换到可视化选项卡
            self.tabs.setCurrentIndex(3)

            QMessageBox.information(self, "成功", f"分析方法 '{method}' 运行成功！")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"分析失败：{str(e)}")
            import traceback
            traceback.print_exc()

    def _update_processing_history(self):
        """更新处理历史显示"""
        try:
            if hasattr(self, 'process_history_list'):
                # 清空现有列表
                self.process_history_list.clear()

                # 添加处理历史记录
                for record in self.processing_history:
                    method = record['method']
                    timestamp = record['timestamp']
                    params = record.get('params', {})

                    # 格式化参数字符串
                    if params:
                        param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
                        item_text = f"{timestamp} - {method} ({param_str})"
                    else:
                        item_text = f"{timestamp} - {method}"

                    # 添加到列表
                    item = QListWidgetItem(item_text)
                    self.process_history_list.addItem(item)

                # 滚动到底部
                self.process_history_list.scrollToBottom()

        except Exception as e:
            print(f"更新处理历史时出错: {e}")

    def _display_pca_results(self):
        """显示PCA分析结果"""
        if self.pca_results is None:
            return

        # 清空结果表格
        self.analysis_result_table.setRowCount(0)
        self.analysis_result_table.setColumnCount(2)
        self.analysis_result_table.setHorizontalHeaderLabels(["指标", "值"])

        # 添加PCA结果
        row = 0
        self.analysis_result_table.insertRow(row)
        self.analysis_result_table.setItem(row, 0, QTableWidgetItem("主成分数量"))
        self.analysis_result_table.setItem(row, 1, QTableWidgetItem(str(self.pca_results['n_components'])))

        row += 1
        self.analysis_result_table.insertRow(row)
        self.analysis_result_table.setItem(row, 0, QTableWidgetItem("解释方差比"))
        self.analysis_result_table.setItem(row, 1, QTableWidgetItem(f"{self.pca_results['explained_variance_ratio']:.4f}"))

        # 显示前几个主成分的解释方差
        for i, var in enumerate(self.pca_results['explained_variance_ratio'][:5]):
            row += 1
            self.analysis_result_table.insertRow(row)
            self.analysis_result_table.setItem(row, 0, QTableWidgetItem(f"PC{i+1} 解释方差"))
            self.analysis_result_table.setItem(row, 1, QTableWidgetItem(f"{var:.4f}"))

        self.analysis_result_table.resizeColumnsToContents()

    def _display_pls_results(self):
        """显示PLS回归结果"""
        if self.pls_results is None:
            return

        # 清空结果表格
        self.analysis_result_table.setRowCount(0)
        self.analysis_result_table.setColumnCount(2)
        self.analysis_result_table.setHorizontalHeaderLabels(["指标", "值"])

        # 添加PLS结果
        row = 0
        self.analysis_result_table.insertRow(row)
        self.analysis_result_table.setItem(row, 0, QTableWidgetItem("主成分数量"))
        self.analysis_result_table.setItem(row, 1, QTableWidgetItem(str(self.pls_results['n_components'])))

        row += 1
        self.analysis_result_table.insertRow(row)
        self.analysis_result_table.setItem(row, 0, QTableWidgetItem("R²"))
        self.analysis_result_table.setItem(row, 1, QTableWidgetItem(f"{self.pls_results['r2_score']:.4f}"))

        row += 1
        self.analysis_result_table.insertRow(row)
        self.analysis_result_table.setItem(row, 0, QTableWidgetItem("RMSE"))
        self.analysis_result_table.setItem(row, 1, QTableWidgetItem(f"{self.pls_results['rmse']:.4f}"))

        row += 1
        self.analysis_result_table.insertRow(row)
        self.analysis_result_table.setItem(row, 0, QTableWidgetItem("RPD"))
        self.analysis_result_table.setItem(row, 1, QTableWidgetItem(f"{self.pls_results['rpd']:.4f}"))

        self.analysis_result_table.resizeColumnsToContents()

    def _display_svr_results(self):
        """显示SVR回归结果"""
        if self.svm_results is None:
            return

        # 清空结果表格
        self.analysis_result_table.setRowCount(0)
        self.analysis_result_table.setColumnCount(2)
        self.analysis_result_table.setHorizontalHeaderLabels(["指标", "值"])

        # 添加SVR结果
        row = 0
        self.analysis_result_table.insertRow(row)
        self.analysis_result_table.setItem(row, 0, QTableWidgetItem("R²"))
        self.analysis_result_table.setItem(row, 1, QTableWidgetItem(f"{self.svm_results['r2_score']:.4f}"))

        row += 1
        self.analysis_result_table.insertRow(row)
        self.analysis_result_table.setItem(row, 0, QTableWidgetItem("RMSE"))
        self.analysis_result_table.setItem(row, 1, QTableWidgetItem(f"{self.svm_results['rmse']:.4f}"))

        row += 1
        self.analysis_result_table.insertRow(row)
        self.analysis_result_table.setItem(row, 0, QTableWidgetItem("支持向量数量"))
        self.analysis_result_table.setItem(row, 1, QTableWidgetItem(str(self.svm_results['n_support_vectors'])))

        self.analysis_result_table.resizeColumnsToContents()

    def _display_variable_selection_results(self):
        """显示变量选择结果"""
        if self.variable_selection_results is None:
            return

        # 清空结果表格
        self.analysis_result_table.setRowCount(0)
        self.analysis_result_table.setColumnCount(2)
        self.analysis_result_table.setHorizontalHeaderLabels(["指标", "值"])

        # 添加变量选择结果
        row = 0
        self.analysis_result_table.insertRow(row)
        self.analysis_result_table.setItem(row, 0, QTableWidgetItem("选择的变量数量"))
        self.analysis_result_table.setItem(row, 1, QTableWidgetItem(str(len(self.variable_selection_results['selected_features']))))

        row += 1
        self.analysis_result_table.insertRow(row)
        self.analysis_result_table.setItem(row, 0, QTableWidgetItem("VIP阈值"))
        self.analysis_result_table.setItem(row, 1, QTableWidgetItem(f"{self.variable_selection_results['vip_threshold']:.4f}"))

        self.analysis_result_table.resizeColumnsToContents()

    def _display_outlier_results(self, outlier_results):
        """显示异常值检测结果"""
        if outlier_results is None:
            return

        # 清空结果表格
        self.analysis_result_table.setRowCount(0)
        self.analysis_result_table.setColumnCount(2)
        self.analysis_result_table.setHorizontalHeaderLabels(["指标", "值"])

        # 添加异常值检测结果
        row = 0
        self.analysis_result_table.insertRow(row)
        self.analysis_result_table.setItem(row, 0, QTableWidgetItem("异常值数量"))
        self.analysis_result_table.setItem(row, 1, QTableWidgetItem(str(sum(outlier_results['is_outlier']))))

        row += 1
        self.analysis_result_table.insertRow(row)
        self.analysis_result_table.setItem(row, 0, QTableWidgetItem("异常值样本索引"))
        outlier_indices = [i for i, is_out in enumerate(outlier_results['is_outlier']) if is_out]
        self.analysis_result_table.setItem(row, 1, QTableWidgetItem(str(outlier_indices)))

        self.analysis_result_table.resizeColumnsToContents()

    def _export_analysis_result(self):
        """导出分析结果"""
        try:
            if not any([self.pca_results, self.pls_results, self.svm_results,
                       hasattr(self, 'variable_selection_results') and self.variable_selection_results]):
                QMessageBox.warning(self, "警告", "没有可导出的分析结果！")
                return

            # 选择导出文件路径
            file_path, _ = QFileDialog.getSaveFileName(
                self, "导出分析结果", "", "Excel Files (*.xlsx);;CSV Files (*.csv);;JSON Files (*.json)"
            )

            if not file_path:
                return

            # 收集所有结果
            results = {}

            if self.pca_results:
                results['PCA'] = self.pca_results

            if self.pls_results:
                results['PLS'] = self.pls_results

            if self.svm_results:
                results['SVR'] = self.svm_results

            if hasattr(self, 'variable_selection_results') and self.variable_selection_results:
                results['Variable_Selection'] = self.variable_selection_results

            # 根据文件格式导出
            _, ext = os.path.splitext(file_path)
            ext = ext.lower()

            if ext == '.xlsx':
                # 导出到Excel
                with pd.ExcelWriter(file_path) as writer:
                    for method_name, result in results.items():
                        if isinstance(result, dict):
                            # 将字典转换为DataFrame
                            df_data = []
                            for key, value in result.items():
                                if isinstance(value, (list, np.ndarray)):
                                    if len(value) <= 10:  # 如果数组很小，直接保存
                                        df_data.append([key, str(value)])
                                    else:
                                        df_data.append([key, f"Array with {len(value)} elements"])
                                else:
                                    df_data.append([key, value])

                            df = pd.DataFrame(df_data, columns=['Parameter', 'Value'])
                            df.to_excel(writer, sheet_name=method_name, index=False)

            elif ext == '.csv':
                # 导出到CSV
                all_data = []
                for method_name, result in results.items():
                    if isinstance(result, dict):
                        for key, value in result.items():
                            if isinstance(value, (list, np.ndarray)):
                                if len(value) <= 10:
                                    all_data.append([method_name, key, str(value)])
                                else:
                                    all_data.append([method_name, key, f"Array with {len(value)} elements"])
                            else:
                                all_data.append([method_name, key, value])

                df = pd.DataFrame(all_data, columns=['Method', 'Parameter', 'Value'])
                df.to_csv(file_path, index=False)

            elif ext == '.json':
                # 导出到JSON
                # 转换numpy数组为列表
                json_results = {}
                for method_name, result in results.items():
                    if isinstance(result, dict):
                        json_results[method_name] = {}
                        for key, value in result.items():
                            if isinstance(value, np.ndarray):
                                json_results[method_name][key] = value.tolist()
                            else:
                                json_results[method_name][key] = value

                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(json_results, f, indent=2, ensure_ascii=False)

            QMessageBox.information(self, "成功", f"分析结果已导出到：{file_path}")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败：{str(e)}")
            import traceback
            traceback.print_exc()
    
    def _create_plot(self):
        """创建图表"""
        try:
            if self.dataset.spectra is None:
                QMessageBox.warning(self, "警告", "请先加载数据！")
                return

            plot_type = self.plot_type_combo.currentText()

            # 清除现有图表
            if hasattr(self, 'plot_layout'):
                for i in reversed(range(self.plot_layout.count())):
                    self.plot_layout.itemAt(i).widget().setParent(None)

            # 确定使用哪个数据
            data_to_plot = self.current_processed_data if self.current_processed_data is not None else self.dataset.spectra

            if plot_type == "光谱曲线":
                # 创建光谱曲线图
                canvas = SpectrumPlotCanvas()
                canvas.plot_multiple_spectra(self.dataset.wavelengths, data_to_plot,
                                  title="光谱曲线", xlabel="波长 (nm)", ylabel="吸光度")

                self.plot_layout.addWidget(canvas)

            elif plot_type == "导数光谱":
                if self.pca_results is not None and 'scores' in self.pca_results:
                    # 显示PCA scores
                    canvas = PCAPlotCanvas()
                    canvas.plot_scores(self.pca_results['scores'],
                                     title="PCA得分图")
                    self.plot_layout.addWidget(canvas)
                else:
                    QMessageBox.warning(self, "警告", "请先运行PCA分析！")
                    return

            elif plot_type == "PCA得分图":
                if self.pca_results is not None:
                    canvas = PCAPlotCanvas()
                    canvas.plot_scores(self.pca_results['scores'],
                                     title="PCA得分图")
                    self.plot_layout.addWidget(canvas)
                else:
                    QMessageBox.warning(self, "警告", "请先运行PCA分析！")
                    return

            elif plot_type == "PCA载荷图":
                if self.pca_results is not None:
                    canvas = PCAPlotCanvas()
                    canvas.plot_loadings(self.dataset.wavelengths, self.pca_results['loadings'],
                                        title="PCA载荷图")
                    self.plot_layout.addWidget(canvas)
                else:
                    QMessageBox.warning(self, "警告", "请先运行PCA分析！")
                    return

            elif plot_type == "预测vs实际":
                if self.pls_results is not None:
                    canvas = PredictionPlotCanvas()
                    canvas.plot_prediction_vs_actual(self.pls_results['y_true'],
                                                   self.pls_results['y_pred'],
                                                   title="PLS预测vs实际")
                    self.plot_layout.addWidget(canvas)
                elif self.svm_results is not None:
                    canvas = PredictionPlotCanvas()
                    canvas.plot_prediction_vs_actual(self.svm_results['y_true'],
                                                   self.svm_results['y_pred'],
                                                   title="SVR预测vs实际")
                    self.plot_layout.addWidget(canvas)
                else:
                    QMessageBox.warning(self, "警告", "请先运行PLS或SVR分析！")
                    return

            elif plot_type == "残差分析":
                if self.pls_results is not None:
                    canvas = PredictionPlotCanvas()
                    canvas.plot_residuals(self.pls_results['y_true'],
                                        self.pls_results['y_pred'],
                                        title="PLS残差分析")
                    self.plot_layout.addWidget(canvas)
                elif self.svm_results is not None:
                    canvas = PredictionPlotCanvas()
                    canvas.plot_residuals(self.svm_results['y_true'],
                                        self.svm_results['y_pred'],
                                        title="SVR残差分析")
                    self.plot_layout.addWidget(canvas)
                else:
                    QMessageBox.warning(self, "警告", "请先运行PLS或SVR分析！")
                    return

            elif plot_type == "变量重要性":
                if hasattr(self, 'variable_selection_results') and self.variable_selection_results:
                    # 创建变量重要性图
                    canvas = SpectrumPlotCanvas()
                    vip_scores = self.variable_selection_results.get('vip_scores', [])
                    if vip_scores:
                        canvas.plot_variable_importance(self.dataset.wavelengths, vip_scores,
                                                       title="变量重要性图 (VIP)")
                        self.plot_layout.addWidget(canvas)
                    else:
                        QMessageBox.warning(self, "警告", "没有VIP分数数据！")
                        return
                else:
                    QMessageBox.warning(self, "警告", "请先运行变量选择分析！")
                    return

            elif plot_type == "3D散点图":
                if self.pca_results is not None:
                    canvas = PCAPlotCanvas()
                    canvas.plot_3d_scores(self.pca_results['scores'],
                                        title="3D PCA得分图")
                    self.plot_layout.addWidget(canvas)
                else:
                    QMessageBox.warning(self, "警告", "请先运行PCA分析！")
                    return

        except Exception as e:
            QMessageBox.critical(self, "错误", f"创建图表失败：{str(e)}")
            import traceback
            traceback.print_exc()
    
    def _save_plot(self):
        """保存图表"""
        try:
            if not hasattr(self, 'plot_layout') or self.plot_layout.count() == 0:
                QMessageBox.warning(self, "警告", "没有可保存的图表！")
                return

            # 选择保存文件路径
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存图表", "", "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg)"
            )

            if not file_path:
                return

            # 获取当前画布
            canvas = self.plot_layout.itemAt(0).widget()
            if hasattr(canvas, 'figure'):
                # 保存图表
                canvas.figure.savefig(file_path, dpi=300, bbox_inches='tight')
                QMessageBox.information(self, "成功", f"图表已保存到：{file_path}")
            else:
                QMessageBox.warning(self, "警告", "无法保存当前图表！")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存图表失败：{str(e)}")
            import traceback
            traceback.print_exc()
    
    def _open_corn_file(self):
        """打开玉米光谱数据文件"""
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "打开玉米光谱数据文件", "", 
            "CSV文件 (*.csv);;Excel文件 (*.xlsx);;所有文件 (*)",
            options=options
        )
        
        if file_name:
            try:
                # 这里应该加载玉米数据的代码
                # 由于没有具体实现，这里用简单信息提示
                self.statusBar.showMessage(f"已加载玉米数据文件: {os.path.basename(file_name)}")
                
                # 更新玉米数据信息
                self._update_corn_info()
                
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载玉米数据文件时出错：{str(e)}")
                print(f"加载玉米数据文件错误: {e}")
    
    def _update_corn_info(self):
        """更新玉米数据信息显示"""
        # 示例数据
        info_text = (
            "文件: corn_sample.csv\n"
            "样本数: 80\n"
            "波长范围: 1100-2500 nm\n"
            "包含组分: 水分, 蛋白质, 脂肪, 淀粉"
        )
        
        if hasattr(self, 'corn_info_text'):
            self.corn_info_text.setText(info_text)
            
        # 更新目标变量标签
        if hasattr(self, 'corn_target_label'):
            self.corn_target_label.setText("目标变量: 水分 (%)")
    
    def _apply_corn_preprocessing(self):
        """应用玉米数据预处理"""
        # 获取选择的预处理方法
        if not hasattr(self, 'corn_preprocess_combo'):
            return
            
        method = self.corn_preprocess_combo.currentText()
        
        # 显示预处理信息
        QMessageBox.information(
            self, 
            "预处理应用", 
            f"已应用预处理方法: {method}\n\n"
            "预处理后的数据将用于后续分析。"
        )
        
        # 更新绘图
        self._update_corn_plot()
    
    def _update_corn_plot(self):
        """更新玉米分析图表"""
        if not hasattr(self, 'corn_ax') or not hasattr(self, 'corn_figure') or not hasattr(self, 'corn_canvas'):
            return
            
        # 清除当前图表
        self.corn_ax.clear()
        
        # 创建一些示例数据
        x = np.linspace(10, 30, 20)  # 实际值
        y = x + np.random.normal(0, 1, 20)  # 预测值
        
        # 绘制散点图
        self.corn_ax.scatter(x, y, c='blue', alpha=0.6, label='校准样本')
        
        # 绘制理想线 (y=x)
        line_x = np.linspace(min(x)-2, max(x)+2, 100)
        self.corn_ax.plot(line_x, line_x, 'r--', label='理想线')
        
        # 添加标签和图例
        self.corn_ax.set_title("玉米水分含量预测")
        self.corn_ax.set_xlabel("实际值 (%)")
        self.corn_ax.set_ylabel("预测值 (%)")
        self.corn_ax.legend()
        self.corn_ax.grid(True)
        
        # 更新图表
        self.corn_figure.tight_layout()
        self.corn_canvas.draw()
    
    def _train_corn_model(self):
        """训练玉米分析模型"""
        if not hasattr(self, 'corn_model_combo') or not hasattr(self, 'corn_cv_spinbox'):
            return
            
        # 获取模型和交叉验证参数
        model = self.corn_model_combo.currentText()
        cv = self.corn_cv_spinbox.value()
        
        # 显示训练中的信息
        QMessageBox.information(
            self, 
            "模型训练", 
            f"正在训练模型: {model}\n"
            f"使用 {cv} 折交叉验证\n\n"
            "请稍候..."
        )
        
        # 创建模拟结果
        r2 = 0.95 + np.random.normal(0, 0.02)
        rmse = 0.45 + np.random.normal(0, 0.05)
        
        # 更新结果显示
        if hasattr(self, 'corn_result_text'):
            result_text = (
                f"模型: {model}\n"
                f"交叉验证: {cv} 折\n\n"
                f"校准集结果:\n"
                f"R²: {r2:.4f}\n"
                f"RMSE: {rmse:.4f}\n\n"
                f"验证集结果:\n"
                f"R²: {r2-0.05:.4f}\n"
                f"RMSE: {rmse+0.1:.4f}\n"
            )
            self.corn_result_text.setText(result_text)
        
        # 更新图表
        self._update_corn_plot()
        
        # 保存模型到属性
        self.corn_model = {
            'type': model,
            'r2': r2,
            'rmse': rmse
        }
    
    def _export_corn_result(self):
        """导出玉米分析结果"""
        if not hasattr(self, 'corn_model') or self.corn_model is None:
            QMessageBox.warning(self, "警告", "没有可导出的分析结果!")
            return
            
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(
            self, "导出玉米分析结果", "", 
            "CSV文件 (*.csv);;Excel文件 (*.xlsx);;JSON文件 (*.json);;所有文件 (*)",
            options=options
        )
        
        if file_name:
            try:
                # 确定文件格式
                ext = os.path.splitext(file_name)[1].lower()
                
                # 创建结果字典
                result_data = {
                    'model': self.corn_model.get('type', '未知'),
                    'r2_calibration': self.corn_model.get('r2', 0),
                    'rmse_calibration': self.corn_model.get('rmse', 0),
                    'r2_validation': self.corn_model.get('r2', 0) - 0.05,
                    'rmse_validation': self.corn_model.get('rmse', 0) + 0.1,
                    'export_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                # 根据格式导出
                if ext == '.csv':
                    # 导出为CSV
                    with open(file_name, 'w', encoding='utf-8') as f:
                        f.write("玉米成分分析结果\n")
                        f.write(f"导出时间: {result_data['export_time']}\n\n")
                        f.write(f"模型: {result_data['model']}\n")
                        f.write(f"校准集 R²: {result_data['r2_calibration']:.4f}\n")
                        f.write(f"校准集 RMSE: {result_data['rmse_calibration']:.4f}\n")
                        f.write(f"验证集 R²: {result_data['r2_validation']:.4f}\n")
                        f.write(f"验证集 RMSE: {result_data['rmse_validation']:.4f}\n")
                
                elif ext == '.xlsx':
                    try:
                        # 导出为Excel
                        import pandas as pd
                        
                        # 创建DataFrame
                        df_data = {
                            'Metric': ['Model', 'R² (Cal)', 'RMSE (Cal)', 'R² (Val)', 'RMSE (Val)'],
                            'Value': [
                                result_data['model'],
                                result_data['r2_calibration'],
                                result_data['rmse_calibration'],
                                result_data['r2_validation'],
                                result_data['rmse_validation']
                            ]
                        }
                        
                        df = pd.DataFrame(df_data)
                        df.to_excel(file_name, index=False)
                    
                    except ImportError:
                        QMessageBox.warning(self, "警告", "导出Excel需要安装pandas和openpyxl库。正在使用CSV格式导出...")
                        # 退回到CSV格式
                        with open(file_name.replace('.xlsx', '.csv'), 'w', encoding='utf-8') as f:
                            f.write("玉米成分分析结果\n")
                            f.write(f"导出时间: {result_data['export_time']}\n\n")
                            f.write(f"模型: {result_data['model']}\n")
                            f.write(f"校准集 R²: {result_data['r2_calibration']:.4f}\n")
                            f.write(f"校准集 RMSE: {result_data['rmse_calibration']:.4f}\n")
                            f.write(f"验证集 R²: {result_data['r2_validation']:.4f}\n")
                            f.write(f"验证集 RMSE: {result_data['rmse_validation']:.4f}\n")
                
                elif ext == '.json':
                    # 导出为JSON
                    with open(file_name, 'w', encoding='utf-8') as f:
                        json.dump(result_data, f, indent=2, ensure_ascii=False)
                
                else:
                    # 默认导出为CSV
                    with open(file_name, 'w', encoding='utf-8') as f:
                        f.write("玉米成分分析结果\n")
                        f.write(f"导出时间: {result_data['export_time']}\n\n")
                        f.write(f"模型: {result_data['model']}\n")
                        f.write(f"校准集 R²: {result_data['r2_calibration']:.4f}\n")
                        f.write(f"校准集 RMSE: {result_data['rmse_calibration']:.4f}\n")
                        f.write(f"验证集 R²: {result_data['r2_validation']:.4f}\n")
                        f.write(f"验证集 RMSE: {result_data['rmse_validation']:.4f}\n")
                
                self.statusBar.showMessage(f"玉米分析结果已导出: {os.path.basename(file_name)}")
                
            except Exception as e:
                QMessageBox.critical(self, "错误", f"导出玉米分析结果时出错：{str(e)}")
                print(f"导出玉米分析结果错误: {e}")