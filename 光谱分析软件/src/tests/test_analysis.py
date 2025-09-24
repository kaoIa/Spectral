#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
光谱分析模块测试
"""

import sys
import os
import unittest
import numpy as np
from pathlib import Path
from sklearn.datasets import make_regression

# 添加项目根目录到路径
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

from src.core.analysis import (pca_analysis, pls_regression, svm_regression,
                             variable_selection, mahalanobis_distance)


class TestAnalysisMethods(unittest.TestCase):
    """测试光谱分析方法"""
    
    def setUp(self):
        """测试前准备工作"""
        # 创建合成数据
        np.random.seed(42)  # 固定随机种子，使测试结果可重复
        
        # 光谱数据维度
        n_samples = 40
        n_features = 200
        
        # 生成用于分类/回归的合成数据
        self.X, self.y = make_regression(n_samples=n_samples, n_features=n_features,
                                         n_informative=10, noise=0.5, random_state=42)
        
        # 归一化特征
        self.X = (self.X - np.mean(self.X, axis=0)) / np.std(self.X, axis=0)
        
        # 添加一些典型的光谱相关性 (相邻变量的相关性)
        for i in range(1, n_features):
            self.X[:, i] = 0.8 * self.X[:, i] + 0.2 * self.X[:, i-1]
    
    def test_pca_analysis(self):
        """测试PCA分析"""
        # 执行PCA分析
        scores, loadings, explained_variance_ratio, model = pca_analysis(self.X, n_components=5)
        
        # 检查输出维度
        self.assertEqual(scores.shape, (self.X.shape[0], 5))
        self.assertEqual(loadings.shape, (5, self.X.shape[1]))
        self.assertEqual(len(explained_variance_ratio), 5)
        
        # 验证解释方差之和接近1
        self.assertLess(abs(np.sum(explained_variance_ratio) - 1.0), 0.1)
        
        # 检查重构误差
        reconstructed = np.dot(scores, loadings)
        reconstruction_error = np.mean((self.X - reconstructed) ** 2)
        self.assertLess(reconstruction_error, 0.5)  # 重构误差应该较小
    
    def test_pls_regression(self):
        """测试PLS回归"""
        # 执行PLS回归
        model, predictions, mse, r2, optimum_components = pls_regression(self.X, self.y, 
                                                                     n_components=10, cv=5)
        
        # 检查输出
        self.assertEqual(len(predictions), len(self.y))
        self.assertGreater(r2, 0)  # R2应该为正
        self.assertLess(mse, np.var(self.y))  # MSE应该小于目标变量的方差
        
        # 最优组分数应该是一个有效数字
        self.assertGreaterEqual(optimum_components, 1)
        self.assertLessEqual(optimum_components, 10)
        
        # 测试模型预测
        y_pred = model.predict(self.X)
        self.assertEqual(len(y_pred), len(self.y))
    
    def test_svm_regression(self):
        """测试SVM回归"""
        # 执行SVM回归
        model, predictions, mse, r2 = svm_regression(self.X, self.y, kernel='rbf', 
                                                 C=1.0, epsilon=0.1, gamma='scale', cv=5)
        
        # 检查输出
        self.assertEqual(len(predictions), len(self.y))
        self.assertGreater(r2, 0)  # R2应该为正
        self.assertLess(mse, np.var(self.y))  # MSE应该小于目标变量的方差
        
        # 测试模型预测
        y_pred = model.predict(self.X)
        self.assertEqual(len(y_pred), len(self.y))
    
    def test_variable_selection(self):
        """测试变量选择"""
        # 使用PLS方法选择变量
        selected_indices, importance_scores = variable_selection(self.X, self.y, 
                                                             method='pls', threshold=0.5)
        
        # 检查输出
        self.assertGreater(len(selected_indices), 0)  # 应该选择一些变量
        self.assertLessEqual(len(selected_indices), self.X.shape[1])  # 不应超过总变量数
        self.assertEqual(len(importance_scores), self.X.shape[1])  # 每个变量应有重要性评分
        
        # 使用VIP方法选择变量
        selected_indices, importance_scores = variable_selection(self.X, self.y, 
                                                             method='vip', threshold=10)
        
        # 检查输出
        self.assertEqual(len(selected_indices), 10)  # 应该选择10个变量
        self.assertEqual(len(importance_scores), self.X.shape[1])  # 每个变量应有重要性评分
    
    def test_mahalanobis_distance(self):
        """测试马氏距离计算"""
        # 计算马氏距离
        distances = mahalanobis_distance(self.X)
        
        # 检查输出
        self.assertEqual(len(distances), self.X.shape[0])  # 每个样本应有一个距离值
        self.assertTrue(np.all(distances >= 0))  # 所有距离应为非负
        
        # 测试极端离群值的距离
        outlier = np.ones((1, self.X.shape[1])) * 10  # 创建极端值
        outlier_dist = mahalanobis_distance(outlier, reference=self.X)
        normal_dist = np.median(distances)
        
        # 离群值的距离应该比正常样本的距离大得多
        self.assertGreater(outlier_dist[0], normal_dist * 2)


if __name__ == "__main__":
    unittest.main() 