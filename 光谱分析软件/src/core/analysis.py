#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
光谱分析算法模块
包含常用的光谱分析方法，如PCA、PLS、SVM等
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score


def pca_analysis(X, n_components=None):
    """
    主成分分析 (Principal Component Analysis)
    
    参数:
        X: 输入光谱数据, shape=(n_samples, n_features)
        n_components: 主成分数量，默认为None (自动选择)
        
    返回:
        scores: 主成分得分
        loadings: 主成分载荷
        explained_variance_ratio: 解释方差比例
        model: PCA模型对象
    """
    # 创建PCA模型
    model = PCA(n_components=n_components)
    
    # 数据变换，获得主成分得分
    scores = model.fit_transform(X)
    
    # 主成分载荷
    loadings = model.components_
    
    # 解释方差比例
    explained_variance_ratio = model.explained_variance_ratio_
    
    return scores, loadings, explained_variance_ratio, model


def pls_regression(X, y, n_components=10, cv=10):
    """
    偏最小二乘回归 (Partial Least Squares Regression)
    
    参数:
        X: 输入光谱数据, shape=(n_samples, n_features)
        y: 目标变量
        n_components: 潜变量数量
        cv: 交叉验证折数
        
    返回:
        model: 训练好的PLS模型
        predictions: 交叉验证预测结果
        mse: 均方误差
        r2: 决定系数
        optimum_components: 最优组分数量 (最小MSE对应的组分数)
    """
    # 存储不同组分数量对应的预测性能
    mse_values = []
    r2_values = []
    
    # 存储不同组分数量下的预测值
    predictions_list = []
    
    # 尝试不同的组分数量
    components_range = range(1, min(n_components+1, min(X.shape[0], X.shape[1])+1))
    
    for n_comp in components_range:
        # 创建PLS模型
        model = PLSRegression(n_components=n_comp)
        
        # 交叉验证预测
        y_pred = cross_val_predict(model, X, y, cv=cv)
        
        # 计算性能指标
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        # 保存结果
        mse_values.append(mse)
        r2_values.append(r2)
        predictions_list.append(y_pred)
    
    # 找到最优组分数量 (MSE最小)
    optimum_idx = np.argmin(mse_values)
    optimum_components = components_range[optimum_idx]
    
    # 用最优组分数量训练最终模型
    final_model = PLSRegression(n_components=optimum_components)
    final_model.fit(X, y)
    
    return final_model, predictions_list[optimum_idx], mse_values[optimum_idx], r2_values[optimum_idx], optimum_components


def svm_regression(X, y, kernel='rbf', C=1.0, epsilon=0.1, gamma='scale', cv=10):
    """
    支持向量机回归 (Support Vector Regression)
    
    参数:
        X: 输入光谱数据, shape=(n_samples, n_features)
        y: 目标变量
        kernel: 核函数类型
        C: 正则化参数
        epsilon: SVR的epsilon参数
        gamma: 'rbf', 'poly' 和 'sigmoid' 核函数的参数
        cv: 交叉验证折数
        
    返回:
        model: 训练好的SVM模型
        predictions: 交叉验证预测结果
        mse: 均方误差
        r2: 决定系数
    """
    # 创建SVR模型
    model = SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma)
    
    # 交叉验证预测
    predictions = cross_val_predict(model, X, y, cv=cv)
    
    # 计算性能指标
    mse = mean_squared_error(y, predictions)
    r2 = r2_score(y, predictions)
    
    # 训练最终模型
    model.fit(X, y)
    
    return model, predictions, mse, r2


def variable_selection(X, y, method='pls', threshold=0.5):
    """
    变量选择 (特征选择)
    
    参数:
        X: 输入光谱数据, shape=(n_samples, n_features)
        y: 目标变量
        method: 选择方法 ('pls' 或 'vip')
        threshold: 选择阈值
        
    返回:
        selected_indices: 选中变量的索引
        importance_scores: 变量重要性评分
    """
    if method == 'pls':
        # 使用PLS系数大小进行变量选择
        model = PLSRegression(n_components=min(10, X.shape[0]-1))
        model.fit(X, y)
        
        # 计算特征重要性评分
        importance_scores = np.abs(model.coef_.ravel())
        
    elif method == 'vip':
        # 使用VIP评分进行变量选择
        model = PLSRegression(n_components=min(10, X.shape[0]-1))
        model.fit(X, y)
        
        # 计算VIP分数
        t = model.x_scores_
        w = model.x_weights_
        q = model.y_loadings_
        
        # 计算每个变量在所有组分上的权重平方和
        p = np.sum(np.square(w), axis=1)
        
        # 计算每个组分解释的Y方差比例
        explained_var = np.square(q).sum(axis=0) / np.square(q).sum()
        
        # 变量在各个组分上的投影
        s = np.square(w) * explained_var
        
        # VIP评分
        importance_scores = np.sqrt(p * s.sum(axis=1) * X.shape[1] / s.sum())
        
    else:
        raise ValueError(f"不支持的变量选择方法: {method}")
    
    # 根据阈值选择变量
    # 如果threshold < 1，则解释为选择重要性大于该百分比的变量
    if threshold < 1:
        cutoff = threshold * np.max(importance_scores)
    else:
        # 否则选择重要性排名前N的变量
        cutoff = np.sort(importance_scores)[-int(threshold)]
    
    selected_indices = np.where(importance_scores >= cutoff)[0]
    
    return selected_indices, importance_scores


def mahalanobis_distance(X, reference=None):
    """
    计算马氏距离，用于光谱异常检测
    
    参数:
        X: 输入光谱数据, shape=(n_samples, n_features)
        reference: 参考样本集，默认使用X
        
    返回:
        distances: 每个样本到参考集中心的马氏距离
    """
    if reference is None:
        reference = X
    
    # 计算均值向量和协方差矩阵
    mean_vector = np.mean(reference, axis=0)
    cov_matrix = np.cov(reference, rowvar=False)
    
    # 为确保协方差矩阵可逆，可能需要加入一些正则化
    cov_matrix += 1e-8 * np.eye(cov_matrix.shape[0])
    
    # 计算协方差矩阵的逆
    try:
        inv_cov = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        # 如果不可逆，使用伪逆
        inv_cov = np.linalg.pinv(cov_matrix)
    
    # 计算每个样本的马氏距离
    n_samples = X.shape[0]
    distances = np.zeros(n_samples)
    
    for i in range(n_samples):
        # 样本与均值向量的差
        diff = X[i, :] - mean_vector
        # 计算马氏距离
        distances[i] = np.sqrt(diff.dot(inv_cov).dot(diff.T))
    
    return distances


def optimized_corn_model(X, y, method='pls', cv=5):
    """
    针对玉米成分含量预测的优化模型
    
    参数:
        X: 输入光谱数据, shape=(n_samples, n_features)
        y: 目标变量 (如蛋白质、水分、脂肪等含量)
        method: 模型方法 ('pls', 'svr', 'rf', 'xgb', 'ensemble')
        cv: 交叉验证折数
        
    返回:
        model: 训练好的模型
        predictions: 交叉验证预测结果
        mse: 均方误差
        r2: 决定系数
        rmse: 均方根误差 
        rpd: 性能指标比(Ratio of Performance to Deviation)
    """
    # 根据方法选择合适的模型
    if method == 'pls':
        # 使用PLS，自动选择最优组分数
        model, predictions, mse, r2, optimum_components = pls_regression(
            X, y, n_components=15, cv=cv
        )
        model_info = {'optimum_components': optimum_components}
        
    elif method == 'svr':
        # 使用SVR
        model, predictions, mse, r2 = svm_regression(
            X, y, kernel='rbf', C=10.0, epsilon=0.1, gamma='scale', cv=cv
        )
        model_info = {'kernel': 'rbf', 'C': 10.0, 'epsilon': 0.1}
        
    elif method == 'rf':
        # 使用随机森林回归
        model = RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_split=2, random_state=42)
        predictions = cross_val_predict(model, X, y, cv=cv)
        mse = mean_squared_error(y, predictions)
        r2 = r2_score(y, predictions)
        model.fit(X, y)
        model_info = {'n_estimators': 100, 'max_features': 'auto'}
        
    elif method == 'xgb':
        # 使用XGBoost回归
        try:
            from xgboost import XGBRegressor
            model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
            predictions = cross_val_predict(model, X, y, cv=cv)
            mse = mean_squared_error(y, predictions)
            r2 = r2_score(y, predictions)
            model.fit(X, y)
            model_info = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3}
        except ImportError:
            raise ImportError("请安装XGBoost库：pip install xgboost")
        
    elif method == 'ensemble':
        # 集成模型：先进行变量选择，然后应用PLS
        selected_indices, _ = variable_selection(X, y, method='vip', threshold=0.8)
        X_selected = X[:, selected_indices]
        
        # 使用PLS
        model, predictions, mse, r2, optimum_components = pls_regression(
            X_selected, y, n_components=10, cv=cv
        )
        
        # 包装成一个自定义模型对象
        class EnsembleModel:
            def __init__(self, base_model, selected_indices):
                self.base_model = base_model
                self.selected_indices = selected_indices
                
            def predict(self, X):
                X_selected = X[:, self.selected_indices]
                return self.base_model.predict(X_selected)
        
        model = EnsembleModel(model, selected_indices)
        model_info = {
            'optimum_components': optimum_components, 
            'n_selected_features': len(selected_indices)
        }
        
    else:
        raise ValueError(f"不支持的方法: {method}")
    
    # 计算额外的评估指标
    rmse = np.sqrt(mse)
    
    # RPD (Ratio of Performance to Deviation)
    # RPD > 2.5 表示模型优秀，2.0-2.5 表示良好，1.5-2.0 表示一般，< 1.5 表示不佳
    rpd = np.std(y) / rmse
    
    # 返回结果
    return model, predictions, mse, r2, rmse, rpd, model_info 