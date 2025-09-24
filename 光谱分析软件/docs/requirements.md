# 光谱分析软件依赖说明

本文档列出了光谱分析软件所需的主要依赖项及其版本要求。

## 基本环境

- Python 3.7+
- 操作系统: Windows/Linux/MacOS

## 核心依赖

以下是软件运行所需的核心依赖库：

| 依赖库 | 最低版本 | 说明 |
|--------|---------|------|
| numpy | 1.19.0 | 用于数值计算的基础库 |
| scipy | 1.5.0 | 科学计算库，提供信号处理等功能 |
| matplotlib | 3.3.0 | 图形绘制库 |
| pandas | 1.1.0 | 数据处理与分析库 |
| scikit-learn | 0.23.0 | 机器学习算法库 |
| PyQt5 | 5.15.0 | 图形用户界面库 |
| h5py | 2.10.0 | 用于处理HDF5文件 |

## 可选依赖

以下依赖提供额外功能：

| 依赖库 | 最低版本 | 说明 |
|--------|---------|------|
| lmfit | 1.0.0 | 非线性最小二乘拟合 |
| openpyxl | 3.0.0 | Excel文件读写支持 |
| xlrd | 1.2.0 | 旧Excel文件格式支持 |
| pillow | 7.0.0 | 图像处理支持 |

## 安装指南

推荐使用conda或pip进行安装。完整安装命令：

```bash
# 使用pip安装
pip install numpy>=1.19.0 scipy>=1.5.0 matplotlib>=3.3.0 pandas>=1.1.0 scikit-learn>=0.23.0 PyQt5>=5.15.0 h5py>=2.10.0

# 可选依赖
pip install lmfit>=1.0.0 openpyxl>=3.0.0 xlrd>=1.2.0 pillow>=7.0.0
```

或使用`requirements.txt`文件安装：

```bash
pip install -r requirements.txt
```

## 开发环境依赖

如果您需要进行开发或运行测试，还需要以下依赖：

| 依赖库 | 最低版本 | 说明 |
|--------|---------|------|
| pytest | 6.0.0 | 测试框架 |
| pytest-cov | 2.10.0 | 测试覆盖率报告 |
| flake8 | 3.8.0 | 代码风格检查 |

开发环境依赖安装：

```bash
pip install pytest>=6.0.0 pytest-cov>=2.10.0 flake8>=3.8.0
``` 