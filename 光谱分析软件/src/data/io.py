#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
光谱数据I/O模块
提供各种格式的光谱数据文件读写功能
"""

import os
import numpy as np
import pandas as pd
import json
from datetime import datetime
import traceback

def read_spectrum_file(file_path, file_format=None, **kwargs):
    """
    读取光谱数据文件
    
    参数:
        file_path: 文件路径
        file_format: 文件格式，如果为None，将根据文件扩展名自动推断
        **kwargs: 传递给特定格式读取函数的参数
        
    返回:
        wavelengths: 波长数组
        spectra: 光谱数据矩阵 [n_samples, n_wavelengths]
        meta_info: 元数据字典
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    # 如果没有指定格式，根据扩展名推断
    if file_format is None:
        _, ext = os.path.splitext(file_path)
        file_format = ext[1:].lower()  # 去掉点号
    
    # 根据格式调用相应的读取函数
    if file_format in ['csv', 'txt']:
        return read_csv_spectrum(file_path, **kwargs)
    elif file_format in ['jdx', 'dx']:
        return read_jcamp_spectrum(file_path, **kwargs)
    elif file_format == 'json':
        return read_json_spectrum(file_path, **kwargs)
    elif file_format in ['h5', 'hdf5']:
        return read_hdf5_spectrum(file_path, **kwargs)
    elif file_format == 'mat':
        return read_matlab_spectrum(file_path, **kwargs)
    else:
        raise ValueError(f"不支持的文件格式: {file_format}")

def write_spectrum_file(file_path, wavelengths, spectra, metadata=None, sample_names=None, file_format=None, **kwargs):
    """
    写入光谱数据到文件
    
    参数:
        file_path: 文件路径
        wavelengths: 波长数组
        spectra: 光谱数据矩阵 [n_samples, n_wavelengths]
        metadata: 元数据字典
        sample_names: 样本名称列表
        file_format: 文件格式，如果为None，将根据文件扩展名自动推断
        **kwargs: 传递给特定格式写入函数的参数
        
    返回:
        success: 成功返回True
    """
    # 如果没有指定格式，根据扩展名推断
    if file_format is None:
        _, ext = os.path.splitext(file_path)
        file_format = ext[1:].lower()  # 去掉点号
    
    # 检查数据格式
    wavelengths = np.array(wavelengths)
    spectra = np.array(spectra)
    
    if len(wavelengths) != spectra.shape[1]:
        raise ValueError(f"波长数量 ({len(wavelengths)}) 与光谱数据列数 ({spectra.shape[1]}) 不匹配")
    
    # 准备元数据
    if metadata is None:
        metadata = {}
    
    # 添加基本信息
    metadata.update({
        'n_samples': spectra.shape[0],
        'n_wavelengths': len(wavelengths),
        'min_wavelength': float(min(wavelengths)),
        'max_wavelength': float(max(wavelengths)),
        'export_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })
    
    # 根据格式调用相应的写入函数
    try:
        if file_format in ['csv', 'txt']:
            return write_csv_spectrum(file_path, wavelengths, spectra, metadata, sample_names, **kwargs)
        elif file_format == 'json':
            return write_json_spectrum(file_path, wavelengths, spectra, metadata, sample_names, **kwargs)
        elif file_format in ['h5', 'hdf5']:
            return write_hdf5_spectrum(file_path, wavelengths, spectra, metadata, sample_names, **kwargs)
        else:
            raise ValueError(f"不支持的文件格式: {file_format}")
    except Exception as e:
        print(f"写入文件失败: {e}")
        traceback.print_exc()
        return False

def read_csv_spectrum(file_path, delimiter=',', header=0, wavelength_unit='nm', **kwargs):
    """
    读取CSV格式的光谱数据
    假设CSV格式：第一列为样本ID，其余列为不同波长的光谱数据
    列名应该是波长值
    
    参数:
        file_path: 文件路径
        delimiter: 分隔符
        header: 表头行号
        wavelength_unit: 波长单位
        **kwargs: 传递给pandas.read_csv的参数
        
    返回:
        wavelengths: 波长数组
        spectra: 光谱数据矩阵 [n_samples, n_wavelengths]
        meta_info: 元数据字典
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path, delimiter=delimiter, header=header, **kwargs)
        
        # 提取样本ID（第一列）
        sample_names = df.iloc[:, 0].tolist()
        
        # 提取光谱数据（从第二列开始）
        spectra = df.iloc[:, 1:].values
        
        # 提取波长（列名）
        try:
            wavelengths = [float(col) for col in df.columns[1:]]
        except ValueError:
            # 如果列名不能转换为浮点数，则使用索引
            wavelengths = np.arange(spectra.shape[1])
            print("警告: 无法从列名提取波长值，使用索引替代")
        
        # 创建元数据
        meta_info = {
            'source_file': file_path,
            'format': 'csv',
            'n_samples': len(sample_names),
            'n_wavelengths': len(wavelengths),
            'wavelength_unit': wavelength_unit,
            'sample_names': sample_names,
            'min_wavelength': min(wavelengths),
            'max_wavelength': max(wavelengths),
            'load_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return wavelengths, spectra, meta_info
    
    except Exception as e:
        raise IOError(f"读取CSV文件失败: {e}")

def write_csv_spectrum(file_path, wavelengths, spectra, metadata=None, sample_names=None, delimiter=',', **kwargs):
    """
    写入光谱数据到CSV文件
    
    参数:
        file_path: 文件路径
        wavelengths: 波长数组
        spectra: 光谱数据矩阵 [n_samples, n_wavelengths]
        metadata: 元数据字典
        sample_names: 样本名称列表
        delimiter: 分隔符
        **kwargs: 传递给pandas.to_csv的参数
        
    返回:
        success: 成功返回True
    """
    try:
        # 创建DataFrame
        df = pd.DataFrame()
        
        # 添加样本ID列
        if sample_names is None:
            sample_names = [f"Sample_{i+1}" for i in range(spectra.shape[0])]
        
        df['sample_id'] = sample_names
        
        # 添加光谱数据列
        for i, wl in enumerate(wavelengths):
            df[f"{wl:.2f}"] = spectra[:, i]
        
        # 保存为CSV
        df.to_csv(file_path, index=False, sep=delimiter, **kwargs)
        
        return True
    
    except Exception as e:
        print(f"写入CSV文件失败: {e}")
        return False

def read_json_spectrum(file_path, **kwargs):
    """
    读取JSON格式的光谱数据
    
    参数:
        file_path: 文件路径
        **kwargs: 额外参数
        
    返回:
        wavelengths: 波长数组
        spectra: 光谱数据矩阵 [n_samples, n_wavelengths]
        meta_info: 元数据字典
    """
    try:
        # 读取JSON文件
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 提取数据
        if 'wavelengths' in data and 'spectra' in data:
            wavelengths = np.array(data['wavelengths'])
            spectra = np.array(data['spectra'])
            
            # 提取样本名称
            sample_names = data.get('sample_names', [f"Sample_{i+1}" for i in range(spectra.shape[0])])
            
            # 提取元数据
            meta_info = data.get('metadata', {})
            meta_info.update({
                'source_file': file_path,
                'format': 'json',
                'n_samples': spectra.shape[0],
                'n_wavelengths': len(wavelengths),
                'sample_names': sample_names,
                'min_wavelength': float(min(wavelengths)),
                'max_wavelength': float(max(wavelengths)),
                'load_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            
            return wavelengths, spectra, meta_info
        else:
            raise ValueError("JSON文件格式不正确，缺少'wavelengths'或'spectra'字段")
    
    except Exception as e:
        raise IOError(f"读取JSON文件失败: {e}")

def write_json_spectrum(file_path, wavelengths, spectra, metadata=None, sample_names=None, **kwargs):
    """
    写入光谱数据到JSON文件
    
    参数:
        file_path: 文件路径
        wavelengths: 波长数组
        spectra: 光谱数据矩阵 [n_samples, n_wavelengths]
        metadata: 元数据字典
        sample_names: 样本名称列表
        **kwargs: 额外参数
        
    返回:
        success: 成功返回True
    """
    try:
        # 准备样本名称
        if sample_names is None:
            sample_names = [f"Sample_{i+1}" for i in range(spectra.shape[0])]
        
        # 准备数据
        data = {
            'wavelengths': wavelengths.tolist() if isinstance(wavelengths, np.ndarray) else wavelengths,
            'spectra': spectra.tolist() if isinstance(spectra, np.ndarray) else spectra,
            'sample_names': sample_names,
            'metadata': metadata or {}
        }
        
        # 写入JSON文件
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        return True
    
    except Exception as e:
        print(f"写入JSON文件失败: {e}")
        return False

def read_hdf5_spectrum(file_path, **kwargs):
    """
    读取HDF5格式的光谱数据
    
    参数:
        file_path: 文件路径
        **kwargs: 额外参数
        
    返回:
        wavelengths: 波长数组
        spectra: 光谱数据矩阵 [n_samples, n_wavelengths]
        meta_info: 元数据字典
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("读取HDF5文件需要安装h5py库")
    
    try:
        # 打开HDF5文件
        with h5py.File(file_path, 'r') as f:
            if 'wavelengths' in f and 'spectra' in f:
                # 读取波长和光谱数据
                wavelengths = np.array(f['wavelengths'])
                spectra = np.array(f['spectra'])
                
                # 读取样本名称
                if 'sample_names' in f:
                    sample_names = [s.decode('utf-8') if isinstance(s, bytes) else s 
                                   for s in f['sample_names']]
                else:
                    sample_names = [f"Sample_{i+1}" for i in range(spectra.shape[0])]
                
                # 读取元数据
                meta_info = {'source_file': file_path, 'format': 'hdf5'}
                if 'metadata' in f:
                    for k, v in f['metadata'].attrs.items():
                        meta_info[k] = v
                
                # 添加基本信息
                meta_info.update({
                    'n_samples': spectra.shape[0],
                    'n_wavelengths': len(wavelengths),
                    'sample_names': sample_names,
                    'min_wavelength': float(min(wavelengths)),
                    'max_wavelength': float(max(wavelengths)),
                    'load_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
                
                return wavelengths, spectra, meta_info
            else:
                raise ValueError("HDF5文件格式不正确，缺少'wavelengths'或'spectra'数据集")
    
    except Exception as e:
        raise IOError(f"读取HDF5文件失败: {e}")

def write_hdf5_spectrum(file_path, wavelengths, spectra, metadata=None, sample_names=None, **kwargs):
    """
    写入光谱数据到HDF5文件
    
    参数:
        file_path: 文件路径
        wavelengths: 波长数组
        spectra: 光谱数据矩阵 [n_samples, n_wavelengths]
        metadata: 元数据字典
        sample_names: 样本名称列表
        **kwargs: 额外参数
        
    返回:
        success: 成功返回True
    """
    try:
        import h5py
    except ImportError:
        print("写入HDF5文件需要安装h5py库")
        return False
    
    try:
        # 准备样本名称
        if sample_names is None:
            sample_names = [f"Sample_{i+1}" for i in range(spectra.shape[0])]
        
        # 创建HDF5文件
        with h5py.File(file_path, 'w') as f:
            # 保存波长和光谱数据
            f.create_dataset('wavelengths', data=wavelengths)
            f.create_dataset('spectra', data=spectra)
            
            # 保存样本名称
            string_dt = h5py.special_dtype(vlen=str)
            sample_names_ds = f.create_dataset('sample_names', (len(sample_names),), dtype=string_dt)
            for i, name in enumerate(sample_names):
                sample_names_ds[i] = name
            
            # 保存元数据
            if metadata:
                meta_grp = f.create_group('metadata')
                for k, v in metadata.items():
                    if isinstance(v, (str, int, float, bool)):
                        meta_grp.attrs[k] = v
        
        return True
    
    except Exception as e:
        print(f"写入HDF5文件失败: {e}")
        traceback.print_exc()
        return False

def read_jcamp_spectrum(file_path, **kwargs):
    """
    读取JCAMP-DX格式的光谱数据
    
    参数:
        file_path: 文件路径
        **kwargs: 额外参数
        
    返回:
        wavelengths: 波长数组
        spectra: 光谱数据矩阵 [n_samples, n_wavelengths]
        meta_info: 元数据字典
    """
    try:
        # 简单实现，实际情况可能需要更复杂的解析逻辑
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        meta_info = {'source_file': file_path, 'format': 'jcamp-dx'}
        x_values = []
        y_values = []
        in_data_section = False
        
        for line in lines:
            line = line.strip()
            
            # 解析元数据
            if line.startswith('##'):
                if '=' in line:
                    key, value = line[2:].split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    meta_info[key] = value
                
                # 检查数据段开始
                if line.startswith('##XYDATA'):
                    in_data_section = True
                    continue
            
            # 解析数据
            elif in_data_section and not line.startswith('##'):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        x = float(parts[0])
                        y = float(parts[1])
                        x_values.append(x)
                        y_values.append(y)
                    except ValueError:
                        pass
        
        # 检查是否有数据
        if not x_values:
            raise ValueError("未找到有效数据")
        
        # 创建波长和光谱数据
        wavelengths = np.array(x_values)
        spectra = np.array(y_values).reshape(1, -1)  # 单个光谱
        
        # 添加基本信息
        meta_info.update({
            'n_samples': 1,
            'n_wavelengths': len(wavelengths),
            'sample_names': ['Sample_1'],
            'min_wavelength': float(min(wavelengths)),
            'max_wavelength': float(max(wavelengths)),
            'load_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
        return wavelengths, spectra, meta_info
    
    except Exception as e:
        raise IOError(f"读取JCAMP-DX文件失败: {e}")

def read_matlab_spectrum(file_path, **kwargs):
    """
    读取MATLAB (.mat) 格式的光谱数据
    
    参数:
        file_path: 文件路径
        **kwargs: 额外参数
        
    返回:
        wavelengths: 波长数组
        spectra: 光谱数据矩阵 [n_samples, n_wavelengths]
        meta_info: 元数据字典
    """
    try:
        import scipy.io as sio
    except ImportError:
        raise ImportError("读取MATLAB文件需要安装scipy库")
    
    try:
        # 加载.mat文件
        mat_data = sio.loadmat(file_path)
        
        # 寻找可能的波长和光谱数据变量
        wavelengths_var = None
        spectra_var = None
        
        # 常见的变量名
        wavelength_candidates = ['wavelength', 'wavelengths', 'x', 'wl', 'lambda']
        spectra_candidates = ['spectra', 'spectrum', 'y', 'data', 'intensity']
        
        # 在mat文件中寻找匹配的变量
        for var_name in mat_data.keys():
            if var_name.startswith('__'):  # 跳过Matlab内部变量
                continue
                
            # 检查变量名和形状判断是否为波长或光谱数据
            var_data = mat_data[var_name]
            
            if var_name.lower() in wavelength_candidates or (var_data.ndim in [1, 2] and min(var_data.shape) == 1):
                if wavelengths_var is None or (var_name.lower() in wavelength_candidates):
                    wavelengths_var = var_name
                    
            if var_name.lower() in spectra_candidates or var_data.ndim == 2:
                if spectra_var is None or (var_name.lower() in spectra_candidates):
                    spectra_var = var_name
        
        # 提取数据
        if wavelengths_var is not None and spectra_var is not None:
            wavelengths = np.ravel(mat_data[wavelengths_var])
            spectra = mat_data[spectra_var]
            
            # 确保光谱数据形状正确
            if spectra.ndim == 1:
                spectra = spectra.reshape(1, -1)  # 单个光谱
            elif spectra.shape[0] < spectra.shape[1]:
                # 假设每行是一个光谱，如果列数更多则转置
                spectra = spectra.T
                
            # 创建元数据
            meta_info = {
                'source_file': file_path,
                'format': 'matlab',
                'wavelength_var': wavelengths_var,
                'spectra_var': spectra_var,
                'n_samples': spectra.shape[0],
                'n_wavelengths': len(wavelengths),
                'sample_names': [f"Sample_{i+1}" for i in range(spectra.shape[0])],
                'min_wavelength': float(min(wavelengths)),
                'max_wavelength': float(max(wavelengths)),
                'load_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # 添加其他变量作为元数据
            for var_name in mat_data.keys():
                if var_name.startswith('__') or var_name in [wavelengths_var, spectra_var]:
                    continue
                    
                var_data = mat_data[var_name]
                if isinstance(var_data, np.ndarray) and var_data.size == 1:
                    meta_info[var_name] = float(var_data)
            
            return wavelengths, spectra, meta_info
        else:
            raise ValueError("无法在MATLAB文件中找到波长和光谱数据")
    
    except Exception as e:
        raise IOError(f"读取MATLAB文件失败: {e}")


def read_corn_data(file_path, **kwargs):
    """
    读取玉米光谱数据专用函数

    参数:
        file_path: 文件路径
        **kwargs: 其他参数

    返回:
        wavelengths: 波长数组
        spectra: 光谱数据矩阵 [n_samples, n_wavelengths]
        target: 目标值数组 (如蛋白质含量、水分等)
        meta_info: 元数据字典
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        # 获取文件扩展名
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        # 根据文件格式读取数据
        if ext in ['.csv', '.txt']:
            # 读取CSV/TXT格式的玉米数据
            df = pd.read_csv(file_path, **kwargs)

            # 假设第一列是波长，其余列是样本
            wavelengths = df.iloc[:, 0].values
            spectra = df.iloc[:, 1:].values.T

            # 如果有额外的列作为目标值
            target = None
            if 'target' in df.columns:
                target = df['target'].values
            elif 'protein' in df.columns:
                target = df['protein'].values
            elif 'moisture' in df.columns:
                target = df['moisture'].values

        elif ext in ['.xlsx', '.xls']:
            # 读取Excel格式的玉米数据
            df = pd.read_excel(file_path, **kwargs)

            wavelengths = df.iloc[:, 0].values
            spectra = df.iloc[:, 1:].values.T

            target = None
            if 'target' in df.columns:
                target = df['target'].values
            elif 'protein' in df.columns:
                target = df['protein'].values
            elif 'moisture' in df.columns:
                target = df['moisture'].values

        else:
            # 对于其他格式，使用通用的光谱读取函数
            wavelengths, spectra, meta_info = read_spectrum_file(file_path, **kwargs)
            target = meta_info.get('target', None)

            return wavelengths, spectra, target, meta_info

        # 创建元数据
        meta_info = {
            'file_path': file_path,
            'file_format': ext,
            'n_samples': spectra.shape[0],
            'n_wavelengths': len(wavelengths),
            'sample_names': [f"Corn_Sample_{i+1}" for i in range(spectra.shape[0])],
            'min_wavelength': float(min(wavelengths)),
            'max_wavelength': float(max(wavelengths)),
            'load_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_type': 'corn_spectra'
        }

        # 如果没有找到目标值，生成示例数据
        if target is None:
            np.random.seed(42)
            target = np.random.normal(15, 3, spectra.shape[0])  # 示例蛋白质含量
            meta_info['target_type'] = 'simulated_protein'
        else:
            meta_info['target_type'] = 'measured'

        meta_info['target'] = target.tolist() if hasattr(target, 'tolist') else target

        return wavelengths, spectra, target, meta_info

    except Exception as e:
        raise IOError(f"读取玉米数据失败: {e}") 