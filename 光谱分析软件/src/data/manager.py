#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据管理模块
提供光谱数据集的管理功能
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import json
import pickle
from datetime import datetime

from src.data.io import read_spectrum_file, write_spectrum_file


class SpectrumDataset:
    """光谱数据集类"""
    
    def __init__(self, name=None):
        """
        初始化光谱数据集
        
        参数:
            name: 数据集名称
        """
        self.name = name or f"Dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.wavelengths = None
        self.spectra = None
        self.sample_names = None
        self.target = None  # 目标变量（如成分含量）
        self.target_name = None  # 目标变量名称
        self.metadata = {}
        self.history = []  # 操作历史记录
        self._add_history("创建数据集")
    
    def _add_history(self, operation, details=None):
        """添加操作历史记录"""
        entry = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'operation': operation
        }
        if details:
            entry['details'] = details
        self.history.append(entry)
    
    def load_from_file(self, file_path, **kwargs):
        """
        从文件加载光谱数据
        
        参数:
            file_path: 文件路径
            **kwargs: 传递给读取函数的参数
            
        返回:
            self: 支持链式调用
        """
        try:
            wavelengths, spectra, meta_info = read_spectrum_file(file_path, **kwargs)
            
            self.wavelengths = wavelengths
            self.spectra = spectra
            
            # 提取样本名称（如果有）
            if 'sample_names' in meta_info and meta_info['sample_names']:
                self.sample_names = meta_info['sample_names']
            else:
                self.sample_names = [f"Sample_{i+1}" for i in range(spectra.shape[0])]
            
            # 更新元数据
            self.metadata.update(meta_info)
            
            # 记录操作历史
            self._add_history("从文件加载数据", {"file": file_path})
            
            return self
        except Exception as e:
            raise IOError(f"加载文件失败: {e}")
    
    # 添加load_file作为load_from_file的别名
    def load_file(self, file_path, file_format='csv'):
        """
        加载文件 (load_from_file的别名)

        参数:
            file_path: 文件路径
            file_format: 文件格式，例如'csv', 'txt', 'json'等

        返回:
            success: 成功返回True
        """
        try:
            # 简单实现，处理常见格式
            if file_format.lower() in ['csv', 'txt']:
                # 尝试使用pandas读取，支持多种编码
                encodings = ['utf-8', 'gbk', 'gb2312', 'ISO-8859-1', 'cp1252']
                df = None

                for encoding in encodings:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        print(f"成功使用 {encoding} 编码读取文件")
                        break
                    except UnicodeDecodeError:
                        continue

                if df is None:
                    raise ValueError(f"无法解码文件，尝试的编码：{encodings}")

                # 检查数据格式并相应处理
                if len(df.columns) < 1:
                    raise ValueError("数据格式错误：至少需要一列数据")

                # 检查是否为玉米数据格式（包含Moisture(%), Oil(%), Protein(%), Starch(%)列）
                corn_columns = ['Moisture(%)', 'Oil(%)', 'Protein(%)', 'Starch(%)']
                is_corn_data = any(col in df.columns for col in corn_columns)

                if is_corn_data:
                    # 玉米数据格式：处理包含成分数据的文件
                    self._load_corn_data_format(df)
                else:
                    # 普通光谱数据格式：自动检测是否有样本ID列
                    self._load_spectrum_data_format(df)

            elif file_format.lower() == 'json':
                # 读取JSON
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 尝试解析JSON结构
                if 'spectra' in data and 'wavelengths' in data:
                    self.spectra = np.array(data['spectra'])
                    self.wavelengths = np.array(data['wavelengths'])
                    if 'sample_names' in data:
                        self.sample_names = data['sample_names']
                    else:
                        self.sample_names = [f"Sample_{i+1}" for i in range(self.spectra.shape[0])]
                else:
                    raise ValueError("JSON格式不兼容")

            elif file_format.lower() in ['h5', 'hdf5']:
                # 需要h5py库
                import h5py
                with h5py.File(file_path, 'r') as f:
                    if 'spectra' in f and 'wavelengths' in f:
                        self.spectra = np.array(f['spectra'])
                        self.wavelengths = np.array(f['wavelengths'])
                        if 'sample_names' in f:
                            self.sample_names = list(f['sample_names'])
                        else:
                            self.sample_names = [f"Sample_{i+1}" for i in range(self.spectra.shape[0])]
                    else:
                        raise ValueError("HDF5文件结构不兼容")

            else:
                # 其他格式可以根据需要添加
                raise ValueError(f"不支持的文件格式: {file_format}")

            # 更新元数据
            self.metadata = {
                'source_file': file_path,
                'format': file_format,
                'n_samples': len(self.sample_names),
                'n_wavelengths': len(self.wavelengths),
                'min_wavelength': min(self.wavelengths) if self.wavelengths is not None else None,
                'max_wavelength': max(self.wavelengths) if self.wavelengths is not None else None,
                'load_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            # 记录操作历史
            self._add_history("加载文件", {"file": file_path, "format": file_format})

            return True

        except Exception as e:
            print(f"加载文件错误: {e}")
            return False

    def _load_corn_data_format(self, df):
        """
        加载玉米数据格式的文件

        参数:
            df: pandas DataFrame包含数据
        """
        # 查找玉米成分列
        corn_columns = ['Moisture(%)', 'Oil(%)', 'Protein(%)', 'Starch(%)']
        found_columns = [col for col in corn_columns if col in df.columns]

        if not found_columns:
            raise ValueError("未找到玉米成分数据列")

        # 查找样本ID列（通常是第一列）
        sample_id_col = None
        if df.columns[0] not in corn_columns and not self._is_wavelength_column(df.columns[0]):
            sample_id_col = df.columns[0]
            self.sample_names = df[sample_id_col].tolist()
            data_start_col = 1
        else:
            self.sample_names = [f"Sample_{i+1}" for i in range(len(df))]
            data_start_col = 0

        # 查找波长列
        wavelength_cols = []
        property_cols = found_columns

        for col in df.columns:
            if col == sample_id_col:
                continue
            if col in property_cols:
                continue
            # 检查是否为波长列（包含nm或为数字）
            if self._is_wavelength_column(col):
                wavelength_cols.append(col)

        if not wavelength_cols:
            raise ValueError("未找到波长数据列")

        # 提取波长
        try:
            self.wavelengths = [float(col.replace('nm', '')) for col in wavelength_cols]
        except:
            self.wavelengths = np.arange(len(wavelength_cols))

        # 提取光谱数据
        self.spectra = df[wavelength_cols].values

        # 提取目标变量（选择第一个找到的成分列）
        if property_cols:
            target_col = property_cols[0]  # 默认使用第一个成分列
            self.target = df[target_col].values
            self.target_name = target_col

            # 如果有多个成分列，存储在元数据中
            if len(property_cols) > 1:
                self.metadata['additional_targets'] = {}
                for col in property_cols[1:]:
                    self.metadata['additional_targets'][col] = df[col].values.tolist()

    def _load_spectrum_data_format(self, df):
        """
        加载普通光谱数据格式的文件

        参数:
            df: pandas DataFrame包含数据
        """
        # 检查第一列是否为样本ID
        first_col = df.columns[0]

        if self._is_wavelength_column(first_col):
            # 格式1：第一列是波长，其余是光谱数据
            try:
                self.wavelengths = [float(col.replace('nm', '')) for col in df.columns]
            except:
                self.wavelengths = np.arange(len(df.columns))

            self.spectra = df.values
            self.sample_names = [f"Sample_{i+1}" for i in range(len(df))]

        else:
            # 格式2：第一列是样本ID，其余是光谱数据
            if len(df.columns) < 2:
                raise ValueError("数据格式错误：至少需要两列数据")

            # 提取样本ID
            self.sample_names = df.iloc[:, 0].tolist()

            # 提取光谱数据
            self.spectra = df.iloc[:, 1:].values

            # 如果列名可能是波长，则提取
            try:
                self.wavelengths = [float(col.replace('nm', '')) for col in df.columns[1:]]
            except:
                self.wavelengths = np.arange(self.spectra.shape[1])

    def _is_wavelength_column(self, col_name):
        """
        检查列名是否为波长列

        参数:
            col_name: 列名

        返回:
            is_wavelength: 如果是波长列返回True，否则返回False
        """
        # 检查是否包含'nm'
        if 'nm' in str(col_name).lower():
            return True

        # 检查是否可以转换为数字
        try:
            float(str(col_name).replace('nm', ''))
            return True
        except:
            return False

    def save_file(self, file_path, format_type='csv'):
        """
        保存文件 (save_to_file的别名)
        
        参数:
            file_path: 文件路径
            format_type: 文件格式，例如'csv', 'txt', 'json'等
            
        返回:
            success: 成功返回True
        """
        try:
            if self.wavelengths is None or self.spectra is None:
                raise ValueError("无数据可保存")
                
            if format_type.lower() in ['csv', 'txt']:
                # 创建DataFrame
                df = pd.DataFrame()
                
                # 添加样本ID列
                df['sample_id'] = self.sample_names if self.sample_names else [f"Sample_{i+1}" for i in range(self.spectra.shape[0])]
                
                # 添加光谱数据列
                for i, wl in enumerate(self.wavelengths):
                    df[f"{wl:.2f}"] = self.spectra[:, i]
                
                # 保存为CSV
                df.to_csv(file_path, index=False)
                
            elif format_type.lower() == 'json':
                # 创建JSON数据
                data = {
                    'wavelengths': self.wavelengths.tolist() if isinstance(self.wavelengths, np.ndarray) else self.wavelengths,
                    'spectra': self.spectra.tolist() if isinstance(self.spectra, np.ndarray) else self.spectra,
                    'sample_names': self.sample_names,
                    'metadata': self.metadata
                }
                
                # 保存为JSON
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
                    
            elif format_type.lower() in ['h5', 'hdf5']:
                # 需要h5py库
                import h5py
                with h5py.File(file_path, 'w') as f:
                    f.create_dataset('wavelengths', data=self.wavelengths)
                    f.create_dataset('spectra', data=self.spectra)
                    if self.sample_names:
                        f.create_dataset('sample_names', data=np.array(self.sample_names, dtype='S'))
                    # 保存元数据
                    meta_grp = f.create_group('metadata')
                    for k, v in self.metadata.items():
                        if isinstance(v, (str, int, float, bool)):
                            meta_grp.attrs[k] = v
                
            else:
                # 其他格式可以根据需要添加
                raise ValueError(f"不支持的文件格式: {format_type}")
                
            # 记录操作历史
            self._add_history("保存文件", {"file": file_path, "format": format_type})
            
            return True
            
        except Exception as e:
            print(f"保存文件错误: {e}")
            return False
    
    def save_to_file(self, file_path, **kwargs):
        """
        将光谱数据保存到文件
        
        参数:
            file_path: 文件路径
            **kwargs: 传递给写入函数的参数
            
        返回:
            success: 成功返回True
        """
        try:
            # 确保数据已加载
            if self.wavelengths is None or self.spectra is None:
                raise ValueError("无数据可保存")
            
            success = write_spectrum_file(
                file_path, 
                self.wavelengths, 
                self.spectra, 
                self.metadata,
                self.sample_names,
                **kwargs
            )
            
            if success:
                self._add_history("保存数据到文件", {"file": file_path})
            
            return success
        except Exception as e:
            raise IOError(f"保存文件失败: {e}")
    
    def save_session(self, file_path):
        """
        保存完整会话状态
        
        参数:
            file_path: 文件路径
            
        返回:
            success: 成功返回True
        """
        try:
            # 创建会话字典
            session = {
                'name': self.name,
                'wavelengths': self.wavelengths,
                'spectra': self.spectra,
                'sample_names': self.sample_names,
                'metadata': self.metadata,
                'history': self.history,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # 保存为pickle文件
            with open(file_path, 'wb') as f:
                pickle.dump(session, f)
            
            self._add_history("保存会话", {"file": file_path})
            return True
        except Exception as e:
            raise IOError(f"保存会话失败: {e}")
    
    def load_session(self, file_path):
        """
        加载会话状态
        
        参数:
            file_path: 文件路径
            
        返回:
            self: 支持链式调用
        """
        try:
            # 加载pickle文件
            with open(file_path, 'rb') as f:
                session = pickle.load(f)
            
            # 恢复状态
            self.name = session['name']
            self.wavelengths = session['wavelengths']
            self.spectra = session['spectra']
            self.sample_names = session['sample_names']
            self.metadata = session['metadata']
            self.history = session['history']
            
            self._add_history("加载会话", {"file": file_path})
            return self
        except Exception as e:
            raise IOError(f"加载会话失败: {e}")
    
    def get_spectrum(self, index_or_name):
        """
        获取指定样本的光谱
        
        参数:
            index_or_name: 样本索引或名称
            
        返回:
            spectrum: 光谱数据
        """
        if self.spectra is None:
            raise ValueError("没有加载光谱数据")
        
        if isinstance(index_or_name, str):
            # 通过名称查找
            if self.sample_names is None:
                raise ValueError("未设置样本名称")
            
            try:
                index = self.sample_names.index(index_or_name)
            except ValueError:
                raise ValueError(f"未找到名称为 '{index_or_name}' 的样本")
        else:
            # 通过索引查找
            index = index_or_name
            
        if index < 0 or index >= self.spectra.shape[0]:
            raise IndexError(f"索引 {index} 超出范围 [0, {self.spectra.shape[0]-1}]")
        
        return self.spectra[index]
    
    def add_spectrum(self, spectrum, sample_name=None):
        """
        添加一条光谱
        
        参数:
            spectrum: 待添加的光谱数据
            sample_name: 样本名称
            
        返回:
            self: 支持链式调用
        """
        if self.wavelengths is None:
            raise ValueError("需要先加载数据才能添加光谱")
        
        # 确保光谱长度匹配
        if len(spectrum) != len(self.wavelengths):
            raise ValueError(f"光谱长度不匹配: 当前 {len(self.wavelengths)}, 添加 {len(spectrum)}")
        
        # 添加光谱数据
        spectrum = np.array(spectrum).reshape(1, -1)
        if self.spectra is None:
            self.spectra = spectrum
        else:
            self.spectra = np.vstack([self.spectra, spectrum])
        
        # 添加样本名称
        if sample_name is None:
            sample_name = f"Sample_{self.spectra.shape[0]}"
        
        if self.sample_names is None:
            self.sample_names = [sample_name]
        else:
            self.sample_names.append(sample_name)
        
        # 更新元数据
        if 'n_samples' in self.metadata:
            self.metadata['n_samples'] = self.spectra.shape[0]
        
        self._add_history("添加光谱", {"sample_name": sample_name})
        return self
    
    def remove_spectrum(self, index_or_name):
        """
        移除指定样本的光谱
        
        参数:
            index_or_name: 样本索引或名称
            
        返回:
            self: 支持链式调用
        """
        if self.spectra is None:
            raise ValueError("没有加载光谱数据")
        
        if isinstance(index_or_name, str):
            # 通过名称查找
            if self.sample_names is None:
                raise ValueError("未设置样本名称")
            
            try:
                index = self.sample_names.index(index_or_name)
            except ValueError:
                raise ValueError(f"未找到名称为 '{index_or_name}' 的样本")
        else:
            # 通过索引查找
            index = index_or_name
        
        # 保存要删除的样本名称（用于历史记录）
        sample_name = self.sample_names[index] if self.sample_names else f"Sample_{index+1}"
        
        # 移除光谱数据
        self.spectra = np.delete(self.spectra, index, axis=0)
        
        # 移除样本名称
        if self.sample_names:
            self.sample_names.pop(index)
        
        # 更新元数据
        if 'n_samples' in self.metadata:
            self.metadata['n_samples'] = self.spectra.shape[0]
        
        self._add_history("移除光谱", {"sample_name": sample_name})
        return self
    
    def get_metadata(self):
        """
        获取数据集元数据
        
        返回:
            metadata: 元数据字典
        """
        # 确保基本信息存在
        if self.wavelengths is not None and self.spectra is not None:
            self.metadata.update({
                'name': self.name,
                'n_samples': self.spectra.shape[0],
                'n_wavelengths': len(self.wavelengths),
                'wavelength_range': (self.wavelengths[0], self.wavelengths[-1]),
                'last_modified': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
        
        return self.metadata
    
    def set_wavelengths(self, wavelengths):
        """
        设置波长数组
        
        参数:
            wavelengths: 新的波长数组
            
        返回:
            self: 支持链式调用
        """
        # 验证波长数组长度
        if self.spectra is not None and len(wavelengths) != self.spectra.shape[1]:
            raise ValueError(f"波长数组长度与光谱维度不匹配: {len(wavelengths)} vs {self.spectra.shape[1]}")
        
        self.wavelengths = np.array(wavelengths)
        
        # 更新元数据
        if 'wavelength_range' in self.metadata:
            self.metadata['wavelength_range'] = (self.wavelengths[0], self.wavelengths[-1])
        if 'n_wavelengths' in self.metadata:
            self.metadata['n_wavelengths'] = len(self.wavelengths)
        
        self._add_history("设置波长数组")
        return self
    
    def set_sample_names(self, sample_names):
        """
        设置样本名称
        
        参数:
            sample_names: 样本名称列表
            
        返回:
            self: 支持链式调用
        """
        if self.spectra is None:
            raise ValueError("需要先加载光谱数据")
        
        if len(sample_names) != self.spectra.shape[0]:
            raise ValueError(f"样本名称数量与光谱数量不匹配: {len(sample_names)} vs {self.spectra.shape[0]}")
        
        self.sample_names = list(sample_names)
        self._add_history("设置样本名称")
        return self
    
    def export_summary(self, file_path, format='csv'):
        """
        导出数据集摘要信息
        
        参数:
            file_path: 输出文件路径
            format: 输出格式，支持 'csv', 'json', 'txt'
            
        返回:
            success: 成功返回True
        """
        if self.spectra is None or self.wavelengths is None:
            raise ValueError("无数据可导出")
        
        # 计算每个样本的基本统计信息
        stats = []
        for i in range(self.spectra.shape[0]):
            spectrum = self.spectra[i]
            sample_name = self.sample_names[i] if self.sample_names else f"Sample_{i+1}"
            
            stat = {
                'sample_name': sample_name,
                'mean': np.mean(spectrum),
                'std': np.std(spectrum),
                'min': np.min(spectrum),
                'max': np.max(spectrum),
                'range': np.max(spectrum) - np.min(spectrum)
            }
            stats.append(stat)
        
        # 导出统计信息
        if format.lower() == 'csv':
            df = pd.DataFrame(stats)
            df.to_csv(file_path, index=False)
        elif format.lower() == 'json':
            summary = {
                'dataset_name': self.name,
                'n_samples': self.spectra.shape[0],
                'n_wavelengths': len(self.wavelengths),
                'wavelength_range': [float(self.wavelengths[0]), float(self.wavelengths[-1])],
                'sample_stats': stats
            }
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
        elif format.lower() == 'txt':
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"数据集名称: {self.name}\n")
                f.write(f"样本数量: {self.spectra.shape[0]}\n")
                f.write(f"波长点数: {len(self.wavelengths)}\n")
                f.write(f"波长范围: {self.wavelengths[0]} - {self.wavelengths[-1]}\n\n")
                
                f.write("样本统计信息:\n")
                f.write("-" * 80 + "\n")
                f.write(f"{'样本':20s} {'均值':10s} {'标准差':10s} {'最小值':10s} {'最大值':10s} {'范围':10s}\n")
                f.write("-" * 80 + "\n")
                
                for stat in stats:
                    f.write(f"{stat['sample_name']:20s} {stat['mean']:10.4f} {stat['std']:10.4f} "
                            f"{stat['min']:10.4f} {stat['max']:10.4f} {stat['range']:10.4f}\n")
        else:
            raise ValueError(f"不支持的导出格式: {format}")
        
        self._add_history("导出数据摘要", {"format": format, "file": file_path})
        return True
    
    def get_wavelength_index(self, wavelength):
        """
        获取最接近指定波长的索引
        
        参数:
            wavelength: 目标波长
            
        返回:
            index: 最接近的波长索引
        """
        if self.wavelengths is None:
            raise ValueError("未加载波长数据")
        
        return np.argmin(np.abs(self.wavelengths - wavelength))
    
    def get_wavelength_slice(self, start_wl, end_wl):
        """
        获取指定波长范围的切片
        
        参数:
            start_wl: 起始波长
            end_wl: 结束波长
            
        返回:
            wavelengths_slice: 波长切片
            spectra_slice: 光谱数据切片
        """
        if self.wavelengths is None or self.spectra is None:
            raise ValueError("未加载数据")
        
        # 找到最接近的索引
        start_idx = self.get_wavelength_index(start_wl)
        end_idx = self.get_wavelength_index(end_wl)
        
        # 确保起始索引小于结束索引
        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx
        
        # 提取切片
        wavelengths_slice = self.wavelengths[start_idx:end_idx+1]
        spectra_slice = self.spectra[:, start_idx:end_idx+1]
        
        return wavelengths_slice, spectra_slice
    
    def get_summary(self):
        """
        获取数据集摘要信息
        
        返回:
            summary: 摘要信息字符串
        """
        if self.wavelengths is None or self.spectra is None:
            return "没有加载数据"
        
        summary = [
            f"数据集名称: {self.name}",
            f"样本数量: {self.spectra.shape[0]}",
            f"波长点数: {len(self.wavelengths)}",
            f"波长范围: {self.wavelengths[0]:.2f} - {self.wavelengths[-1]:.2f}",
            f"光谱数据形状: {self.spectra.shape}",
            f"操作历史记录数: {len(self.history)}"
        ]
        
        return "\n".join(summary)
    
    def __str__(self):
        """字符串表示"""
        return self.get_summary()
    
    def __repr__(self):
        """返回对象的字符串表示"""
        return self.__str__()
        
    def batch_process(self, process_func, **kwargs):
        """
        批量处理数据集中的所有光谱
        
        参数:
            process_func: 处理函数，接收一个光谱作为输入，返回处理后的光谱
            **kwargs: 传递给处理函数的参数
            
        返回:
            processed_dataset: 处理后的新数据集
        """
        if self.spectra is None:
            raise ValueError("没有数据可处理")
        
        # 创建新的数据集对象
        processed_dataset = SpectrumDataset(name=f"{self.name}_processed")
        processed_dataset.wavelengths = self.wavelengths.copy()
        processed_dataset.sample_names = self.sample_names.copy() if self.sample_names else None
        processed_dataset.metadata = self.metadata.copy()
        
        # 创建空数组来存储处理后的光谱
        processed_spectra = np.zeros_like(self.spectra)
        
        # 逐个处理光谱
        for i in range(self.spectra.shape[0]):
            if isinstance(process_func, list):
                # 如果是多个处理函数的列表，则顺序应用
                spectrum = self.spectra[i].copy()
                for func in process_func:
                    spectrum = func(spectrum.reshape(1, -1), **kwargs)[0]
                processed_spectra[i] = spectrum
            else:
                # 单个处理函数
                processed_spectra[i] = process_func(self.spectra[i].reshape(1, -1), **kwargs)[0]
        
        # 设置处理后的光谱
        processed_dataset.spectra = processed_spectra
        
        # 添加处理历史
        func_name = process_func.__name__ if not isinstance(process_func, list) else "多步处理"
        kwargs_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
        processed_dataset._add_history(f"批量处理: {func_name}({kwargs_str})")
        
        return processed_dataset
    
    def split_dataset(self, train_ratio=0.7, random_seed=None):
        """
        将数据集分为训练集和测试集
        
        参数:
            train_ratio: 训练集占比
            random_seed: 随机种子，用于重现性
            
        返回:
            train_dataset: 训练集
            test_dataset: 测试集
        """
        if self.spectra is None:
            raise ValueError("没有数据可分割")
        
        # 设置随机种子
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # 获取样本数量
        n_samples = self.spectra.shape[0]
        
        # 计算训练集大小
        n_train = int(n_samples * train_ratio)
        
        # 随机打乱索引
        indices = np.random.permutation(n_samples)
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]
        
        # 创建训练集
        train_dataset = SpectrumDataset(name=f"{self.name}_train")
        train_dataset.wavelengths = self.wavelengths.copy()
        train_dataset.spectra = self.spectra[train_indices]
        train_dataset.metadata = self.metadata.copy()
        
        if self.sample_names:
            train_dataset.sample_names = [self.sample_names[i] for i in train_indices]
        
        # 创建测试集
        test_dataset = SpectrumDataset(name=f"{self.name}_test")
        test_dataset.wavelengths = self.wavelengths.copy()
        test_dataset.spectra = self.spectra[test_indices]
        test_dataset.metadata = self.metadata.copy()
        
        if self.sample_names:
            test_dataset.sample_names = [self.sample_names[i] for i in test_indices]
        
        # 添加历史记录
        train_dataset._add_history(f"从 {self.name} 创建训练集 (比例: {train_ratio})")
        test_dataset._add_history(f"从 {self.name} 创建测试集 (比例: {1-train_ratio})")
        
        return train_dataset, test_dataset
    
    def load_from_corn_file(self, file_path, **kwargs):
        """
        从玉米光谱数据文件加载光谱数据和目标变量
        
        参数:
            file_path: 文件路径
            **kwargs: 传递给读取函数的参数
            
        返回:
            self: 支持链式调用
        """
        try:
            from src.data.io import read_corn_data
            wavelengths, spectra, target, meta_info = read_corn_data(file_path, **kwargs)
            
            self.wavelengths = wavelengths
            self.spectra = spectra
            self.target = target
            
            # 提取样本名称（如果有）
            if 'sample_names' in meta_info and meta_info['sample_names']:
                self.sample_names = meta_info['sample_names']
            else:
                self.sample_names = [f"Sample_{i+1}" for i in range(spectra.shape[0])]
            
            # 设置目标变量名称
            if 'target_name' in meta_info:
                self.target_name = meta_info['target_name']
            
            # 更新元数据
            self.metadata.update(meta_info)
            
            # 记录操作历史
            self._add_history("从玉米数据文件加载数据", {"file": file_path})
            
            return self
        except Exception as e:
            raise IOError(f"加载玉米数据文件失败: {e}")
    
    def get_sample_count(self):
        """
        获取样本数量
        
        返回:
            count: 样本数量
        """
        if self.spectra is None:
            return 0
        return self.spectra.shape[0]
    
    def get_wavelength_range(self):
        """
        获取波长范围
        
        返回:
            wavelength_range: 波长范围(最小值, 最大值)的元组
        """
        if self.wavelengths is None:
            return (0, 0)
        return (float(min(self.wavelengths)), float(max(self.wavelengths)))
    
    def get_data_shape(self):
        """
        获取光谱数据的形状

        返回:
            shape: 数据形状的元组 (样本数, 波长点数)
        """
        if self.spectra is None:
            return (0, 0)
        return self.spectra.shape

    def get_data(self):
        """
        获取光谱数据

        返回:
            data: 光谱数据数组
        """
        return self.spectra

    def get_wavelengths(self):
        """
        获取波长数组

        返回:
            wavelengths: 波长数组
        """
        return self.wavelengths

    def get_sample_ids(self):
        """
        获取样本ID列表

        返回:
            sample_ids: 样本ID列表
        """
        return self.sample_names

    def get_target(self):
        """
        获取目标变量数据

        返回:
            target: 目标变量数据
        """
        return self.target

    def get_target_name(self):
        """
        获取目标变量名称

        返回:
            target_name: 目标变量名称
        """
        return self.target_name

    def get_additional_targets(self):
        """
        获取额外的目标变量数据

        返回:
            additional_targets: 额外的目标变量字典
        """
        return self.metadata.get('additional_targets', {})

    def is_empty(self):
        """
        检查数据集是否为空

        返回:
            empty: 如果数据集为空则返回True，否则返回False
        """
        return self.spectra is None or self.spectra.shape[0] == 0 