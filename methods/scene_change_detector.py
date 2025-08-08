#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
场景切换检测器 - 用于检测视频中的场景变化
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
from collections import deque

class SceneChangeDetector:
    """
    场景切换检测器
    - 检测视频中的场景变化
    - 支持多种检测方法
    - 统计检测结果
    """
    
    def __init__(self, config: Dict):
        """初始化场景切换检测器"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 场景切换配置
        self.scene_config = config.get('scene_change_detection', {})
        self.enable_detection = self.scene_config.get('enable_scene_change_detection', True)
        self.detection_method = self.scene_config.get('detection_method', 'hybrid')
        
        # 直方图方法参数
        self.hist_threshold = self.scene_config.get('hist_threshold', 0.3)
        self.hist_bins = self.scene_config.get('hist_bins', 256)
        self.hist_channels = self.scene_config.get('hist_channels', [0, 1, 2])
        
        # 边缘检测方法参数
        self.edge_threshold = self.scene_config.get('edge_threshold', 0.4)
        self.edge_kernel_size = self.scene_config.get('edge_kernel_size', 3)
        self.edge_low_threshold = self.scene_config.get('edge_low_threshold', 50)
        self.edge_high_threshold = self.scene_config.get('edge_high_threshold', 150)
        
        # 混合方法参数
        self.hybrid_weight_hist = self.scene_config.get('hybrid_weight_hist', 0.6)
        self.hybrid_weight_edge = self.scene_config.get('hybrid_weight_edge', 0.4)
        self.hybrid_threshold = self.scene_config.get('hybrid_threshold', 0.35)
        
        # 时间窗口参数
        self.min_scene_duration = self.scene_config.get('min_scene_duration', 2.0)
        self.max_scene_changes_per_segment = self.scene_config.get('max_scene_changes_per_segment', 1)
        
        # 统计信息
        self.stats = {
            'total_frames': 0,
            'scene_changes_detected': 0,
            'histogram_changes': 0,
            'edge_changes': 0,
            'hybrid_changes': 0,
            'average_similarity': 0.0,
            'min_similarity': 1.0,
            'max_similarity': 0.0
        }
        
        # 帧缓存
        self.frame_buffer = deque(maxlen=3)  # 保存最近3帧用于比较
        self.last_frame = None
        
        # 场景变化记录
        self.scene_changes = []
        self.current_scene_start = 0
    
    def detect_scene_change(self, frame: np.ndarray, frame_idx: int, fps: float) -> Dict:
        """
        检测场景切换
        
        Args:
            frame: 当前帧
            frame_idx: 帧索引
            fps: 帧率
            
        Returns:
            检测结果字典
        """
        if not self.enable_detection:
            return {
                'is_scene_change': False,
                'similarity_score': 1.0,
                'change_type': 'none',
                'message': '场景切换检测未启用'
            }
        
        self.stats['total_frames'] += 1
        
        # 第一帧，初始化
        if self.last_frame is None:
            self.last_frame = frame.copy()
            self.current_scene_start = frame_idx
            return {
                'is_scene_change': False,
                'similarity_score': 1.0,
                'change_type': 'none',
                'message': '第一帧，无场景切换'
            }
        
        try:
            # 计算相似度分数
            similarity_scores = {}
            
            # 直方图相似度
            if self.detection_method in ['histogram', 'hybrid']:
                hist_similarity = self._calculate_histogram_similarity(self.last_frame, frame)
                similarity_scores['histogram'] = hist_similarity
                
                if hist_similarity < self.hist_threshold:
                    self.stats['histogram_changes'] += 1
            
            # 边缘检测相似度
            if self.detection_method in ['edge', 'hybrid']:
                edge_similarity = self._calculate_edge_similarity(self.last_frame, frame)
                similarity_scores['edge'] = edge_similarity
                
                if edge_similarity < self.edge_threshold:
                    self.stats['edge_changes'] += 1
            
            # 混合方法
            if self.detection_method == 'hybrid':
                hybrid_similarity = (
                    self.hybrid_weight_hist * similarity_scores['histogram'] +
                    self.hybrid_weight_edge * similarity_scores['edge']
                )
                similarity_scores['hybrid'] = hybrid_similarity
                
                if hybrid_similarity < self.hybrid_threshold:
                    self.stats['hybrid_changes'] += 1
            
            # 确定最终相似度分数
            if self.detection_method == 'histogram':
                final_similarity = similarity_scores['histogram']
                threshold = self.hist_threshold
            elif self.detection_method == 'edge':
                final_similarity = similarity_scores['edge']
                threshold = self.edge_threshold
            else:  # hybrid
                final_similarity = similarity_scores['hybrid']
                threshold = self.hybrid_threshold
            
            # 更新统计信息
            self.stats['average_similarity'] = (
                (self.stats['average_similarity'] * (self.stats['total_frames'] - 1) + final_similarity) 
                / self.stats['total_frames']
            )
            self.stats['min_similarity'] = min(self.stats['min_similarity'], final_similarity)
            self.stats['max_similarity'] = max(self.stats['max_similarity'], final_similarity)
            
            # 判断是否为场景切换
            is_scene_change = final_similarity < threshold
            
            # 检查最小场景持续时间
            if is_scene_change:
                scene_duration = (frame_idx - self.current_scene_start) / fps
                if scene_duration < self.min_scene_duration:
                    is_scene_change = False  # 场景太短，忽略
                else:
                    self.stats['scene_changes_detected'] += 1
                    self.scene_changes.append({
                        'frame_idx': frame_idx,
                        'timestamp': frame_idx / fps,
                        'similarity': final_similarity,
                        'method': self.detection_method
                    })
                    self.current_scene_start = frame_idx
            
            # 更新最后一帧
            self.last_frame = frame.copy()
            
            # 构建返回结果
            if is_scene_change:
                change_type = self.detection_method
                message = f"场景切换检测: 相似度={final_similarity:.3f} < {threshold:.3f}"
            else:
                change_type = 'none'
                message = f"无场景切换: 相似度={final_similarity:.3f} >= {threshold:.3f}"
            
            return {
                'is_scene_change': is_scene_change,
                'similarity_score': final_similarity,
                'change_type': change_type,
                'message': message,
                'similarity_scores': similarity_scores
            }
            
        except Exception as e:
            self.logger.error(f"场景切换检测失败: {e}")
            return {
                'is_scene_change': False,
                'similarity_score': 1.0,
                'change_type': 'error',
                'message': f'检测失败: {e}'
            }
    
    def _calculate_histogram_similarity(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        计算直方图相似度
        
        Args:
            frame1: 第一帧
            frame2: 第二帧
            
        Returns:
            相似度分数 (0-1)
        """
        try:
            # 确保两帧格式一致
            if len(frame1.shape) != len(frame2.shape):
                self.logger.warning("帧格式不一致，跳过直方图计算")
                return 1.0
            
            # 根据图像通道数动态设置直方图通道
            if len(frame1.shape) == 3:
                # 彩色图像：使用所有通道
                channels = list(range(frame1.shape[2]))
                hist_bins = [self.hist_bins] * frame1.shape[2]
                ranges = [0, 256] * frame1.shape[2]
            else:
                # 灰度图像：使用单通道
                channels = [0]
                hist_bins = [self.hist_bins]
                ranges = [0, 256]
            
            # 计算直方图
            hist1 = cv2.calcHist([frame1], channels, None, hist_bins, ranges)
            hist2 = cv2.calcHist([frame2], channels, None, hist_bins, ranges)
            
            # 归一化直方图
            cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
            
            # 计算相似度（使用相关系数）
            similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            
            # 确保结果在0-1范围内
            return max(0, min(1, similarity))
            
        except Exception as e:
            self.logger.error(f"直方图相似度计算失败: {e}")
            return 1.0
    
    def _calculate_edge_similarity(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        计算边缘检测相似度
        
        Args:
            frame1: 第一帧
            frame2: 第二帧
            
        Returns:
            相似度分数 (0-1)
        """
        try:
            # 转换为灰度图
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            
            # 边缘检测
            edges1 = cv2.Canny(gray1, self.edge_low_threshold, self.edge_high_threshold)
            edges2 = cv2.Canny(gray2, self.edge_low_threshold, self.edge_high_threshold)
            
            # 计算边缘密度
            edge_density1 = np.sum(edges1 > 0) / (edges1.shape[0] * edges1.shape[1])
            edge_density2 = np.sum(edges2 > 0) / (edges2.shape[0] * edges2.shape[1])
            
            # 计算边缘密度差异
            density_diff = abs(edge_density1 - edge_density2)
            
            # 转换为相似度分数（差异越小，相似度越高）
            similarity = max(0, 1 - density_diff)
            
            return similarity
            
        except Exception as e:
            self.logger.error(f"边缘相似度计算失败: {e}")
            return 1.0
    
    def get_statistics(self) -> Dict:
        """获取检测统计信息"""
        return self.stats.copy()
    
    def print_statistics(self):
        """打印检测统计信息"""
        if not self.enable_detection:
            print("📊 场景切换检测统计: 未启用")
            return
        
        stats = self.get_statistics()
        
        print(f"\n📊 场景切换检测统计:")
        print(f"   检测方法: {self.detection_method}")
        print(f"   总帧数: {stats['total_frames']}")
        print(f"   场景切换次数: {stats['scene_changes_detected']}")
        print(f"   直方图变化次数: {stats['histogram_changes']}")
        print(f"   边缘变化次数: {stats['edge_changes']}")
        print(f"   混合变化次数: {stats['hybrid_changes']}")
        
        if stats['total_frames'] > 0:
            change_rate = (stats['scene_changes_detected'] / stats['total_frames']) * 100
            print(f"   场景切换率: {change_rate:.2f}%")
            print(f"   平均相似度: {stats['average_similarity']:.3f}")
            print(f"   最小相似度: {stats['min_similarity']:.3f}")
            print(f"   最大相似度: {stats['max_similarity']:.3f}")
    
    def reset_statistics(self):
        """重置统计信息"""
        self.stats = {
            'total_frames': 0,
            'scene_changes_detected': 0,
            'histogram_changes': 0,
            'edge_changes': 0,
            'hybrid_changes': 0,
            'average_similarity': 0.0,
            'min_similarity': 1.0,
            'max_similarity': 0.0
        }
        self.scene_changes = []
        self.current_scene_start = 0
        self.last_frame = None
    
    def get_scene_changes(self) -> List[Dict]:
        """获取场景切换列表"""
        return self.scene_changes.copy()
    
    def cleanup(self):
        """清理资源"""
        self.frame_buffer.clear()
        self.scene_changes.clear()
        self.last_frame = None 