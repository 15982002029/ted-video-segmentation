#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TED视频分割处理管道 - YOLO+MediaPipe版本
基于YOLO人体检测和MediaPipe姿态估计的视频分割系统
"""

import os
import yaml
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2

# 导入工具和方法
from utils.smart_downloader import SmartVideoDownloader
from methods.yolo_mediapipe import YOLOMediaPipeSegmenter

class VideoSegmentationPipeline:
    """视频分割处理管道 - 专注于YOLO+MediaPipe方法"""
    
    def __init__(self, config_file: str):
        """初始化视频分割管道"""
        # 加载配置
        self.config = self._load_config(config_file)
        
        # 创建输出目录结构
        self.base_output_dir = Path("results")
        self.base_output_dir.mkdir(exist_ok=True)
        
        # 统一的视频片段文件夹
        self.segments_dir = self.base_output_dir / "video_segments"
        self.segments_dir.mkdir(exist_ok=True)
        
        # 报告文件夹 - 只包含一个报告文件
        self.reports_dir = self.base_output_dir / "reports"
        self.reports_dir.mkdir(exist_ok=True)
        
        # 设置日志
        self._setup_logging()
        
        # 初始化组件
        self.downloads_dir = Path("downloads")
        self.downloads_dir.mkdir(exist_ok=True)
        self.downloader = SmartVideoDownloader(output_dir=str(self.downloads_dir))
        
        # 初始化方法 - 只使用YOLO+MediaPipe
        self.methods = self._initialize_methods()
        
        # 全局片段计数器
        self.global_segment_counter = self._get_next_segment_number()
        
        # 报告文件路径
        self.report_file = self.reports_dir / "video_segmentation_report.md"
        
        # echomimicv2输出文件跟踪
        self.current_echomimicv2_file = None
        
        print(f"📁 输出目录结构:")
        print(f"   🎬 视频片段: {self.segments_dir}")
        print(f"   📊 报告文件: {self.report_file}")
        print(f"   🔢 当前片段编号: {self.global_segment_counter}")
    
    def _get_next_segment_number(self) -> int:
        """获取下一个片段编号"""
        # 查找现有的片段文件，获取最大编号
        max_number = 0
        for file in self.segments_dir.glob("*.mp4"):
            try:
                # 从文件名中提取编号，格式如: 1_yolo_mediapipe_1_quality_0.739_duration_5.2s.mp4
                parts = file.stem.split('_')
                if len(parts) >= 4 and parts[0].isdigit():
                    number = int(parts[0])
                    max_number = max(max_number, number)
            except:
                continue
        
        return max_number + 1
    
    def _initialize_methods(self) -> Dict:
        """初始化处理方法 - 只使用YOLO+MediaPipe"""
        methods = {}
        
        # 只初始化YOLO+MediaPipe方法
        try:
            from methods.yolo_mediapipe import YOLOMediaPipeSegmenter  # 修复类名
            # 合并需要的配置段
            yolo_config = self.config['yolo_mediapipe'].copy()
            # 直接合并general参数到顶层
            yolo_config.update(self.config['general'])
            methods['yolo_mediapipe'] = YOLOMediaPipeSegmenter(yolo_config)
            print("✅ YOLO+MediaPipe方法已初始化")
        except Exception as e:
            print(f"⚠️ YOLO+MediaPipe方法初始化失败: {e}")
        
        return methods
    
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"✅ 配置文件加载成功: {config_path}")
            return config
        except Exception as e:
            print(f"❌ 配置文件加载失败: {e}")
            # 返回默认配置
            return {
                'general': {'frame_skip': 15, 'min_segment_length': 60},  # 修改为60帧（约2.5秒）
                'yolo_mediapipe': {
                    'yolo_confidence_threshold': 0.5,
                    'quality_threshold': 0.4,
                    'segment_quality_threshold': 0.55
                }
            }
    
    def _setup_logging(self):
        """设置日志"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        log_config = self.config.get('logging', {})
        log_level = log_config.get('log_level', 'INFO')
        log_file = log_config.get('log_file', 'logs/segmentation.log')
        
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("日志系统初始化完成")
    
    def initialize_methods(self) -> List[str]:
        """初始化处理方法 - 只初始化YOLO+MediaPipe"""
        print("🔧 初始化YOLO+MediaPipe方法...")
        
        try:
            # 获取YOLO+MediaPipe配置，并合并general配置
            mediapipe_config = self.config['yolo_mediapipe'].copy()
            mediapipe_config.update(self.config['general'])
            
            # 初始化YOLO+MediaPipe分割器
            self.methods['yolo_mediapipe'] = YOLOMediaPipeSegmenter(mediapipe_config)
            
            print("✅ YOLO+MediaPipe方法初始化成功")
            return ['yolo_mediapipe']
            
        except Exception as e:
            print(f"❌ YOLO+MediaPipe方法初始化失败: {e}")
            self.logger.error(f"方法初始化失败: {e}")
            return []
    
    def process_input_file(self, input_file: str, max_videos: Optional[int] = None, 
                          echomimicv2_output_file: Optional[str] = None) -> Dict:
        """从输入文件处理视频 - 支持简单URL列表格式"""
        print(f"\n📋 从输入文件处理: {input_file}")
        
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"输入文件不存在: {input_file}")
        
        # 读取输入文件
        video_entries = self._read_input_file(input_file, max_videos)
        print(f"📊 读取到 {len(video_entries)} 个视频条目")
        
        if not video_entries:
            print("⚠️ 输入文件中没有有效的视频条目")
            return {}
        
        # 注释掉基于下载文件的跳过逻辑，改为完全基于filtered_ted_segments.txt
        # video_entries = self._skip_downloaded_videos(video_entries)
        
        # 初始化方法
        available_methods = self.initialize_methods()
        
        # 设置echomimicv2输出文件
        if echomimicv2_output_file:
            self.current_echomimicv2_file = echomimicv2_output_file
            print(f"📄 将增量写入echomimicv2文件: {echomimicv2_output_file}")
        
        # 处理每个视频
        all_results = {}
        total_processing_time = 0
        
        for video_index, entry in enumerate(video_entries, 1):
            print(f"\n{'='*80}")
            print(f"🎬 处理第 {video_index} 个视频")
            print(f"   URL: {entry['url']}")
            print(f"   将分析完整视频并提取符合标准的时间段")
            print(f"{'='*80}")
            
            # 下载完整视频
            video_path = self._download_video_segment(entry, video_index)
            if not video_path:
                print(f"❌ 第 {video_index} 个视频下载失败，跳过")
                continue
            
            # 处理视频
            video_start_time = time.time()
            video_results = self._process_single_video(video_path, video_index, available_methods)
            video_processing_time = time.time() - video_start_time
            total_processing_time += video_processing_time
            
            print(f"⏱️ 第 {video_index} 个视频处理时间: {video_processing_time:.2f}秒")
            
            # 保存结果
            all_results[f"video_{video_index}"] = {
                'entry_info': entry,
                'video_path': video_path,
                'processing_time': video_processing_time,
                'methods_results': video_results
            }
            
            # 🔄 每处理完一个视频就立即更新报告和echomimicv2文件
            print(f"📊 更新第 {video_index} 个视频的处理报告...")
            self.save_analysis_results(all_results)
            print(f"✅ 第 {video_index} 个视频报告已更新")
            
            # 🔄 增量写入echomimicv2文件
            print(f"   当前echomimicv2文件: {self.current_echomimicv2_file}")
            if self.current_echomimicv2_file:
                is_first = video_index == 1
                segment_count = self.append_echomimicv2_output(
                    all_results[f"video_{video_index}"], 
                    self.current_echomimicv2_file, 
                    is_first
                )
                print(f"✅ 第 {video_index} 个视频的 {segment_count} 个时间段已写入文件")
            else:
                print(f"⚠️ 当前echomimicv2文件未设置，跳过增量写入")
        
        # 总体统计
        print(f"\n{'='*80}")
        print("🎯 处理完成！总体统计")
        print(f"{'='*80}")
        print(f"📊 处理视频数: {len(all_results)}")
        print(f"⏱️ 总处理时间: {total_processing_time:.2f}秒")
        if len(all_results) > 0:
            print(f"⏱️ 平均处理时间: {total_processing_time/len(all_results):.2f}秒/视频")
        else:
            print("⏱️ 平均处理时间: 无数据")
        
        # 统计片段数量
        total_segments = 0
        for video_result in all_results.values():
            for method_name, method_result in video_result['methods_results'].items():
                if method_result and 'segments' in method_result:
                    total_segments += len(method_result['segments'])
        
        print(f"🎬 生成片段总数: {total_segments}")
        print(f"📁 输出目录: {self.segments_dir}")
        print(f"📄 最终报告: {self.report_file}")
        
        return all_results
    
    def process_benchmark_file(self, benchmark_file: str, max_videos: Optional[int] = None) -> Dict:
        """从benchmark文件处理视频 - 每处理完一个视频就更新报告"""
        print(f"\n📋 从benchmark文件处理: {benchmark_file}")
        
        if not os.path.exists(benchmark_file):
            raise FileNotFoundError(f"Benchmark文件不存在: {benchmark_file}")
        
        # 读取benchmark文件
        video_entries = self._read_benchmark_file(benchmark_file, max_videos)
        print(f"📊 读取到 {len(video_entries)} 个视频条目")
        
        if not video_entries:
            print("⚠️ benchmark文件中没有有效的视频条目")
            return {}
        
        # 初始化方法
        available_methods = self.initialize_methods()
        
        # 处理每个视频
        all_results = {}
        total_processing_time = 0
        
        for video_index, entry in enumerate(video_entries, 1):
            print(f"\n{'='*80}")
            print(f"🎬 处理第 {video_index} 个完整视频")
            print(f"   URL: {entry['url']}")
            print(f"   注意: 将处理完整视频，时间段参数已忽略")
            print(f"{'='*80}")
            
            # 下载完整视频
            video_path = self._download_video_segment(entry, video_index)
            if not video_path:
                print(f"❌ 第 {video_index} 个视频下载失败，跳过")
                continue
            
            # 处理视频
            video_start_time = time.time()
            video_results = self._process_single_video(video_path, video_index, available_methods)
            video_processing_time = time.time() - video_start_time
            total_processing_time += video_processing_time
            
            print(f"⏱️ 第 {video_index} 个视频处理时间: {video_processing_time:.2f}秒")
            
            # 保存结果
            all_results[f"video_{video_index}"] = {
                'entry_info': entry,
                'video_path': video_path,
                'processing_time': video_processing_time,
                'methods_results': video_results
            }
            
            # 🔄 每处理完一个视频就立即更新报告
            print(f"📊 更新第 {video_index} 个视频的处理报告...")
            self.save_analysis_results(all_results)
            print(f"✅ 第 {video_index} 个视频报告已更新")
        
        # 总体统计
        print(f"\n{'='*80}")
        print("🎯 处理完成！总体统计")
        print(f"{'='*80}")
        print(f"📊 处理视频数: {len(all_results)}")
        print(f"⏱️ 总处理时间: {total_processing_time:.2f}秒")
        if len(all_results) > 0:
            print(f"⏱️ 平均处理时间: {total_processing_time/len(all_results):.2f}秒/视频")
        else:
            print("⏱️ 平均处理时间: 无数据")
        
        # 统计片段数量
        total_segments = 0
        for video_result in all_results.values():
            for method_name, method_result in video_result['methods_results'].items():
                if method_result and 'segments' in method_result:
                    total_segments += len(method_result['segments'])
        
        print(f"🎬 生成片段总数: {total_segments}")
        print(f"📁 输出目录: {self.segments_dir}")
        print(f"📄 最终报告: {self.report_file}")
        
        return all_results
    
    def process_local_videos(self, max_videos: Optional[int] = None, 
                            echomimicv2_output_file: Optional[str] = None) -> Dict:
        """处理downloads文件夹中的现有视频文件"""
        print(f"\n📁 处理downloads文件夹中的本地视频文件")
        
        downloads_dir = Path("downloads")
        if not downloads_dir.exists():
            print(f"❌ downloads文件夹不存在: {downloads_dir}")
            return {}
        
        # 查找所有视频文件
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
        video_files = []
        
        for file_path in downloads_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in video_extensions:
                video_files.append(file_path)
        
        # 按文件名排序以确保一致性
        video_files.sort()
        
        if not video_files:
            print(f"⚠️ downloads文件夹中没有找到视频文件")
            return {}
        
        # 限制处理数量
        if max_videos and len(video_files) > max_videos:
            video_files = video_files[:max_videos]
            print(f"🔢 限制处理前 {max_videos} 个视频文件")
        
        print(f"📊 找到 {len(video_files)} 个视频文件:")
        for i, video_file in enumerate(video_files, 1):
            print(f"   {i}. {video_file.name}")
        
        # 初始化方法
        available_methods = self.initialize_methods()
        
        # 设置echomimicv2输出文件
        if echomimicv2_output_file:
            self.current_echomimicv2_file = echomimicv2_output_file
            print(f"📄 将增量写入echomimicv2文件: {echomimicv2_output_file}")
        
        # 处理每个视频
        all_results = {}
        total_processing_time = 0
        
        for video_index, video_path in enumerate(video_files, 1):
            print(f"\n{'='*80}")
            print(f"🎬 处理第 {video_index} 个本地视频")
            print(f"   文件: {video_path.name}")
            print(f"   路径: {video_path}")
            print(f"{'='*80}")
            
            # 处理视频
            video_start_time = time.time()
            video_results = self._process_single_video(str(video_path), video_index, available_methods)
            video_processing_time = time.time() - video_start_time
            total_processing_time += video_processing_time
            
            print(f"⏱️ 第 {video_index} 个视频处理时间: {video_processing_time:.2f}秒")
            
            # 保存结果
            all_results[f"video_{video_index}"] = {
                'entry_info': {
                    'url': f"本地文件: {video_path.name}",
                    'start_time': '00:00:00.000',
                    'end_time': '完整视频'
                },
                'video_path': str(video_path),
                'processing_time': video_processing_time,
                'methods_results': video_results
            }
            
            # 🔄 每处理完一个视频就立即更新报告和echomimicv2文件
            print(f"📊 更新第 {video_index} 个视频的处理报告...")
            self.save_analysis_results(all_results)
            print(f"✅ 第 {video_index} 个视频报告已更新")
            
            # 🔄 增量写入echomimicv2文件
            print(f"   当前echomimicv2文件: {self.current_echomimicv2_file}")
            if self.current_echomimicv2_file:
                is_first = video_index == 1
                segment_count = self.append_echomimicv2_output(
                    all_results[f"video_{video_index}"], 
                    self.current_echomimicv2_file, 
                    is_first
                )
                print(f"✅ 第 {video_index} 个视频的 {segment_count} 个时间段已写入文件")
            else:
                print(f"⚠️ 当前echomimicv2文件未设置，跳过增量写入")
        
        # 总体统计
        print(f"\n{'='*80}")
        print("🎯 处理完成！总体统计")
        print(f"{'='*80}")
        print(f"📊 处理视频数: {len(all_results)}")
        print(f"⏱️ 总处理时间: {total_processing_time:.2f}秒")
        if len(all_results) > 0:
            print(f"⏱️ 平均处理时间: {total_processing_time/len(all_results):.2f}秒/视频")
        else:
            print("⏱️ 平均处理时间: 无数据")
        
        # 统计片段数量
        total_segments = 0
        for video_result in all_results.values():
            for method_name, method_result in video_result['methods_results'].items():
                if method_result and 'segments' in method_result:
                    total_segments += len(method_result['segments'])
        
        print(f"🎬 生成片段总数: {total_segments}")
        print(f"📁 输出目录: {self.segments_dir}")
        print(f"📄 最终报告: {self.report_file}")
        
        return all_results
    
    def _read_input_file(self, input_file: str, max_videos: Optional[int]) -> List[Dict]:
        """读取输入文件 - 支持简单URL列表（每行一个URL）"""
        video_entries = []
        
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            # 跳过空行和注释行
            if not line or line.startswith('#'):
                continue
            
            # 检查是否是标题行
            if line.upper().startswith('URL'):
                continue
                
            # 处理URL
            if line.startswith('http'):
                entry = {
                    'url': line,
                    'start_time': '00:00:00.000',  # 默认从开始
                    'end_time': '99:59:59.999'     # 默认到结束（实际会被忽略）
                }
                video_entries.append(entry)
                
                if max_videos and len(video_entries) >= max_videos:
                    break
        
        return video_entries
    
    def _skip_downloaded_videos(self, video_entries: List[Dict]) -> List[Dict]:
        """跳过已下载的视频，只处理新视频"""
        print("🔄 检查已下载视频：跳过已处理的视频...")
        
        # 获取已下载的视频文件
        downloaded_videos = []
        for video_file in self.downloads_dir.glob("*.mp4"):
            video_id = video_file.stem.replace('_full_video', '')
            downloaded_videos.append(video_id)
        
        print(f"📁 找到 {len(downloaded_videos)} 个已下载的视频文件")
        
        # 过滤掉已下载的视频
        new_video_entries = []
        skipped_videos = []
        
        for entry in video_entries:
            video_id = self._extract_video_id_from_url(entry['url'])
            if video_id in downloaded_videos:
                skipped_videos.append(entry['url'])
                print(f"⏭️  跳过已下载视频: {video_id}")
            else:
                new_video_entries.append(entry)
                print(f"🆕 新视频待处理: {video_id}")
        
        print(f"📊 过滤结果:")
        print(f"   跳过已下载视频: {len(skipped_videos)} 个")
        print(f"   新视频待处理: {len(new_video_entries)} 个")
        
        if not new_video_entries:
            print("⚠️  所有视频都已下载过，没有新视频需要处理")
        
        return new_video_entries
    
    def _extract_video_id_from_url(self, url: str) -> str:
        """从URL中提取视频ID"""
        if 'youtube.com/watch?v=' in url:
            return url.split('watch?v=')[1].split('&')[0]
        elif 'youtu.be/' in url:
            return url.split('youtu.be/')[1].split('?')[0]
        return ""
    
    def _read_benchmark_file(self, benchmark_file: str, max_videos: Optional[int]) -> List[Dict]:
        """读取benchmark文件 - 只读取URL，处理完整视频"""
        video_entries = []
        
        with open(benchmark_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 跳过标题行
        for line in lines[1:]:
            parts = line.strip().split(',')
            if len(parts) >= 1:  # 只需要URL
                entry = {
                    'url': parts[0].strip(),
                    'start_time': '00:00:00.000',  # 默认从开始
                    'end_time': '99:59:59.999'     # 默认到结束（实际会被忽略）
                }
                video_entries.append(entry)
                
                if max_videos and len(video_entries) >= max_videos:
                    break
        
        return video_entries
    
    def _download_video_segment(self, entry: Dict, video_index: int) -> Optional[str]:
        """下载完整视频"""
        print(f"📥 下载第 {video_index} 个完整视频...")
        
        try:
            video_path = self.downloader.download_video_segment(
                url=entry['url'],
                start_timecode=entry['start_time'],
                end_timecode=entry['end_time'],
                buffer_seconds=0,      # 不添加缓冲，下载完整视频
                max_duration=0         # 不限制时长，下载完整视频
            )
            
            if video_path and os.path.exists(video_path):
                print(f"✅ 下载成功: {os.path.basename(video_path)}")
                return video_path
            else:
                print(f"❌ 下载失败: 文件不存在")
                return None
                
        except Exception as e:
            print(f"❌ 下载异常: {e}")
            return None
    
    def _process_single_video(self, video_path: str, video_index: int, available_methods: List[str]) -> Dict:
        """处理单个视频"""
        video_results = {}
        
        for method_name in available_methods:
            method = self.methods[method_name]
            if method is None:
                continue
            
            print(f"\n🔍 使用 {method_name} 方法处理...")
            
            try:
                # 处理视频
                method_start_time = time.time()
                segments, statistics_info = method.segment_video(video_path)
                method_processing_time = time.time() - method_start_time
                
                print(f"  ✅ {method_name} 处理完成")
                print(f"  🎬 检测到 {len(segments)} 个有效片段")
                print(f"  ⏱️ 处理时间: {method_processing_time:.2f}秒")
                
                # 保存视频片段
                saved_segments = self._save_video_segments(
                    video_path, segments, video_index, method_name
                )
                
                video_results[method_name] = {
                    'segments': segments,
                    'saved_segments': saved_segments,
                    'processing_time': method_processing_time,
                    'segment_count': len(segments),
                    'statistics': statistics_info
                }
                
            except Exception as e:
                print(f"  ❌ {method_name} 方法处理失败: {e}")
                video_results[method_name] = {
                    'error': str(e),
                    'segments': [],
                    'saved_segments': [],
                    'processing_time': 0,
                    'segment_count': 0
                }
        
        return video_results
    
    def _save_video_segments(self, video_path: str, segments: List[Dict], 
                           video_index: int, method_name: str) -> List[str]:
        """保存视频片段 - 使用全局编号"""
        if not segments:
            return []
        
        print(f"  💾 保存 {len(segments)} 个 {method_name} 片段...")
        saved_files = []
        
        # 获取视频信息
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)  # 保留浮点精度，避免int()转换
        cap.release()
        
        for segment_index, segment in enumerate(segments, 1):
            try:
                # 计算片段信息
                start_time = segment.get('start_time', 0)
                end_time = segment.get('end_time', 0)
                duration = end_time - start_time
                
                # 计算质量分数
                if 'avg_quality' in segment:
                    quality = segment['avg_quality']
                elif 'quality_scores' in segment and segment['quality_scores']:
                    quality = sum(segment['quality_scores']) / len(segment['quality_scores'])
                else:
                    quality = 0.5  # 默认质量
                
                # 使用全局编号生成文件名
                global_segment_number = self.global_segment_counter
                filename = f"{global_segment_number}_yolo_mediapipe_{segment_index}_quality_{quality:.3f}_duration_{duration:.1f}s.mp4"
                
                output_path = self.segments_dir / filename
                
                # 🔍 详细的保存信息日志
                print(f"    📊 片段 {global_segment_number} 保存信息:")
                print(f"       ⏰ 时间范围: {start_time:.2f}s - {end_time:.2f}s")
                print(f"       ⏱️ 时长: {duration:.2f}s")
                print(f"       🎯 质量分数: {quality:.3f}")
                print(f"       📁 文件名: {filename}")
                
                # 使用ffmpeg切割视频
                self._extract_video_segment(video_path, start_time, end_time, str(output_path))
                
                if output_path.exists() and output_path.stat().st_size > 0:
                    saved_files.append(filename)  # 只保存文件名，不包含路径
                    print(f"    ✅ 保存片段 {global_segment_number} 成功: {filename}")
                    # 更新全局计数器
                    self.global_segment_counter += 1
                else:
                    print(f"    ❌ 保存片段 {global_segment_number} 失败")
                
            except Exception as e:
                print(f"    ❌ 保存片段 {self.global_segment_counter} 异常: {e}")
        
        return saved_files
    
    def _extract_video_segment(self, input_path: str, start_time: float, 
                             end_time: float, output_path: str):
        """使用ffmpeg提取视频片段 - 修复时长差异问题"""
        import subprocess
        
        # 🔧 修复时长问题：使用精确的帧级切割
        # 方法1：使用两步法确保精确时长
        duration = end_time - start_time
        
        # 构建ffmpeg命令 - 使用精确切割
        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-ss', str(start_time),  # 在输入后指定，确保精度
            '-t', str(duration),     # 精确的时长
            '-c:v', 'libx264',       # 重新编码以确保精确切割
            '-c:a', 'aac',           # 音频编码
            '-preset', 'fast',       # 快速编码
            '-avoid_negative_ts', 'make_zero',
            '-y',  # 覆盖输出文件
            output_path
        ]
        
        try:
            # 使用项目中的FFmpeg
            ffmpeg_path = Path("ffmpeg/ffmpeg.exe")
            if ffmpeg_path.exists():
                cmd[0] = str(ffmpeg_path)
            else:
                # 备用：尝试系统FFmpeg
                cmd[0] = 'ffmpeg'
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                print(f"      ❌ ffmpeg失败 (返回码: {result.returncode})")
                print(f"      错误信息: {result.stderr}")
                raise Exception(f"FFmpeg处理失败: {result.stderr}")
            else:
                print(f"      ✅ ffmpeg执行成功")
                
        except subprocess.TimeoutExpired:
            print(f"      ❌ ffmpeg超时")
            raise Exception("FFmpeg处理超时")
        except FileNotFoundError:
            print(f"      ❌ 未找到ffmpeg")
            raise Exception("FFmpeg不可用")
    
    def save_analysis_results(self, all_results: Dict, output_filename: Optional[str] = None):
        """保存分析结果 - 持续更新单一报告文件，包含详细检测信息"""
        # 时间格式转换函数
        def seconds_to_time_format(seconds):
            """将秒数转换为 HH:MM:SS 格式"""
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        
        # 生成新的报告内容
        new_report_content = self._generate_markdown_report(all_results, seconds_to_time_format)
        
        # 🔍 生成详细检测报告
        detailed_report_content = self._generate_detailed_detection_report(all_results, seconds_to_time_format)
        detailed_report_file = self.reports_dir / "detailed_detection_report.md"
        
        # 保存到固定报告文件
        try:
            with open(self.report_file, 'w', encoding='utf-8') as f:
                f.write(new_report_content)
            
            # 保存详细检测报告
            with open(detailed_report_file, 'w', encoding='utf-8') as f:
                f.write(detailed_report_content)
                
            print(f"📊 主报告已更新: {self.report_file}")
            print(f"📊 详细报告已生成: {detailed_report_file}")
            return str(self.report_file)
        except Exception as e:
            print(f"❌ 保存报告失败: {e}")
            return None
    
    def _generate_markdown_report(self, all_results: Dict, time_formatter) -> str:
        """生成Markdown格式的报告"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 统计总体信息
        total_videos = len(all_results)
        total_segments = 0
        total_processing_time = 0
        method_stats = {}
        
        for video_result in all_results.values():
            total_processing_time += video_result['processing_time']
            for method_name, method_result in video_result['methods_results'].items():
                if method_name not in method_stats:
                    method_stats[method_name] = {
                        'total_segments': 0,
                        'total_processing_time': 0,
                        'videos_processed': 0,
                        'success_count': 0
                    }
                
                method_stats[method_name]['videos_processed'] += 1
                method_stats[method_name]['total_processing_time'] += method_result.get('processing_time', 0)
                method_stats[method_name]['total_segments'] += method_result.get('segment_count', 0)
                if not method_result.get('error'):
                    method_stats[method_name]['success_count'] += 1
                total_segments += method_result.get('segment_count', 0)
        
        # 生成报告内容
        report = f"""# 🎬 TED视频分割处理报告

**生成时间**: {timestamp}  
**处理视频数**: {total_videos}  
**总片段数**: {total_segments}  
**总处理时间**: {time_formatter(total_processing_time)}

## 📊 总体统计

| 指标 | 数值 |
|------|------|
| 处理视频数 | {total_videos} |
| 生成片段数 | {total_segments} |
| 总处理时间 | {time_formatter(total_processing_time)} |
| 平均处理时间 | {time_formatter(total_processing_time / max(total_videos, 1))} |

## 🔧 方法性能对比

| 方法 | 处理视频数 | 成功数 | 成功率 | 总片段数 | 平均片段数 | 总处理时间 | 平均处理时间 |
|------|------------|--------|--------|----------|------------|------------|--------------|
"""
        
        for method_name, stats in method_stats.items():
            success_rate = (stats['success_count'] / max(stats['videos_processed'], 1)) * 100
            avg_segments = stats['total_segments'] / max(stats['videos_processed'], 1)
            avg_time = stats['total_processing_time'] / max(stats['videos_processed'], 1)
            
            method_display = "YOLO+MediaPipe" if method_name == "yolo_mediapipe" else "YOLO+OpenPose"
            
            report += f"| {method_display} | {stats['videos_processed']} | {stats['success_count']} | {success_rate:.1f}% | {stats['total_segments']} | {avg_segments:.1f} | {time_formatter(stats['total_processing_time'])} | {time_formatter(avg_time)} |\n"
        
        report += "\n## 🎥 视频处理详情\n\n"
        
        for video_key, video_result in all_results.items():
            video_name = Path(video_result['video_path']).stem
            report += f"### 📹 {video_name}\n\n"
            report += f"**处理时间**: {time_formatter(video_result['processing_time'])}\n\n"
            
            for method_name, method_result in video_result['methods_results'].items():
                method_display = "YOLO+MediaPipe" if method_name == "yolo_mediapipe" else "YOLO+OpenPose"
                status = "✅ 成功" if not method_result.get('error') else "❌ 失败"
                
                report += f"#### {method_display} - {status}\n\n"
                
                if not method_result.get('error'):
                    report += f"- **片段数**: {method_result.get('segment_count', 0)}\n"
                    report += f"- **处理时间**: {time_formatter(method_result.get('processing_time', 0))}\n"
                    
                    # 🔍 添加筛选统计信息
                    statistics = method_result.get('statistics', {})
                    if statistics:
                        total_frames = statistics.get('total_frames', 0)
                        stage1_count = statistics.get('stage1_yolo_detection', 0)
                        stage2_count = statistics.get('stage2_geometry_assessment', 0)
                        stage3_count = statistics.get('stage3_mediapipe_detection', 0)
                        stage4_count = statistics.get('stage4_pose_quality', 0)
                        stage5_count = statistics.get('stage5_dwpose_multi_person', 0)
                        stage6_count = statistics.get('stage6_scene_change_detection', 0)
                        
                        pass_rates = statistics.get('stage_pass_rates', {})
                        
                        # 检查是否为六关卡模式
                        is_six_stages = (stage5_count > 0 or stage6_count > 0 or 
                                        'stage5_dwpose_multi_person' in statistics or 
                                        'stage6_scene_change_detection' in statistics)
                        
                        if is_six_stages:
                            report += f"**📊 六关卡筛选统计**:\n"
                        else:
                            report += f"**📊 四关卡筛选统计**:\n"
                        
                        report += f"- **总帧数**: {total_frames}\n"
                        report += f"- **🚪 关卡1 YOLO检测**: {stage1_count} ({pass_rates.get('stage1', 0):.1f}%)\n"
                        report += f"- **🚪 关卡2 几何评估**: {stage2_count} ({pass_rates.get('stage2', 0):.1f}%)\n"
                        report += f"- **🚪 关卡3 MediaPipe检测**: {stage3_count} ({pass_rates.get('stage3', 0):.1f}%)\n"
                        report += f"- **🚪 关卡4 姿态质量**: {stage4_count} ({pass_rates.get('stage4', 0):.1f}%)\n"
                        
                        # 如果是六关卡模式，添加关卡5和关卡6
                        if is_six_stages:
                            report += f"- **🚪 关卡5 DWpose多人检测**: {stage5_count} ({pass_rates.get('stage5', 0):.1f}%)\n"
                            report += f"- **🚪 关卡6 场景切换检测**: {stage6_count} ({pass_rates.get('stage6', 0):.1f}%)\n"
                        
                        report += f"- **🎯 最终通过率**: {pass_rates.get('final', 0):.2f}%\n\n"
                        
                        # 🔍 添加关卡1详细统计 - YOLO人体检测
                        stage1_detailed = statistics.get('stage1_detailed', {})
                        if stage1_detailed.get('total_frames', 0) > 0 or total_frames > 0:
                            report += "**📊 关卡1详细统计** (YOLO人体检测流程):\n"
                            
                            input_frames = total_frames
                            detected_frames = stage1_count
                            no_detection_frames = input_frames - detected_frames
                            
                            detection_rate = (detected_frames / input_frames * 100) if input_frames > 0 else 0
                            
                            report += f"- 💫 **输入总帧数**: {input_frames}帧\n"
                            report += f"- 🎯 **YOLO检测到人体**: {detection_rate:.1f}% ({detected_frames}/{input_frames})\n"
                            report += f"- ❌ **未检测到人体**: {100-detection_rate:.1f}% ({no_detection_frames}/{input_frames}) [被淘汰]\n"
                            
                            # 如果有更详细的YOLO统计
                            if stage1_detailed:
                                confidence_avg = stage1_detailed.get('average_confidence', 0)
                                max_confidence = stage1_detailed.get('max_confidence', 0)
                                min_confidence = stage1_detailed.get('min_confidence', 0)
                                
                                if confidence_avg > 0:
                                    report += f"- 📊 **检测置信度**: 平均{confidence_avg:.3f}, 范围{min_confidence:.3f}-{max_confidence:.3f}\n"
                            
                            report += f"- 🎯 **关卡1通过率**: {detection_rate:.1f}% ({detected_frames}/{input_frames})\n"
                            
                            if no_detection_frames > 0:
                                report += f"- ❌ **主要淘汰原因**: 画面中无人体或人体太小，淘汰{no_detection_frames}帧\n"
                            
                            report += "\n"
                        
                        # 🔍 添加关卡2详细统计 - 按筛选流程顺序展示
                        stage2_detailed = statistics.get('stage2_detailed', {})
                        if stage2_detailed.get('total_detections', 0) > 0:
                            report += "**📊 关卡2详细统计** (逐步筛选流程):\n"
                            
                            # 获取各步骤的实际数据
                            input_frames = stage2_detailed['total_detections']  # 进入关卡2的帧数
                            area_passed = stage2_detailed.get('area_check_passed', 0)
                            height_passed = stage2_detailed.get('height_check_passed', 0) 
                            face_size_passed = stage2_detailed.get('face_size_check_passed', 0)
                            face_width_passed = stage2_detailed.get('face_width_check_passed', 0)
                            face_height_passed = stage2_detailed.get('face_height_check_passed', 0)
                            face_aspect_passed = stage2_detailed.get('face_aspect_check_passed', 0)
                            face_center_passed = stage2_detailed.get('face_center_check_passed', 0)
                            center_pos_passed = stage2_detailed.get('center_position_check_passed', 0)
                            final_passed = stage2_detailed.get('final_passed', 0)
                            
                            # 计算每步的通过率 - 以实际输入数量为基准
                            basic_checks_passed = max(area_passed, height_passed, face_size_passed, face_width_passed, face_height_passed)
                            
                            report += f"- 💫 **进入关卡2**: {input_frames}帧\n"
                            report += f"- ✅ **基础检查通过**: {basic_checks_passed}帧\n"
                            report += f"  - 面积比例: 100.0% ({area_passed}/{basic_checks_passed})\n"
                            report += f"  - 高度比例: 100.0% ({height_passed}/{basic_checks_passed})\n"
                            report += f"  - 面部大小: 100.0% ({face_size_passed}/{basic_checks_passed})\n"
                            report += f"  - 面部宽度: 100.0% ({face_width_passed}/{basic_checks_passed})\n"
                            report += f"  - 面部高度: 100.0% ({face_height_passed}/{basic_checks_passed})\n"
                            
                            # 面部宽高比检查 - 关键筛选点
                            if basic_checks_passed > 0:
                                aspect_rate = (face_aspect_passed / basic_checks_passed * 100)
                                report += f"- 🔥 **面部宽高比检查**: {aspect_rate:.1f}% ({face_aspect_passed}/{basic_checks_passed}) **[主要筛选瓶颈]**\n"
                            
                            # 后续检查 - 基于通过面部宽高比的帧数
                            if face_aspect_passed > 0:
                                center_rate = (face_center_passed / face_aspect_passed * 100) if face_aspect_passed > 0 else 0
                                report += f"- ✅ **面部中心检查**: {center_rate:.1f}% ({face_center_passed}/{face_aspect_passed})\n"
                            
                            # 最终通过率
                            final_rate = (final_passed / input_frames * 100)
                            report += f"- 🎯 **关卡2最终通过**: {final_rate:.1f}% ({final_passed}/{input_frames})\n"
                            
                            # 显示主要淘汰原因
                            if basic_checks_passed > face_aspect_passed:
                                eliminated = basic_checks_passed - face_aspect_passed
                                report += f"- ❌ **主要淘汰原因**: 面部宽高比不合格，淘汰{eliminated}帧\n"
                            
                            report += "\n"
                        
                        # 🔍 添加关卡3详细统计 - MediaPipe姿态检测逐步筛选
                        stage3_detailed = statistics.get('stage3_detailed', {})
                        if stage3_detailed.get('total_checks', 0) > 0:
                            report += "**📊 关卡3详细统计** (MediaPipe姿态检测流程):\n"
                            
                            input_frames = stage3_detailed['total_checks']  # 进入关卡3的帧数
                            upper_passed = stage3_detailed.get('upper_body_check_passed', 0)
                            # 移除肩膀检查相关逻辑 - shoulder_passed = stage3_detailed.get('shoulder_quality_check_passed', 0)
                            lower_passed = stage3_detailed.get('lower_body_check_passed', 0)
                            final_passed = stage3_detailed.get('final_passed', 0)
                            
                            report += f"- 💫 **进入关卡3**: {input_frames}帧\n"
                            
                            # 上半身关键点检查
                            upper_rate = (upper_passed / input_frames * 100) if input_frames > 0 else 0
                            report += f"- 👤 **上半身7关键点检查**: {upper_rate:.1f}% ({upper_passed}/{input_frames})\n"
                            
                            # 移除肩膀质量检查逻辑 - 已删除该检查步骤
                            
                            # 下半身零容忍检查 (并行检查)
                            lower_rate = (lower_passed / input_frames * 100) if input_frames > 0 else 0
                            report += f"- 🚫 **下半身零容忍检查**: {lower_rate:.1f}% ({lower_passed}/{input_frames})\n"
                            
                            # 最终通过率
                            final_rate = (final_passed / input_frames * 100) if input_frames > 0 else 0
                            report += f"- 🎯 **关卡3最终通过**: {final_rate:.1f}% ({final_passed}/{input_frames})\n"
                            
                            # 显示主要失败原因
                            if upper_passed == 0:
                                report += f"- ❌ **主要失败原因**: 上半身关键点不足，无法通过MediaPipe检测\n"
                            
                            report += "\n"
                        
                        # 🔍 添加关卡4详细统计 - 姿态质量评估
                        stage4_detailed = statistics.get('stage4_detailed', {})
                        if stage4_detailed.get('total_checks', 0) > 0:
                            report += "**📊 关卡4详细统计** (姿态质量评估流程):\n"
                            
                            input_frames = stage4_detailed['total_checks']
                            stability_passed = stage4_detailed.get('stability_check_passed', 0)
                            gesture_passed = stage4_detailed.get('gesture_check_passed', 0)
                            final_passed = stage4_detailed.get('final_passed', 0)
                            
                            report += f"- 💫 **进入关卡4**: {input_frames}帧\n"
                            
                            # 姿态质量检查
                            stability_rate = (stability_passed / input_frames * 100) if input_frames > 0 else 0
                            gesture_rate = (gesture_passed / input_frames * 100) if input_frames > 0 else 0
                            final_rate = (final_passed / input_frames * 100) if input_frames > 0 else 0
                            
                            report += f"- 🏃 **身体稳定性检查**: {stability_rate:.1f}% ({stability_passed}/{input_frames})\n"
                            report += f"- 👋 **演讲手势检查**: {gesture_rate:.1f}% ({gesture_passed}/{input_frames})\n"
                            report += f"- 🎯 **关卡4最终通过**: {final_rate:.1f}% ({final_passed}/{input_frames})\n"
                            
                            # 显示主要失败原因
                            if stability_passed < input_frames:
                                failed_stability = input_frames - stability_passed
                                report += f"- ❌ **主要失败原因**: 身体姿态不稳定，淘汰{failed_stability}帧\n"
                            elif gesture_passed < stability_passed:
                                failed_gesture = stability_passed - gesture_passed
                                report += f"- ❌ **主要失败原因**: 缺乏演讲手势，淘汰{failed_gesture}帧\n"
                            
                            report += "\n"
                        
                        # 🔍 添加关卡5详细统计（如果是六关卡模式）- DWPose多人检测
                        if is_six_stages:
                            stage5_detailed = statistics.get('stage5_detailed', {})
                            if stage5_detailed.get('total_frames_processed', 0) > 0:
                                report += "**📊 关卡5详细统计** (DWPose多人检测流程):\n"
                                
                                input_frames = stage5_detailed['total_frames_processed']
                                single_frames = stage5_detailed.get('single_person_frames', 0)
                                multi_frames = stage5_detailed.get('multi_person_frames', 0)
                                no_person_frames = stage5_detailed.get('no_person_frames', 0)
                                failure_frames = stage5_detailed.get('detection_failures', 0)
                                total_persons = stage5_detailed.get('total_persons_detected', 0)
                                
                                report += f"- 💫 **进入关卡5**: {input_frames}帧\n"
                                
                                # 检测结果分类
                                single_rate = (single_frames / input_frames * 100) if input_frames > 0 else 0
                                multi_rate = (multi_frames / input_frames * 100) if input_frames > 0 else 0
                                no_person_rate = (no_person_frames / input_frames * 100) if input_frames > 0 else 0
                                failure_rate = (failure_frames / input_frames * 100) if input_frames > 0 else 0
                                
                                report += f"- ✅ **单人检测**: {single_rate:.1f}% ({single_frames}/{input_frames}) [符合要求]\n"
                                report += f"- 🚫 **多人检测**: {multi_rate:.1f}% ({multi_frames}/{input_frames}) [被淘汰]\n"
                                report += f"- ❓ **未检测到人**: {no_person_rate:.1f}% ({no_person_frames}/{input_frames}) [被淘汰]\n"
                                report += f"- ❌ **检测失败**: {failure_rate:.1f}% ({failure_frames}/{input_frames}) [技术问题]\n"
                                
                                # 平均人数统计
                                avg_persons = total_persons / input_frames if input_frames > 0 else 0
                                report += f"- 📊 **平均检测人数**: {avg_persons:.2f}人/帧\n"
                                
                                # 关卡5通过率 (只有单人帧通过)
                                pass_rate = single_rate
                                report += f"- 🎯 **关卡5通过率**: {pass_rate:.1f}% ({single_frames}/{input_frames})\n"
                                
                                # 显示主要淘汰原因
                                if multi_frames > 0:
                                    report += f"- ❌ **主要淘汰原因**: 检测到多人场景，淘汰{multi_frames}帧\n"
                                elif no_person_frames > 0:
                                    report += f"- ❌ **主要淘汰原因**: 未检测到人体，淘汰{no_person_frames}帧\n"
                                
                                report += "\n"
                            
                            # 🔍 添加关卡6详细统计 - 场景切换检测
                            stage6_detailed = statistics.get('stage6_detailed', {})
                            # 场景切换检测器使用total_frames字段，不是total_frames_processed
                            if stage6_detailed.get('total_frames', 0) > 0:
                                report += "**📊 关卡6详细统计** (场景切换检测流程):\n"
                                
                                input_frames = stage6_detailed['total_frames']
                                scene_changes_detected = stage6_detailed.get('scene_changes_detected', 0)
                                
                                # 计算稳定帧数（总帧数 - 场景切换帧数）
                                stable_frames = input_frames - scene_changes_detected
                                
                                avg_similarity = stage6_detailed.get('average_similarity', 0)
                                min_similarity = stage6_detailed.get('min_similarity', 0)
                                max_similarity = stage6_detailed.get('max_similarity', 0)
                                
                                report += f"- 💫 **进入关卡6**: {input_frames}帧\n"
                                
                                # 场景检测结果分类
                                stable_rate = (stable_frames / input_frames * 100) if input_frames > 0 else 0
                                change_rate = (scene_changes_detected / input_frames * 100) if input_frames > 0 else 0
                                
                                report += f"- ✅ **场景稳定**: {stable_rate:.1f}% ({stable_frames}/{input_frames}) [符合要求]\n"
                                report += f"- 🔄 **场景切换**: {change_rate:.1f}% ({scene_changes_detected}/{input_frames}) [被淘汰]\n"
                                
                                # 相似度统计
                                report += f"- 📊 **相似度统计**: 平均{avg_similarity:.3f}, 范围{min_similarity:.3f}-{max_similarity:.3f}\n"
                                report += f"- 🎬 **场景切换次数**: {scene_changes_detected}次\n"
                                
                                # 关卡6通过率 (只有稳定场景通过)
                                pass_rate = stable_rate
                                report += f"- 🎯 **关卡6通过率**: {pass_rate:.1f}% ({stable_frames}/{input_frames})\n"
                                
                                # 显示主要淘汰原因
                                if scene_changes_detected > 0:
                                    report += f"- ❌ **主要淘汰原因**: 检测到场景切换，淘汰{scene_changes_detected}帧\n"
                                
                                report += "\n"
                    
                    if method_result.get('segments'):
                        report += "**片段详情**:\n"
                        for i, segment in enumerate(method_result['segments'], 1):
                            start_time = segment.get('start_time', 0)
                            end_time = segment.get('end_time', 0)
                            duration = end_time - start_time
                            quality = segment.get('avg_quality', segment.get('quality', 0))
                            
                            # 获取对应的保存文件名
                            saved_files = method_result.get('saved_segments', [])
                            filename = saved_files[i-1] if i-1 < len(saved_files) else "未保存"
                            
                            report += f"  - 片段{i}: {time_formatter(start_time)} - {time_formatter(end_time)} (时长: {time_formatter(duration)}, 质量: {quality:.3f}, 文件: {filename})\n"
                else:
                    report += f"- **错误**: {method_result['error']}\n"
                
                report += "\n"
        
        report += f"""## 📁 输出文件

- **视频片段文件夹**: `results/video_segments/`
- **报告文件**: `results/reports/`

## 📝 说明

- **质量分数**: 0.8以上为高质量，0.6-0.8为中等质量，0.6以下为低质量
- **成功率**: 成功处理的视频占总视频数的比例
- **处理时间**: 包含视频下载、检测、分割和保存的完整时间
- **片段时长**: 每个片段的时间范围，格式为 HH:MM:SS

---
*报告由TED视频分割处理管道自动生成*
"""
        
        return report

    def append_echomimicv2_output(self, video_result: Dict, output_file: str, is_first_video: bool = False) -> int:
        """增量写入echomimicv2格式的输出文件 - 每处理完一个视频就写入"""
        output_path = Path(output_file)
        entry_info = video_result.get('entry_info', {})
        original_url = entry_info.get('url', '')
        
        # 获取YOLO+MediaPipe方法的结果
        methods_results = video_result.get('methods_results', {})
        yolo_mediapipe_result = methods_results.get('yolo_mediapipe', {})
        
        segment_count = 0
        video_total_duration = 0  # 当前视频的总时长
        
        if yolo_mediapipe_result and not yolo_mediapipe_result.get('error'):
            segments = yolo_mediapipe_result.get('segments', [])
            
            if segments:
                # 决定写入模式：如果是第一个视频且文件不存在，使用写入模式；否则使用追加模式
                file_exists = output_path.exists()
                
                if is_first_video and not file_exists:
                    mode = 'w'
                    write_header = True
                else:
                    mode = 'a'
                    write_header = False
                
                with open(output_path, mode, encoding='utf-8') as f:
                    # 如果需要，写入标题行
                    if write_header:
                        f.write("URL,Start Timecode,End Timecode\n")
                    
                    # 写入该视频的所有片段
                    for segment in segments:
                        start_time = segment.get('start_time', 0)
                        end_time = segment.get('end_time', 0)
                        duration = end_time - start_time
                        video_total_duration += duration
                        
                        # 转换为echomimicv2格式的时间码 (HH:MM:SS.mmm)
                        start_timecode = self._seconds_to_timecode(start_time)
                        end_timecode = self._seconds_to_timecode(end_time)
                        
                        f.write(f"{original_url},{start_timecode},{end_timecode}\n")
                        segment_count += 1
                
                if file_exists and is_first_video:
                    print(f"📄 已追加 {segment_count} 个时间段到现有文件 {output_file}")
                else:
                    print(f"📄 已{'创建并' if write_header else '追加'}写入 {segment_count} 个时间段到 {output_file}")
                
                # 显示当前视频的时长统计
                if video_total_duration > 0:
                    print(f"📊 当前视频片段总时长: {video_total_duration:.2f}秒 ({video_total_duration/60:.1f}分钟)")
                    
                    # 计算并更新总体时长统计（基于整个文件）
                    self._update_total_duration_statistics()
        
        return segment_count

    def generate_echomimicv2_output(self, all_results: Dict, output_file: str) -> str:
        """生成echomimicv2格式的输出文件 - 批量模式（保留兼容性）"""
        print(f"\n📄 生成echomimicv2格式输出文件: {output_file}")
        
        output_entries = []
        
        # 遍历所有处理结果
        for video_key, video_result in all_results.items():
            entry_info = video_result.get('entry_info', {})
            original_url = entry_info.get('url', '')
            
            # 获取YOLO+MediaPipe方法的结果
            methods_results = video_result.get('methods_results', {})
            yolo_mediapipe_result = methods_results.get('yolo_mediapipe', {})
            
            if yolo_mediapipe_result and not yolo_mediapipe_result.get('error'):
                segments = yolo_mediapipe_result.get('segments', [])
                
                # 为每个片段创建一个条目
                for segment in segments:
                    start_time = segment.get('start_time', 0)
                    end_time = segment.get('end_time', 0)
                    
                    # 转换为echomimicv2格式的时间码 (HH:MM:SS.mmm)
                    start_timecode = self._seconds_to_timecode(start_time)
                    end_timecode = self._seconds_to_timecode(end_time)
                    
                    output_entries.append({
                        'url': original_url,
                        'start_timecode': start_timecode,
                        'end_timecode': end_timecode
                    })
        
        # 写入输出文件
        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            # 写入标题行
            f.write("URL,Start Timecode,End Timecode\n")
            
            # 写入数据行
            for entry in output_entries:
                f.write(f"{entry['url']},{entry['start_timecode']},{entry['end_timecode']}\n")
        
        print(f"✅ echomimicv2格式文件已生成")
        print(f"📊 总共生成 {len(output_entries)} 个时间段条目")
        
        return str(output_path)
    
    def _seconds_to_timecode(self, seconds: float) -> str:
        """将秒数转换为echomimicv2格式的时间码 (HH:MM:SS.mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"
    
    def _update_report_total_duration(self, total_duration: float):
        """更新报告文件中的数据集总时长统计"""
        try:
            report_path = Path(self.report_file)
            if report_path.exists():
                # 读取现有报告内容
                with open(report_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 计算片段数量
                segment_count = 0
                segments_file = Path("filtered_ted_segments.txt")
                if segments_file.exists():
                    with open(segments_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip() and not line.startswith('URL'):
                                segment_count += 1
                
                # 查找并更新总时长信息
                import re
                
                # 新的统计信息（多行）
                avg_duration = total_duration / segment_count if segment_count > 0 else 0
                new_stats_section = f"""## 📊 数据集统计

- **总片段数量**: {segment_count} 个
- **总时长**: {total_duration:.2f}秒 ({total_duration/60:.1f}分钟, {total_duration/3600:.2f}小时)
- **平均片段时长**: {avg_duration:.1f}秒

---
"""
                
                # 查找现有的统计部分
                stats_pattern = r'## 📊 数据集统计.*?---\n'
                
                if re.search(stats_pattern, content, re.DOTALL):
                    # 如果找到现有统计部分，替换它
                    content = re.sub(stats_pattern, new_stats_section, content, flags=re.DOTALL)
                else:
                    # 如果没有找到，在报告开头添加
                    lines = content.split('\n')
                    # 在第一个标题后插入
                    for i, line in enumerate(lines):
                        if line.startswith('# ') and 'TED视频分割' in line:
                            lines.insert(i + 1, '')  # 空行
                            lines.insert(i + 2, new_stats_section.strip())
                            lines.insert(i + 3, '')  # 空行
                            break
                    content = '\n'.join(lines)
                
                # 写回文件
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                print(f"✅ 已更新报告文件统计: {segment_count}个片段, 总时长{total_duration:.2f}秒")
                
        except Exception as e:
            print(f"⚠️ 更新报告文件失败: {e}")
    
    def _calculate_total_dataset_duration(self) -> float:
        """计算filtered_ted_segments.txt中所有片段的总时长"""
        total_duration = 0.0
        segments_file = Path("filtered_ted_segments.txt")
        
        if not segments_file.exists():
            return 0.0
        
        try:
            with open(segments_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('URL') and ',' in line:
                        parts = line.split(',')
                        if len(parts) >= 3:
                            start_timecode = parts[1].strip()
                            end_timecode = parts[2].strip()
                            
                            # 转换时间码为秒数
                            start_seconds = self._timecode_to_seconds(start_timecode)
                            end_seconds = self._timecode_to_seconds(end_timecode)
                            
                            if start_seconds >= 0 and end_seconds >= 0:
                                duration = end_seconds - start_seconds
                                if duration > 0:
                                    total_duration += duration
        
        except Exception as e:
            print(f"⚠️ 计算总时长失败: {e}")
        
        return total_duration
    
    def _timecode_to_seconds(self, timecode: str) -> float:
        """将时间码(HH:MM:SS.mmm)转换为秒数"""
        try:
            parts = timecode.split(':')
            if len(parts) == 3:
                hours = int(parts[0])
                minutes = int(parts[1])
                seconds = float(parts[2])
                return hours * 3600 + minutes * 60 + seconds
        except:
            pass
        return -1
    
    def _update_total_duration_statistics(self, new_duration: float = None):
        """更新总体时长统计（基于filtered_ted_segments.txt的实际内容）"""
        try:
            # 直接从filtered_ted_segments.txt计算总时长
            total_duration = self._calculate_total_dataset_duration()
            
            # 更新报告文件
            self._update_report_total_duration(total_duration)
            
            print(f"📊 数据集总时长: {total_duration:.2f}秒 ({total_duration/60:.1f}分钟, {total_duration/3600:.1f}小时)")
            
        except Exception as e:
            print(f"⚠️ 更新总时长统计失败: {e}")

    def _generate_detailed_detection_report(self, all_results: Dict, time_formatter) -> str:
        """生成详细的检测和时间分析报告"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""# 🔍 TED视频分割详细检测报告

**生成时间**: {timestamp}

本报告包含每个视频片段的详细检测信息，包括：
- 🎯 检测到的具体帧序列
- ⏰ 帧范围和时间计算详情
- 🎬 实际保存的视频信息
- 📊 时间差异分析

---

"""
        
        # 处理每个视频的详细信息
        for video_key, video_result in all_results.items():
            video_name = Path(video_result['video_path']).stem
            report += f"## 📹 视频: {video_name}\n\n"
            
            for method_name, method_result in video_result['methods_results'].items():
                if method_result.get('error') or not method_result.get('segments'):
                    continue
                    
                method_display = "YOLO+MediaPipe" if method_name == "yolo_mediapipe" else method_name
                report += f"### 🔧 处理方法: {method_display}\n\n"
                report += f"**检测到片段数**: {len(method_result['segments'])}\n\n"
                
                # 详细的片段信息
                for i, segment in enumerate(method_result['segments'], 1):
                    segment_number = i  # 在这个上下文中使用相对编号
                    saved_files = method_result.get('saved_segments', [])
                    filename = saved_files[i-1] if i-1 < len(saved_files) else "未保存"
                    
                    report += f"#### 片段 {segment_number}\n\n"
                    report += f"**文件名**: `{filename}`\n\n"
                    
                    # 🔍 检测信息
                    report += "##### 🎯 检测信息\n"
                    detection_frames = segment.get('detection_frames', [])
                    detection_frame_count = segment.get('detection_frame_count', len(detection_frames))
                    quality_score = segment.get('avg_quality', segment.get('quality_score', 0))
                    
                    report += f"- **检测帧数量**: {detection_frame_count} 帧\n"
                    if detection_frames:
                        report += f"- **检测帧范围**: {min(detection_frames)} - {max(detection_frames)}\n"
                        # 显示前10个检测帧，如果太多就省略
                        if len(detection_frames) <= 10:
                            report += f"- **具体检测帧**: {detection_frames}\n"
                        else:
                            report += f"- **具体检测帧**: {detection_frames[:10]}... (共{len(detection_frames)}帧)\n"
                    report += f"- **质量分数**: {quality_score:.3f}\n\n"
                    
                    # 🔍 帧和时间信息
                    report += "##### ⏰ 帧和时间信息\n"
                    start_frame = segment.get('start_frame', 0)
                    end_frame = segment.get('end_frame', 0)
                    start_time = segment.get('start_time', 0)
                    end_time = segment.get('end_time', 0)
                    calculated_duration = end_time - start_time
                    total_frames = end_frame - start_frame + 1
                    
                    # 假设使用标准FPS
                    estimated_fps = 23.98  # 从日志中看到的FPS
                    
                    report += f"- **视频FPS**: {estimated_fps:.2f} (估计)\n"
                    report += f"- **保存帧范围**: {start_frame} - {end_frame} (共{total_frames}帧)\n"
                    report += f"- **计算时间范围**: {start_time:.2f}s - {end_time:.2f}s\n"
                    report += f"- **计算视频时长**: {calculated_duration:.2f}s\n"
                    
                    # 🔍 时间计算验证
                    frame_based_duration = total_frames / estimated_fps
                    time_calc_diff = abs(calculated_duration - frame_based_duration)
                    
                    report += f"- **基于帧数计算的时长**: {frame_based_duration:.2f}s\n"
                    if time_calc_diff > 0.1:
                        report += f"- **⚠️ 时间计算差异**: {time_calc_diff:.2f}s\n"
                    else:
                        report += f"- **✅ 时间计算**: 一致 (差异: {time_calc_diff:.2f}s)\n"
                    
                    report += "\n"
                
                report += "---\n\n"
        
        # 添加总结分析
        total_segments = sum(len(video_result['methods_results'].get('yolo_mediapipe', {}).get('segments', [])) 
                           for video_result in all_results.values())
        
        report += f"""## 📊 总体分析

**总片段数**: {total_segments}

### 🔍 检测质量分析
- **高质量片段** (质量分数 ≥ 0.8): 推荐用于训练
- **中等质量片段** (质量分数 0.6-0.8): 可考虑使用
- **低质量片段** (质量分数 < 0.6): 建议筛除

### ⏰ 时间计算说明
- **计算时间范围**: 基于帧索引和FPS计算的理论时间
- **实际视频时长**: 保存的视频文件的实际时长
- **时间差异**: 如果差异过大(>10%)，可能存在以下问题：
  - FPS计算不准确
  - 视频编码问题
  - 帧索引计算错误

### 🎯 检测帧说明
- **检测帧**: 系统实际检测到人物姿态的帧
- **保存帧范围**: 最终保存到视频文件的帧范围（可能包含检测帧之间的插值帧）
- **跳帧处理**: 系统每隔N帧检测一次，中间帧通过连续性判断

---
*详细报告由TED视频分割处理管道自动生成*
"""
        
        return report


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='TED视频分割处理管道')
    parser.add_argument('--benchmark', '-b', type=str, 
                       default='echomimicv2_benchmark_url+start_timecode+end_timecode.txt',
                       help='Benchmark文件路径')
    parser.add_argument('--config', '-c', type=str,
                       default='configs/segmentation_parameters.yaml',
                       help='配置文件路径')
    parser.add_argument('--max-videos', '-m', type=int,
                       help='最大处理视频数量')
    parser.add_argument('--output', '-o', type=str,
                       help='输出分析文件名')
    
    args = parser.parse_args()
    
    print("🚀 TED视频分割处理管道")
    print("="*80)
    
    try:
        # 创建处理管道
        pipeline = VideoSegmentationPipeline(args.config)
        
        # 处理benchmark文件
        all_results = pipeline.process_benchmark_file(
            benchmark_file=args.benchmark,
            max_videos=args.max_videos
        )
        
        print("\n✅ 处理完成！")
        print(f"📊 处理视频数: {len(all_results)}")
        print(f"📄 报告文件: {pipeline.report_file}")
        
        return all_results
    
    except Exception as e:
        print(f"\n❌ 管道处理失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 