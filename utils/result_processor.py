#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
结果处理器 - 生成详细的分析报告
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

class ResultProcessor:
    """结果处理器 - 只负责报告生成，不负责视频保存"""
    
    def __init__(self, segments_dir: str = "results/video_segments", 
                 reports_dir: str = "results/reports"):
        """初始化结果处理器"""
        self.segments_dir = Path(segments_dir)
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # 用于跟踪已处理的结果
        self.results_data = []
        self.global_segment_counter = 1
    
    def process_results(self, video_path: str, video_name: str, method_name: str, 
                       segments: List[Dict], original_duration: float = 0, 
                       original_url: str = ""):
        """处理单个视频的结果 - 只生成报告，不保存视频"""
        
        print(f"📊 处理 {method_name} 方法的结果报告")
        
        segment_times = []
        segment_files = []
        segment_details = []
        
        for j, segment in enumerate(segments):
            # 获取片段信息
            start_frame = segment.get('start_frame', 0)
            end_frame = segment.get('end_frame', 0)
            start_time = segment.get('start_time', 0)
            end_time = segment.get('end_time', 0)
            duration = end_time - start_time
            
            # 获取视频信息
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            
            # 获取质量分数
            if 'avg_quality' in segment:
                quality = segment['avg_quality']
            elif 'quality_scores' in segment and segment['quality_scores']:
                quality = sum(segment['quality_scores']) / len(segment['quality_scores'])
            else:
                quality = 0.5
            
            # 生成文件名（应该与实际保存的文件名一致）
            global_segment_number = self.global_segment_counter
            filename = f"{global_segment_number}_yolo_mediapipe_{j+1}_quality_{quality:.3f}_duration_{duration:.1f}s.mp4"
            
            # 检查文件是否实际存在
            video_file_path = self.segments_dir / filename
            actual_duration = 0
            if video_file_path.exists():
                actual_duration = self._get_video_duration(video_file_path)
            
            print(f"    📊 片段 {global_segment_number} 报告信息:")
            print(f"       ⏰ 时间范围: {start_time:.2f}s - {end_time:.2f}s")
            print(f"       ⏱️ 计算时长: {duration:.2f}s")
            print(f"       🎯 质量分数: {quality:.3f}")
            print(f"       📁 预期文件名: {filename}")
            if actual_duration > 0:
                print(f"       🔍 实际视频时长: {actual_duration:.2f}s")
            
            # 格式化时间用于显示
            start_time_str = self._format_time(start_time)
            end_time_str = self._format_time(end_time)
            
            segment_times.append(f"{start_time_str} - {end_time_str}")
            segment_files.append(filename)
            
            # 详细的片段信息记录
            segment_detail = {
                'segment_number': global_segment_number,
                'method_name': method_name,
                'detection_frames': segment.get('detection_frames', []),
                'detection_frame_count': segment.get('detection_frame_count', 0),
                'start_frame': start_frame,
                'end_frame': end_frame,
                'total_frames': end_frame - start_frame + 1,
                'start_time': start_time,
                'end_time': end_time,
                'calculated_duration': duration,
                'actual_duration': actual_duration,
                'quality_score': quality,
                'filename': filename,
                'fps': fps
            }
            segment_details.append(segment_detail)
            
            self.global_segment_counter += 1
        
        # 添加到结果数据
        self._add_result_row(video_name, original_url, original_duration, 
                            method_name, segment_times, segment_files, segment_details)
        
        return segment_times, segment_files, segment_details
    
    def _get_video_duration(self, video_path):
        """获取视频的实际时长"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            
            if fps > 0:
                return frame_count / fps
            else:
                return 0.0
        except Exception as e:
            print(f"❌ 获取视频时长失败: {e}")
            return 0.0
    
    def _format_time(self, seconds):
        """格式化时间，返回分钟:秒.毫秒格式"""
        minutes = int(seconds // 60)
        seconds_remainder = seconds % 60
        return f"{minutes:02d}:{seconds_remainder:05.2f}"
    
    def _add_result_row(self, video_name, original_url, original_duration, 
                        method_name, segment_times, segment_files, segment_details):
        """添加结果行到数据中"""
        row = {
            'video_name': video_name,
            'original_url': original_url,
            'original_duration': original_duration,
            'method_name': method_name,
            'segment_times': segment_times,
            'segment_files': segment_files,
            'segment_details': segment_details,
            'segment_count': len(segment_files)
        }
        self.results_data.append(row)
    
    def generate_reports(self):
        """生成所有报告"""
        if not self.results_data:
            print("⚠️ 没有结果数据，跳过报告生成")
            return
        
        print("📋 生成详细分析报告...")
        
        # 生成详细报告
        detailed_report_path = self.reports_dir / "detailed_detection_report.md"
        self._generate_detailed_report(detailed_report_path)
        
        print(f"✅ 详细报告已生成: {detailed_report_path}")
    
    def _generate_detailed_report(self, report_path):
        """生成详细的检测报告"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 🔍 TED视频分割详细检测报告\n\n")
            f.write(f"**生成时间**: {timestamp}\n\n")
            f.write("本报告包含每个视频片段的详细检测信息，包括：\n")
            f.write("- 🎯 检测到的具体帧序列\n")
            f.write("- ⏰ 帧范围和时间计算详情\n")
            f.write("- 🎬 实际保存的视频信息\n")
            f.write("- 📊 时间差异分析\n\n")
            f.write("---\n\n")
            
            # 按视频分组处理
            videos_data = {}
            for row in self.results_data:
                video_name = row['video_name']
                if video_name not in videos_data:
                    videos_data[video_name] = []
                videos_data[video_name].append(row)
            
            for video_name, video_rows in videos_data.items():
                f.write(f"## 📹 视频: {video_name}\n\n")
                
                for row in video_rows:
                    method_name = row['method_name']
                    segment_details = row['segment_details']
                    
                    f.write(f"### 🔧 处理方法: {method_name.replace('_', '+').upper()}\n\n")
                    f.write(f"**检测到片段数**: {len(segment_details)}\n\n")
                    
                    # 详细的片段信息
                    for i, detail in enumerate(segment_details, 1):
                        f.write(f"#### 片段 {detail['segment_number']}\n\n")
                        f.write(f"**文件名**: `{detail['filename']}`\n\n")
                        
                        # 🔍 检测信息
                        f.write("##### 🎯 检测信息\n")
                        f.write(f"- **检测帧数量**: {detail['detection_frame_count']} 帧\n")
                        if detail['detection_frames']:
                            f.write(f"- **检测帧范围**: {min(detail['detection_frames'])} - {max(detail['detection_frames'])}\n")
                            if len(detail['detection_frames']) <= 10:
                                f.write(f"- **具体检测帧**: {detail['detection_frames']}\n")
                            else:
                                f.write(f"- **具体检测帧**: {detail['detection_frames'][:10]}... (共{len(detail['detection_frames'])}帧)\n")
                        f.write(f"- **质量分数**: {detail['quality_score']:.3f}\n\n")
                        
                        # 🔍 帧和时间信息
                        f.write("##### ⏰ 帧和时间信息\n")
                        f.write(f"- **视频FPS**: {detail['fps']:.2f} (估计)\n")
                        f.write(f"- **保存帧范围**: {detail['start_frame']} - {detail['end_frame']} (共{detail['total_frames']}帧)\n")
                        f.write(f"- **计算时间范围**: {detail['start_time']:.2f}s - {detail['end_time']:.2f}s\n")
                        f.write(f"- **计算视频时长**: {detail['calculated_duration']:.2f}s\n")
                        
                        # 🔍 时间计算验证
                        frame_based_duration = detail['total_frames'] / detail['fps']
                        time_calc_diff = abs(detail['calculated_duration'] - frame_based_duration)
                        
                        f.write(f"- **基于帧数计算的时长**: {frame_based_duration:.2f}s\n")
                        if time_calc_diff > 0.1:
                            f.write(f"- **⚠️ 时间计算差异**: {time_calc_diff:.2f}s\n")
                        else:
                            f.write(f"- **✅ 时间计算**: 一致 (差异: {time_calc_diff:.2f}s)\n")
                        
                        # 实际视频时长信息
                        if detail['actual_duration'] > 0:
                            actual_diff = abs(detail['actual_duration'] - detail['calculated_duration'])
                            f.write(f"- **实际视频时长**: {detail['actual_duration']:.2f}s\n")
                            f.write(f"- **时长差异**: {actual_diff:.2f}s\n")
                        
                        f.write("\n")
                
                f.write("---\n\n")
            
            # 总结分析
            total_segments = sum(len(row['segment_details']) for row in self.results_data)
            f.write("## 📊 总体分析\n\n")
            f.write(f"**总片段数**: {total_segments}\n\n")
            
            f.write("### 🔍 检测质量分析\n")
            f.write("- **高质量片段** (质量分数 ≥ 0.8): 推荐用于训练\n")
            f.write("- **中等质量片段** (质量分数 0.6-0.8): 可考虑使用\n")
            f.write("- **低质量片段** (质量分数 < 0.6): 建议筛除\n\n")
            
            f.write("### ⏰ 时间计算说明\n")
            f.write("- **计算时间范围**: 基于帧索引和FPS计算的理论时间\n")
            f.write("- **实际视频时长**: 保存的视频文件的实际时长\n")
            f.write("- **时间差异**: 如果差异过大(>10%)，可能存在以下问题：\n")
            f.write("  - FPS计算不准确\n")
            f.write("  - 视频编码问题\n")
            f.write("  - 帧索引计算错误\n\n")
            
            f.write("### 🎯 检测帧说明\n")
            f.write("- **检测帧**: 系统实际检测到人物姿态的帧\n")
            f.write("- **保存帧范围**: 最终保存到视频文件的帧范围（可能包含检测帧之间的插值帧）\n")
            f.write("- **跳帧处理**: 系统每隔N帧检测一次，中间帧通过连续性判断\n\n")
            
            f.write("---\n")
            f.write("*详细报告由TED视频分割处理管道自动生成*\n")

if __name__ == "__main__":
    # 测试用例
    processor = ResultProcessor()
    print("📋 结果处理器已初始化")
    print("💡 此模块专注于生成详细的分析报告") 