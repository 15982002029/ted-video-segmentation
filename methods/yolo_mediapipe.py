import cv2
import numpy as np
try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("警告: MediaPipe未安装，请运行 pip install mediapipe")

from ultralytics import YOLO
import os
import logging
from typing import List, Dict, Tuple, Optional

class YOLOMediaPipeSegmenter:
    """
    基于YOLO+MediaPipe的视频切片方法
    - YOLO: 快速人体检测
    - MediaPipe: 现代姿态估计（替代OpenPose）
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe未安装，请运行: pip install mediapipe")
        
        # 初始化YOLO模型
        try:
            self.yolo_model = YOLO('yolov8n.pt')
            self.logger.info("✅ YOLO模型加载成功")
        except Exception as e:
            self.logger.error(f"❌ YOLO模型加载失败: {e}")
            raise
        
        # 初始化MediaPipe
        try:
            self.mp_pose = mp_pose
            self.mp_drawing = mp_drawing
            
            # 尝试使用轻量模型，避免网络下载
            try:
                self.pose = self.mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,  # 使用中等复杂度模型，避免heavy模型的网络下载
                    enable_segmentation=False,  # 不需要分割，提高速度
                    min_detection_confidence=self.config['mediapipe_detection_confidence'],
                    min_tracking_confidence=self.config['mediapipe_tracking_confidence']
                )
                self.logger.info("✅ MediaPipe模型加载成功 (model_complexity=1)")
            except Exception as e1:
                # 如果中等模型失败，尝试轻量模型
                self.logger.warning(f"⚠️ 中等模型加载失败，尝试轻量模型: {e1}")
                self.pose = self.mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=0,  # 使用最轻量模型
                    enable_segmentation=False,
                    min_detection_confidence=self.config['mediapipe_detection_confidence'],
                    min_tracking_confidence=self.config['mediapipe_tracking_confidence']
                )
                self.logger.info("✅ MediaPipe模型加载成功 (model_complexity=0, 轻量版)")
                
        except Exception as e:
            self.logger.error(f"❌ MediaPipe模型加载失败: {e}")
            raise
        
        # 统计信息
        self.stats = {
            'total_frames': 0,
            'person_detected_frames': 0,
            'pose_detected_frames': 0,
            'valid_pose_frames': 0,
            # 六关卡统计
            'stage1_yolo_detection': 0,
            'stage2_geometry_assessment': 0,
            'stage3_mediapipe_detection': 0,
            'stage4_pose_quality': 0,
            'stage5_dwpose_multi_person': 0,
            'stage6_scene_change_detection': 0
        }
        
        # 初始化DWpose ONNX检测器
        try:
            from methods.dwpose_onnx_detector import DWPoseONNXDetector
            self.dwpose_detector = DWPoseONNXDetector(config)
            self.logger.info("✅ DWpose ONNX检测器初始化成功")
        except Exception as e:
            self.logger.warning(f"⚠️ DWpose ONNX检测器初始化失败: {e}")
            # 尝试使用旧版本作为后备
            try:
                from methods.dwpose_detector import DWposeDetector
                self.dwpose_detector = DWposeDetector(config)
                self.logger.info("✅ DWpose检测器(后备版本)初始化成功")
            except Exception as e2:
                self.logger.warning(f"⚠️ DWpose检测器(后备版本)初始化失败: {e2}")
                self.dwpose_detector = None
        
        # 初始化场景切换检测器
        try:
            from methods.scene_change_detector import SceneChangeDetector
            self.scene_change_detector = SceneChangeDetector(config)
            self.logger.info("✅ 场景切换检测器初始化成功")
        except Exception as e:
            self.logger.warning(f"⚠️ 场景切换检测器初始化失败: {e}")
            self.scene_change_detector = None
    
    def segment_video(self, video_path: str) -> Tuple[List[Dict], Dict]:
        """主要的视频分割方法 - 四关卡筛选，返回片段和统计信息"""
        self.logger.info(f"🎬 开始处理视频: {video_path}")
        self.logger.info("📋 六关卡筛选: YOLO检测 → 几何质量评估 → MediaPipe姿态估计 → 姿态质量评估 → DWpose多人检测 → 场景切换检测")
        
        # 重置统计
        self.stats = {
            'total_frames': 0,
            'person_detected_frames': 0,
            'pose_detected_frames': 0,
            'valid_pose_frames': 0,
            # 六关卡统计
            'stage1_yolo_detection': 0,
            'stage2_geometry_assessment': 0,
            'stage3_mediapipe_detection': 0,
            'stage4_pose_quality': 0,
            'stage5_dwpose_multi_person': 0,
            'stage6_scene_change_detection': 0
        }
        
        # 重置详细统计
        self._stage2_stats = {
            'total_detections': 0,
            'area_check_passed': 0,
            'area_check_failed': 0,
            'height_check_passed': 0,
            'height_check_failed': 0,
            'face_size_check_passed': 0,
            'face_size_check_failed': 0,
            'face_width_check_passed': 0,
            'face_width_check_failed': 0,
            'face_height_check_passed': 0,
            'face_height_check_failed': 0,
            'face_aspect_check_passed': 0,
            'face_aspect_check_failed': 0,
            'face_center_check_passed': 0,
            'face_center_check_failed': 0,
            'center_position_check_passed': 0,
            'center_position_check_failed': 0,
            'final_passed': 0
        }
        
        self._stage3_stats = {
            'total_checks': 0,
            'upper_body_check_passed': 0,
            'upper_body_check_failed': 0,

            'lower_body_check_passed': 0,
            'lower_body_check_failed': 0,
            'final_passed': 0
        }
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)  # 保留浮点精度
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.logger.info(f"📊 视频信息: {total_frames}帧, {fps:.2f}fps")
        
        valid_segments = []
        current_segment = []
        frame_count = 0
        
        # 跳帧处理（每5帧检测一次）
        skip_frames = self.config['skip_frames']
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            self.stats['total_frames'] += 1
            
            # 跳帧处理：每5帧检测1帧（检测第5,10,15,20...帧）
            if frame_count % (skip_frames + 1) != 0:
                continue
            
            # 显示进度
            if frame_count % 300 == 0:
                progress = (frame_count / total_frames) * 100
                self.logger.info(f"⏳ 处理进度: {progress:.1f}% ({frame_count}/{total_frames})")
            
            # 六关卡筛选流程
            frame_result = self._process_frame_six_stages(frame, frame_count, fps)
            
            if frame_result and frame_result['is_valid']:
                current_segment.append(frame_result)
            else:
                # 当前帧无效，检查是否有有效片段需要保存
                min_segment_length = self.config['min_segment_length']
                if len(current_segment) >= min_segment_length:
                    segment = self._finalize_segment(current_segment, fps, video_path)
                    if segment:
                        valid_segments.append(segment)
                
                # 重置当前片段
                current_segment = []
        
        # 处理最后一个片段
        min_segment_length = self.config['min_segment_length']
        if len(current_segment) >= min_segment_length:
            segment = self._finalize_segment(current_segment, fps, video_path)
            if segment:
                valid_segments.append(segment)
        
        cap.release()
        
        # 打印统计信息
        self._print_statistics_six_stages()
        
        # 打印DWpose检测器统计
        if self.dwpose_detector:
            self.dwpose_detector.print_statistics()
        
        # 打印场景切换检测器统计
        if self.scene_change_detector:
            self.scene_change_detector.print_statistics()
        
        # 计算通过率
        stage_pass_rates = self._calculate_stage_pass_rates()
        
        # 收集关卡5和关卡6的详细统计
        stage5_detailed = {}
        stage6_detailed = {}
        
        if self.dwpose_detector:
            stage5_detailed = self.dwpose_detector.get_statistics()
        
        if self.scene_change_detector:
            stage6_detailed = self.scene_change_detector.get_statistics()
        
        # 返回片段和统计信息
        statistics_info = {
            'total_frames': self.stats['total_frames'],
            'stage1_yolo_detection': self.stats['stage1_yolo_detection'],
            'stage2_geometry_assessment': self.stats['stage2_geometry_assessment'],
            'stage3_mediapipe_detection': self.stats['stage3_mediapipe_detection'],
            'stage4_pose_quality': self.stats['stage4_pose_quality'],
            'stage5_dwpose_multi_person': self.stats['stage5_dwpose_multi_person'],
            'stage6_scene_change_detection': self.stats['stage6_scene_change_detection'],
            'stage_pass_rates': stage_pass_rates,
            'stage2_detailed': dict(self._stage2_stats),
            'stage3_detailed': dict(self._stage3_stats),
            'stage5_detailed': stage5_detailed,
            'stage6_detailed': stage6_detailed
        }
        
        return valid_segments, statistics_info
    
    def _calculate_stage_pass_rates(self) -> Dict:
        """计算各关卡通过率"""
        total = self.stats['total_frames']
        if total == 0:
            return {'stage1': 0, 'stage2': 0, 'stage3': 0, 'stage4': 0, 'stage5': 0, 'stage6': 0, 'final': 0}
        
        stage1_rate = (self.stats['stage1_yolo_detection'] / total * 100) if total > 0 else 0
        stage2_rate = (self.stats['stage2_geometry_assessment'] / self.stats['stage1_yolo_detection'] * 100) if self.stats['stage1_yolo_detection'] > 0 else 0
        stage3_rate = (self.stats['stage3_mediapipe_detection'] / self.stats['stage2_geometry_assessment'] * 100) if self.stats['stage2_geometry_assessment'] > 0 else 0
        stage4_rate = (self.stats['stage4_pose_quality'] / self.stats['stage3_mediapipe_detection'] * 100) if self.stats['stage3_mediapipe_detection'] > 0 else 0
        stage5_rate = (self.stats['stage5_dwpose_multi_person'] / self.stats['stage4_pose_quality'] * 100) if self.stats['stage4_pose_quality'] > 0 else 0
        stage6_rate = (self.stats['stage6_scene_change_detection'] / self.stats['stage5_dwpose_multi_person'] * 100) if self.stats['stage5_dwpose_multi_person'] > 0 else 0
        final_rate = (self.stats['stage6_scene_change_detection'] / total * 100) if total > 0 else 0
        
        return {
            'stage1': stage1_rate,
            'stage2': stage2_rate,
            'stage3': stage3_rate,
            'stage4': stage4_rate,
            'stage5': stage5_rate,
            'stage6': stage6_rate,
            'final': final_rate
        }
    
    def _process_frame_six_stages(self, frame: np.ndarray, frame_idx: int, fps: float) -> Optional[Dict]:
        """六关卡处理单帧"""
        # 关卡1: YOLO人体检测
        person_detections = self._stage1_yolo_detection(frame)
        if not person_detections:
            return None
        
        self.stats['stage1_yolo_detection'] += 1
        
        # 关卡2: 几何质量评估（上半身筛选）
        best_detection = self._stage2_geometry_assessment(person_detections, frame)
        if not best_detection:
            return None
        
        self.stats['stage2_geometry_assessment'] += 1
        
        # 关卡3: MediaPipe姿态估计
        pose_landmarks = self._stage3_mediapipe_detection(frame, best_detection, frame_idx)
        if not pose_landmarks:
            return None
        
        self.stats['stage3_mediapipe_detection'] += 1
        
        # 关卡4: 姿态质量评估
        pose_quality_score = self._stage4_pose_quality_assessment(pose_landmarks, frame.shape, best_detection)
        if pose_quality_score < self.config['quality_threshold']:
            return None
        
        self.stats['stage4_pose_quality'] += 1
        
        # 关卡5: DWpose多人检测
        if self.dwpose_detector:
            # 获取第一步的所有YOLO人体检测结果
            all_yolo_detections = person_detections  # 来自关卡1的所有人体检测
            yolo_detections = []
            for det in all_yolo_detections:
                # 正确处理YOLO检测的bbox格式
                if 'bbox' in det:
                    # 原始YOLO格式: {'bbox': [x1, y1, x2, y2], 'confidence': conf}
                    bbox = det['bbox']
                    yolo_detections.append({
                        'bbox': [bbox[0], bbox[1], bbox[2], bbox[3]], 
                        'confidence': det['confidence'], 
                        'class_id': 0
                    })
                elif 'x1' in det:
                    # 几何评估后格式: {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'confidence': conf}
                    yolo_detections.append({
                        'bbox': [det['x1'], det['y1'], det['x2'], det['y2']], 
                        'confidence': det['confidence'], 
                        'class_id': 0
                    })
            
            dwpose_result = self.dwpose_detector.detect_upper_body_keypoints(frame, yolo_detections)
            if not dwpose_result['is_single_person']:
                return None
            
            self.stats['stage5_dwpose_multi_person'] += 1
        else:
            dwpose_result = {'is_single_person': True, 'message': 'DWpose未启用'}
        
        # 关卡6: 场景切换检测
        if self.scene_change_detector:
            scene_result = self.scene_change_detector.detect_scene_change(frame, frame_idx, fps)
            if scene_result['is_scene_change']:
                return None
            
            self.stats['stage6_scene_change_detection'] += 1
        else:
            scene_result = {'is_scene_change': False, 'message': '场景切换检测未启用'}
        
        return {
            'frame_idx': frame_idx,
            'timestamp': frame_idx / fps,
            'is_valid': True,
            'yolo_detections': person_detections,
            'best_detection': best_detection,
            'pose_landmarks': pose_landmarks,
            'pose_quality_score': pose_quality_score,
            'dwpose_result': dwpose_result,
            'scene_result': scene_result
        }
    
    def _stage1_yolo_detection(self, frame: np.ndarray) -> List[Dict]:
        """关卡1: YOLO人体检测"""
        try:
            # 确保帧格式正确
            if frame is None:
                self.logger.warning("YOLO检测失败: frame为None")
                return []
                
            if frame.size == 0:
                self.logger.warning("YOLO检测失败: frame大小为0")
                return []
            
            # 修复帧格式以兼容YOLO (基于测试脚本的成功方案)
            try:
                # 强制重新创建numpy数组以解决OpenCV兼容性问题
                fixed_frame = np.array(frame, dtype=np.uint8, copy=True)
                
                # 确保是3维数组
                if len(fixed_frame.shape) != 3:
                    self.logger.warning(f"YOLO检测失败: 帧不是3维数组, shape={fixed_frame.shape}")
                    return []
                    
                # 确保是BGR格式 (高度, 宽度, 3)
                if fixed_frame.shape[2] != 3:
                    self.logger.warning(f"YOLO检测失败: 帧不是3通道, shape={fixed_frame.shape}")
                    return []
                
                # 确保值在正确范围内
                try:
                    min_val = fixed_frame.min()
                    max_val = fixed_frame.max()
                    if min_val < 0 or max_val > 255:
                        fixed_frame = np.clip(fixed_frame, 0, 255).astype(np.uint8)
                except:
                    pass  # 忽略值范围检查异常
                
                # 确保连续内存布局
                try:
                    if not fixed_frame.flags['C_CONTIGUOUS']:
                        fixed_frame = np.ascontiguousarray(fixed_frame, dtype=np.uint8)
                except:
                    fixed_frame = np.ascontiguousarray(fixed_frame, dtype=np.uint8)
                
                # 添加调试信息（第一次检测时）
                if not hasattr(self, '_debug_logged'):
                    self.logger.info(f"🔍 YOLO帧格式修复: shape={fixed_frame.shape}, dtype={fixed_frame.dtype}")
                    self._debug_logged = True
                
            except Exception as format_error:
                self.logger.warning(f"帧格式修复失败: {format_error}")
                return []
            
            # 检查修复后的帧是否有效
            if fixed_frame.shape[0] == 0 or fixed_frame.shape[1] == 0:
                self.logger.warning(f"YOLO检测失败: 帧尺寸无效, shape={fixed_frame.shape}")
                return []
            
            # 使用修复后的帧进行YOLO检测
            try:
                yolo_results = self.yolo_model(fixed_frame, verbose=False)
            except Exception as yolo_error:
                self.logger.warning(f"YOLO模型调用失败: {yolo_error}")
                return []
            
            # 🔍 关卡1：基础人体检测 - 只检查是否有人体
            person_boxes = []
            
            for r in yolo_results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        # 检查是否为人体 (class_id=0) 且置信度足够
                        if class_id == 0 and confidence > self.config['yolo_confidence_threshold']:
                            bbox = box.xyxy[0].cpu().numpy()
                            person_boxes.append({
                                'bbox': bbox,
                                'confidence': confidence
                            })
            
            # 添加检测结果调试
            if not hasattr(self, '_detection_logged') and len(person_boxes) > 0:
                self.logger.info(f"🎯 YOLO首次检测成功: 发现{len(person_boxes)}个人体")
                self._detection_logged = True
            
            return person_boxes
            
        except Exception as e:
            self.logger.warning(f"YOLO检测异常: {type(e).__name__}: {e}")
            import traceback
            self.logger.warning(f"详细错误: {traceback.format_exc()}")
            return []
    
    def _stage2_geometry_assessment(self, detections: List[Dict], frame: np.ndarray) -> Optional[Dict]:
        """关卡2: 几何质量评估 - 面部清晰度专项检查 + 中心位置检查"""
        frame_height, frame_width = frame.shape[:2]
        frame_area = frame_height * frame_width
        
        # 面部清晰度专项参数 - 关卡2只关注面部清晰度，不限制上半身
        min_person_area_ratio = self.config['min_person_area_ratio']
        min_person_height_ratio = self.config['min_person_height_ratio']
        # 移除上半身限制：max_person_height_ratio 和 max_head_position_ratio
        min_face_size_ratio = self.config['min_face_size_ratio']
        
        # 新增面部清晰度参数
        min_face_width_ratio = self.config['min_face_width_ratio']
        min_face_height_ratio = self.config['min_face_height_ratio']
        max_face_aspect_ratio = self.config['max_face_aspect_ratio']
        min_face_center_ratio = self.config['min_face_center_ratio']
        
        # 🔧 新增：中心位置检查参数 - 排除边缘人物（如观众）
        min_center_x_ratio = self.config['min_center_x_ratio']
        max_center_x_ratio = self.config['max_center_x_ratio']
        min_center_y_ratio = self.config['min_center_y_ratio']
        max_center_y_ratio = self.config['max_center_y_ratio']
        
        # 🔍 添加调试输出 - 显示实际使用的参数
        if not hasattr(self, '_geometry_debug_logged'):
            self.logger.info("🔍 面部清晰度专项筛选参数:")
            self.logger.info(f"   基础参数:")
            self.logger.info(f"     min_person_area_ratio: {min_person_area_ratio}")
            self.logger.info(f"     min_person_height_ratio: {min_person_height_ratio}")
            self.logger.info(f"     min_face_size_ratio: {min_face_size_ratio}")
            self.logger.info(f"   面部清晰度参数:")
            self.logger.info(f"     min_face_width_ratio: {min_face_width_ratio}")
            self.logger.info(f"     min_face_height_ratio: {min_face_height_ratio}")
            self.logger.info(f"     max_face_aspect_ratio: {max_face_aspect_ratio}")
            self.logger.info(f"     min_face_center_ratio: {min_face_center_ratio}")
            self.logger.info(f"   🔧 中心位置检查参数（排除观众等边缘人物）:")
            self.logger.info(f"     中心X轴范围: {min_center_x_ratio:.1f} - {max_center_x_ratio:.1f}")
            self.logger.info(f"     中心Y轴范围: {min_center_y_ratio:.1f} - {max_center_y_ratio:.1f}")
            self.logger.info(f"   帧尺寸: {frame_height}x{frame_width}")
            self._geometry_debug_logged = True
        
        # 统计每个检查项的通过/失败情况
        if not hasattr(self, '_stage2_stats'):
            self._stage2_stats = {
                'total_detections': 0,
                'area_check_passed': 0,
                'area_check_failed': 0,
                'height_check_passed': 0,
                'height_check_failed': 0,
                'face_size_check_passed': 0,
                'face_size_check_failed': 0,
                'face_width_check_passed': 0,
                'face_width_check_failed': 0,
                'face_height_check_passed': 0,
                'face_height_check_failed': 0,
                'face_aspect_check_passed': 0,
                'face_aspect_check_failed': 0,
                'face_center_check_passed': 0,
                'face_center_check_failed': 0,
                'center_position_check_passed': 0,  # 新增
                'center_position_check_failed': 0,  # 新增
                'final_passed': 0
            }
        
        passed_count = 0
        failed_count = 0
        
        for detection in detections:
            self._stage2_stats['total_detections'] += 1
            
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            person_width = x2 - x1
            person_height = y2 - y1
            person_area = person_width * person_height
            
            # 计算人体中心位置
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            center_x_ratio = center_x / frame_width
            center_y_ratio = center_y / frame_height
            
            # 🔍 显示前几个检测框的详细信息
            if self._stage2_stats['total_detections'] <= 5:
                self.logger.info(f"🔍 检测框 #{self._stage2_stats['total_detections']}:")
                self.logger.info(f"   坐标: ({x1:.1f}, {y1:.1f}) -> ({x2:.1f}, {y2:.1f})")
                self.logger.info(f"   尺寸: {person_width:.1f} x {person_height:.1f}")
                self.logger.info(f"   面积: {person_area:.1f}")
                self.logger.info(f"   高度比例: {person_height/frame_height:.3f}")
                self.logger.info(f"   底部位置: {(y1 + person_height)/frame_height:.3f}")
                self.logger.info(f"   🔧 中心位置: ({center_x_ratio:.3f}, {center_y_ratio:.3f})")
            
            # 🔧 新增：中心位置检查 - 必须在画面中央区域
            center_position_passed = (min_center_x_ratio <= center_x_ratio <= max_center_x_ratio and 
                                    min_center_y_ratio <= center_y_ratio <= max_center_y_ratio)
            
            if not center_position_passed:
                self._stage2_stats['center_position_check_failed'] += 1
                failed_count += 1
                self.logger.debug(f"❌ 人物位置过于边缘: 中心位置({center_x_ratio:.3f}, {center_y_ratio:.3f}) 不在中央区域内")
                continue
            else:
                self._stage2_stats['center_position_check_passed'] += 1
            
            # 计算基础几何属性
            area_ratio = person_area / frame_area
            height_ratio = person_height / frame_height
            head_y = y1 + person_height * 0.15  # 假设头部在人体上15%的位置
            head_position_ratio = head_y / frame_height
            
            # 计算面部区域（更精确的面部定位）
            face_region_height = person_height * 0.20  # 面部占人体高度的20%
            face_region_width = person_width * 0.60    # 面部占人体宽度的60%
            face_region_y = y1 + person_height * 0.10  # 面部从人体上10%开始
            face_region_x = x1 + person_width * 0.20   # 面部从人体左20%开始
            
            face_area_ratio = (face_region_width * face_region_height) / frame_area
            face_width_ratio = face_region_width / frame_width
            face_height_ratio = face_region_height / frame_height
            face_aspect_ratio = face_region_width / face_region_height
            face_center_ratio = (face_region_y + face_region_height/2) / frame_height
            
            # 上半身特写检查：确保是上半身而不是全身
            person_bottom_y = y1 + person_height
            frame_bottom_ratio = person_bottom_y / frame_height
            
            # 🔍 详细的检查过程
            check_results = {
                'center_position_check': center_position_passed,  # 新增
                'area_check': area_ratio >= min_person_area_ratio,
                'height_check': height_ratio >= min_person_height_ratio,
                'face_size_check': face_area_ratio >= min_face_size_ratio,
                'face_width_check': face_width_ratio >= min_face_width_ratio,
                'face_height_check': face_height_ratio >= min_face_height_ratio,
                'face_aspect_check': face_aspect_ratio <= max_face_aspect_ratio,
                'face_center_check': face_center_ratio >= min_face_center_ratio
            }
            
            # 1. 基础显著性检查：人体占画面比例
            if area_ratio < min_person_area_ratio:
                self._stage2_stats['area_check_failed'] += 1
                failed_count += 1
                self.logger.debug(f"❌ 面积比例不符合: {area_ratio:.3f} < {min_person_area_ratio}")
                continue
            else:
                self._stage2_stats['area_check_passed'] += 1
            
            # 2. 最小高度检查：确保人体足够大
            if height_ratio < min_person_height_ratio:
                self._stage2_stats['height_check_failed'] += 1
                failed_count += 1
                self.logger.debug(f"❌ 高度比例不符合: {height_ratio:.3f} < {min_person_height_ratio}")
                continue
            else:
                self._stage2_stats['height_check_passed'] += 1
            

            
            # 5. 面部大小检查：确保面部可见
            if face_area_ratio < min_face_size_ratio:
                self._stage2_stats['face_size_check_failed'] += 1
                failed_count += 1
                self.logger.debug(f"❌ 面部太小: {face_area_ratio:.4f} < {min_face_size_ratio}")
                continue
            else:
                self._stage2_stats['face_size_check_passed'] += 1
            
            # 6. 面部宽度检查：确保面部足够宽，能看到完整表情
            if face_width_ratio < min_face_width_ratio:
                self._stage2_stats['face_width_check_failed'] += 1
                failed_count += 1
                self.logger.debug(f"❌ 面部宽度不足: {face_width_ratio:.3f} < {min_face_width_ratio}")
                continue
            else:
                self._stage2_stats['face_width_check_passed'] += 1
            
            # 7. 面部高度检查：确保面部足够高，能看到完整表情
            if face_height_ratio < min_face_height_ratio:
                self._stage2_stats['face_height_check_failed'] += 1
                failed_count += 1
                self.logger.debug(f"❌ 面部高度不足: {face_height_ratio:.3f} < {min_face_height_ratio}")
                continue
            else:
                self._stage2_stats['face_height_check_passed'] += 1
            
            # 8. 面部宽高比检查：确保面部比例正常，避免过度拉伸
            if face_aspect_ratio > max_face_aspect_ratio:
                self._stage2_stats['face_aspect_check_failed'] += 1
                failed_count += 1
                self.logger.debug(f"❌ 面部比例异常: {face_aspect_ratio:.2f} > {max_face_aspect_ratio}")
                continue
            else:
                self._stage2_stats['face_aspect_check_passed'] += 1
            
            # 9. 面部中心位置检查：确保面部在画面中心区域
            if face_center_ratio < min_face_center_ratio:
                self._stage2_stats['face_center_check_failed'] += 1
                failed_count += 1
                self.logger.debug(f"❌ 面部位置过低: {face_center_ratio:.3f} < {min_face_center_ratio}")
                continue
            else:
                self._stage2_stats['face_center_check_passed'] += 1
            
            # 🔍 如果通过所有检查，记录详细信息
            passed_count += 1
            self._stage2_stats['final_passed'] += 1
            self.logger.debug(f"✅ 面部清晰度检查通过:")
            self.logger.debug(f"   🔧 中心位置检查: ({center_x_ratio:.3f}, {center_y_ratio:.3f}) 在有效范围内")
            self.logger.debug(f"   基础检查:")
            self.logger.debug(f"     area_ratio: {area_ratio:.3f} >= {min_person_area_ratio}")
            self.logger.debug(f"     height_ratio: {height_ratio:.3f} >= {min_person_height_ratio}")
            self.logger.debug(f"   面部清晰度检查:")
            self.logger.debug(f"     face_area: {face_area_ratio:.4f} >= {min_face_size_ratio}")
            self.logger.debug(f"     face_width: {face_width_ratio:.3f} >= {min_face_width_ratio}")
            self.logger.debug(f"     face_height: {face_height_ratio:.3f} >= {min_face_height_ratio}")
            self.logger.debug(f"     face_aspect: {face_aspect_ratio:.2f} <= {max_face_aspect_ratio}")
            self.logger.debug(f"     face_center: {face_center_ratio:.3f} >= {min_face_center_ratio}")
            
            # 通过所有几何检查
            detection['geometry_scores'] = {
                'area_ratio': area_ratio,
                'height_ratio': height_ratio,
                'face_area_ratio': face_area_ratio,
                'face_width_ratio': face_width_ratio,
                'face_height_ratio': face_height_ratio,
                'face_aspect_ratio': face_aspect_ratio,
                'face_center_ratio': face_center_ratio,
                'frame_bottom_ratio': frame_bottom_ratio,
                'center_x_ratio': center_x_ratio,  # 新增
                'center_y_ratio': center_y_ratio,  # 新增
                'check_results': check_results
            }
            return detection
        
        # 🔍 记录统计信息
        total_detections = passed_count + failed_count
        if total_detections > 0:
            pass_rate = passed_count / total_detections * 100
            self.logger.debug(f"📊 面部清晰度筛选: 通过{passed_count}个, 失败{failed_count}个, 通过率{pass_rate:.1f}%")
        
        return None
    
    def _analyze_geometry_distribution(self, all_detections: List[Dict], frame_shape: Tuple[int, int]):
        """分析所有检测框的几何属性分布"""
        if not all_detections:
            return
        
        frame_height, frame_width = frame_shape[:2]
        frame_area = frame_height * frame_width
        
        # 收集所有几何属性
        area_ratios = []
        height_ratios = []
        head_position_ratios = []
        face_area_ratios = []
        
        for detection in all_detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            person_width = x2 - x1
            person_height = y2 - y1
            person_area = person_width * person_height
            
            area_ratio = person_area / frame_area
            height_ratio = person_height / frame_height
            head_y = y1 + person_height * 0.15
            head_position_ratio = head_y / frame_height
            face_region_height = person_height * 0.15
            face_area_ratio = (person_width * face_region_height) / frame_area
            
            area_ratios.append(area_ratio)
            height_ratios.append(height_ratio)
            head_position_ratios.append(head_position_ratio)
            face_area_ratios.append(face_area_ratio)
        
        # 统计分布
        import numpy as np
        self.logger.info("📊 几何属性分布统计:")
        self.logger.info(f"   面积比例: min={min(area_ratios):.3f}, max={max(area_ratios):.3f}, mean={np.mean(area_ratios):.3f}")
        self.logger.info(f"   高度比例: min={min(height_ratios):.3f}, max={max(height_ratios):.3f}, mean={np.mean(height_ratios):.3f}")
        self.logger.info(f"   头部位置: min={min(head_position_ratios):.3f}, max={max(head_position_ratios):.3f}, mean={np.mean(head_position_ratios):.3f}")
        self.logger.info(f"   面部面积: min={min(face_area_ratios):.4f}, max={max(face_area_ratios):.4f}, mean={np.mean(face_area_ratios):.4f}")
    
    def _stage3_mediapipe_detection(self, frame: np.ndarray, detection: Dict, frame_idx: int) -> Optional[object]:
        """关卡3: MediaPipe姿态估计"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 可选：裁剪到人体区域以提高精度
            if self.config['crop_to_person']:
                bbox = detection['bbox']
                x1, y1, x2, y2 = bbox.astype(int)
                h, w = frame.shape[:2]
                margin = 0.1
                x1 = max(0, int(x1 - (x2-x1) * margin))
                y1 = max(0, int(y1 - (y2-y1) * margin))
                x2 = min(w, int(x2 + (x2-x1) * margin))
                y2 = min(h, int(y2 + (y2-y1) * margin))
                
                cropped_frame = rgb_frame[y1:y2, x1:x2]
                if cropped_frame.size == 0:
                    return None
                
                results = self.pose.process(cropped_frame)
                
                # 调整关键点坐标到原图
                if results.pose_landmarks:
                    for landmark in results.pose_landmarks.landmark:
                        landmark.x = landmark.x * (x2 - x1) / w + x1 / w
                        landmark.y = landmark.y * (y2 - y1) / h + y1 / h
            else:
                results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                # 关卡3质量检查：验证关键点可见性和姿态完整性
                # 注意：多人检测已移至关卡5的DWpose检测器中
                if self._validate_stage3_quality(results.pose_landmarks, frame_idx):
                    return results.pose_landmarks
            
        except Exception as e:
            self.logger.warning(f"MediaPipe姿态检测失败: {e}")
        
        return None
    
    # 已移除 _validate_single_person 方法
    # 多人检测功能已转移到关卡5的DWpose检测器中
    # 这样可以避免重复检测，并使用更准确的DWpose人数检测
    
    def _validate_stage3_quality(self, landmarks, frame_idx) -> bool:
        """关卡3质量验证：要求上半身关键点7个都可见，没有下半身关键点"""
        try:
            # 配置参数
            visibility_threshold = self.config['stage3_visibility_threshold']
            
            # 初始化关卡3统计信息
            if not hasattr(self, '_stage3_stats'):
                self._stage3_stats = {
                    'total_checks': 0,
                    'upper_body_check_passed': 0,
                    'upper_body_check_failed': 0,

                    'lower_body_check_passed': 0,
                    'lower_body_check_failed': 0,
                    'final_passed': 0
                }
            
            self._stage3_stats['total_checks'] += 1
            
            # 上半身关键点（必须全部7个可见）
            upper_body_landmarks = [
                self.mp_pose.PoseLandmark.NOSE,
                self.mp_pose.PoseLandmark.LEFT_SHOULDER,
                self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
                self.mp_pose.PoseLandmark.LEFT_ELBOW,
                self.mp_pose.PoseLandmark.RIGHT_ELBOW,
                self.mp_pose.PoseLandmark.LEFT_WRIST,
                self.mp_pose.PoseLandmark.RIGHT_WRIST
            ]
            
            # 1. 检查上半身关键点：必须全部7个可见
            upper_body_visible = 0
            upper_body_details = []
            for landmark_id in upper_body_landmarks:
                landmark = landmarks.landmark[landmark_id]
                is_visible = landmark.visibility > visibility_threshold
                if is_visible:
                    upper_body_visible += 1
                upper_body_details.append(f"{landmark_id.name}: {landmark.visibility:.2f} ({'可见' if is_visible else '不可见'})")
            
            # 🔍 添加详细调试：当帧在问题时间段时输出详细信息
            if hasattr(self, '_debug_frame_range'):
                # 检查是否在问题时间段（101-104秒，对应帧2443-2496）
                current_frame_idx = getattr(self, '_current_frame_idx', 0)
                if 2440 <= current_frame_idx <= 2500:  # 扩大范围确保捕获
                    self.logger.info(f"🔍 问题片段调试 - 帧{current_frame_idx}:")
                    self.logger.info(f"   上半身关键点详细信息:")
                    for detail in upper_body_details:
                        self.logger.info(f"     {detail}")
                    self.logger.info(f"   可见关键点总数: {upper_body_visible}/7")
            
            # 🔍 添加问题片段调试：检查是否在手机屏幕问题时间段
            if 2440 <= frame_idx <= 2500:  # 检查问题帧范围（101-104秒）
                self.logger.info(f"🔍 手机屏幕问题片段调试 - 帧{frame_idx}:")
                self.logger.info(f"   上半身关键点详细信息:")
                for detail in upper_body_details:
                    self.logger.info(f"     {detail}")
                self.logger.info(f"   可见关键点总数: {upper_body_visible}/7，要求: 7")
                self.logger.info(f"   可见度阈值: {visibility_threshold}")
            
            # 上半身关键点必须全部7个可见
            upper_body_passed = upper_body_visible >= 7
            if upper_body_passed:
                self._stage3_stats['upper_body_check_passed'] += 1
            else:
                self._stage3_stats['upper_body_check_failed'] += 1
                self.logger.debug(f"❌ 上半身关键点不足: {upper_body_visible}/7")
                self.logger.debug(f"   详细: {', '.join(upper_body_details)}")
                return False
            
            # 移除重复的肩膀检测（已在上半身关键点检查中包含）
            
            # 2. 下半身零容忍检查：允许腰部(髋部)，但膝盖、脚踝、脚趾零容忍
            # 腰部关键点 - 完全允许（不检查）
            # 膝盖、脚踝、脚趾 - 零容忍检查
            
            # 零容忍的下半身关键点：膝盖、脚踝、脚趾
            zero_tolerance_landmarks = [
                # 膝盖
                self.mp_pose.PoseLandmark.LEFT_KNEE,
                self.mp_pose.PoseLandmark.RIGHT_KNEE,
                # 脚踝
                self.mp_pose.PoseLandmark.LEFT_ANKLE,
                self.mp_pose.PoseLandmark.RIGHT_ANKLE,
                # 脚趾
                self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
                self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX
            ]
            
            # 零容忍检查：只要检测到任何一个膝盖、脚踝、脚趾就排除
            min_detection_threshold = self.config['leg_foot_visibility_threshold']
            
            lower_body_passed = True
            detected_lower_body = []
            for landmark_id in zero_tolerance_landmarks:
                landmark = landmarks.landmark[landmark_id]
                if landmark.visibility > min_detection_threshold:
                    lower_body_passed = False
                    detected_lower_body.append(f"{landmark_id.name}({landmark.visibility:.2f})")
            
            if lower_body_passed:
                self._stage3_stats['lower_body_check_passed'] += 1
            else:
                self._stage3_stats['lower_body_check_failed'] += 1
                self.logger.debug(f"❌ 检测到膝盖/脚踝/脚趾: {', '.join(detected_lower_body)}，立即排除")
                return False
            
            # 所有检查都通过
            self._stage3_stats['final_passed'] += 1
            self.logger.debug(f"✅ 关卡三通过: 上半身{upper_body_visible}/7可见，下半身0个")
            
            return True
            
        except Exception as e:
            self.logger.warning(f"关卡三验证异常: {e}")
            return False
    
    def _stage4_pose_quality_assessment(self, landmarks: object, frame_shape: Tuple, detection: Dict) -> float:
        """关卡4: 姿态质量评估"""
        try:
            scores = {
                'upper_body_completeness': self._assess_upper_body_completeness(landmarks),
                'face_orientation': self._assess_face_orientation(landmarks),
                'speaking_gesture': self._assess_speaking_gesture(landmarks),
                'body_stability': self._assess_body_stability(landmarks, frame_shape)
            }
            
            # 加权计算总分
            weights = self.config.get('quality_weights', {
                'upper_body': 0.4,
                'face_orientation': 0.3,
                'hand_gesture': 0.2,
                'body_ratio': 0.1
            })
            
            final_score = (
                scores['upper_body_completeness'] * weights.get('upper_body', 0.4) +
                scores['face_orientation'] * weights.get('face_orientation', 0.3) +
                scores['speaking_gesture'] * weights.get('hand_gesture', 0.2) +
                scores['body_stability'] * weights.get('body_ratio', 0.1)
            )
            
            return final_score
        
        except Exception as e:
            self.logger.warning(f"姿态质量评估失败: {e}")
            return 0.0
    
    def _assess_upper_body_completeness(self, landmarks) -> float:
        """评估上半身完整性 - 要求全部7个关键点可见"""
        try:
            upper_body_landmarks = [
                self.mp_pose.PoseLandmark.NOSE,
                self.mp_pose.PoseLandmark.LEFT_SHOULDER,
                self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
                self.mp_pose.PoseLandmark.LEFT_ELBOW,
                self.mp_pose.PoseLandmark.RIGHT_ELBOW,
                self.mp_pose.PoseLandmark.LEFT_WRIST,
                self.mp_pose.PoseLandmark.RIGHT_WRIST,
            ]
            
            visibility_threshold = self.config.get('visibility_threshold', 0.7)
            visible_count = 0
            total_visibility = 0
            
            for landmark_id in upper_body_landmarks:
                landmark = landmarks.landmark[landmark_id]
                if landmark.visibility > visibility_threshold:
                    visible_count += 1
                    total_visibility += landmark.visibility
            
            # 要求全部7个关键点可见（与关卡三保持一致）
            if visible_count == 7:  # 必须全部7个关键点可见
                return (visible_count / len(upper_body_landmarks)) * (total_visibility / visible_count)
            else:
                return 0.0
        except:
            return 0.0
    
    def _assess_face_orientation(self, landmarks) -> float:
        """评估面部朝向"""
        try:
            nose = landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
            left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            
            visibility_threshold = self.config.get('visibility_threshold', 0.7)
            
            if (nose.visibility > visibility_threshold and 
                left_shoulder.visibility > visibility_threshold and 
                right_shoulder.visibility > visibility_threshold):
                
                # 计算面部是否朝向前方
                shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
                nose_deviation = abs(nose.x - shoulder_center_x)
                return max(0, 1.0 - nose_deviation * 5)
            
            return 0.0
        except:
            return 0.0
    
    def _assess_speaking_gesture(self, landmarks) -> float:
        """评估演讲手势"""
        try:
            left_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            
            visibility_threshold = self.config.get('visibility_threshold', 0.7)
            
            if (left_wrist.visibility > visibility_threshold and 
                right_wrist.visibility > visibility_threshold):
                
                # 检查手部位置
                wrist_y_avg = (left_wrist.y + right_wrist.y) / 2
                shoulder_y_avg = (left_shoulder.y + right_shoulder.y) / 2
                
                # 手部在合理的演讲区域
                if wrist_y_avg < shoulder_y_avg + 0.3:
                    return min(1.0, (left_wrist.visibility + right_wrist.visibility) / 2)
            
            return 0.0
        except:
            return 0.0
    
    def _assess_body_stability(self, landmarks, frame_shape) -> float:
        """评估身体稳定性"""
        try:
            h, w = frame_shape[:2]
            body_landmarks = [lm for lm in landmarks.landmark if lm.visibility > 0.5]
            
            if len(body_landmarks) >= 5:
                # 计算身体占画面的比例
                y_coords = [lm.y for lm in body_landmarks]
                body_height = max(y_coords) - min(y_coords)
                return min(1.0, body_height * 2)
            
            return 0.0
        except:
            return 0.0
    
    def _print_statistics_six_stages(self):
        """打印六关卡统计信息"""
        total = self.stats['total_frames']
        
        self.logger.info("📊 YOLO+MediaPipe 六关卡处理统计:")
        self.logger.info(f"  总帧数: {total}")
        
        if total > 0:
            stage1_rate = (self.stats['stage1_yolo_detection'] / total * 100)
            self.logger.info(f"  🚪 关卡1 YOLO检测: {self.stats['stage1_yolo_detection']} ({stage1_rate:.1f}%)")
            
            if self.stats['stage1_yolo_detection'] > 0:
                stage2_rate = (self.stats['stage2_geometry_assessment'] / self.stats['stage1_yolo_detection'] * 100)
                self.logger.info(f"  🚪 关卡2 几何评估: {self.stats['stage2_geometry_assessment']} ({stage2_rate:.1f}%)")
                
                # 🔍 添加关卡2详细统计
                if hasattr(self, '_stage2_stats') and self._stage2_stats['total_detections'] > 0:
                    self.logger.info("  📊 关卡2详细统计:")
                    total_detections = self._stage2_stats['total_detections']
                    
                    # 面积检查统计
                    area_passed = self._stage2_stats['area_check_passed']
                    area_failed = self._stage2_stats['area_check_failed']
                    area_rate = (area_passed / total_detections * 100) if total_detections > 0 else 0
                    self.logger.info(f"    面积比例检查: 通过{area_passed}, 失败{area_failed} ({area_rate:.1f}%)")
                    
                    # 高度检查统计
                    height_passed = self._stage2_stats['height_check_passed']
                    height_failed = self._stage2_stats['height_check_failed']
                    height_rate = (height_passed / total_detections * 100) if total_detections > 0 else 0
                    self.logger.info(f"    高度比例检查: 通过{height_passed}, 失败{height_failed} ({height_rate:.1f}%)")
                    
                    # 面部大小检查统计
                    face_passed = self._stage2_stats['face_size_check_passed']
                    face_failed = self._stage2_stats['face_size_check_failed']
                    face_rate = (face_passed / total_detections * 100) if total_detections > 0 else 0
                    self.logger.info(f"    面部大小检查: 通过{face_passed}, 失败{face_failed} ({face_rate:.1f}%)")
                    
                    # 面部宽度检查统计
                    face_width_passed = self._stage2_stats['face_width_check_passed']
                    face_width_failed = self._stage2_stats['face_width_check_failed']
                    face_width_rate = (face_width_passed / total_detections * 100) if total_detections > 0 else 0
                    self.logger.info(f"    面部宽度检查: 通过{face_width_passed}, 失败{face_width_failed} ({face_width_rate:.1f}%)")
                    
                    # 面部高度检查统计
                    face_height_passed = self._stage2_stats['face_height_check_passed']
                    face_height_failed = self._stage2_stats['face_height_check_failed']
                    face_height_rate = (face_height_passed / total_detections * 100) if total_detections > 0 else 0
                    self.logger.info(f"    面部高度检查: 通过{face_height_passed}, 失败{face_height_failed} ({face_height_rate:.1f}%)")
                    
                    # 面部宽高比检查统计
                    face_aspect_passed = self._stage2_stats['face_aspect_check_passed']
                    face_aspect_failed = self._stage2_stats['face_aspect_check_failed']
                    face_aspect_rate = (face_aspect_passed / total_detections * 100) if total_detections > 0 else 0
                    self.logger.info(f"    面部宽高比检查: 通过{face_aspect_passed}, 失败{face_aspect_failed} ({face_aspect_rate:.1f}%)")
                    
                    # 面部中心位置检查统计
                    face_center_passed = self._stage2_stats['face_center_check_passed']
                    face_center_failed = self._stage2_stats['face_center_check_failed']
                    face_center_rate = (face_center_passed / total_detections * 100) if total_detections > 0 else 0
                    self.logger.info(f"    面部中心位置检查: 通过{face_center_passed}, 失败{face_center_failed} ({face_center_rate:.1f}%)")
                    
                    # 中心位置检查统计
                    center_position_passed = self._stage2_stats['center_position_check_passed']
                    center_position_failed = self._stage2_stats['center_position_check_failed']
                    center_position_rate = (center_position_passed / total_detections * 100) if total_detections > 0 else 0
                    self.logger.info(f"    中心位置检查: 通过{center_position_passed}, 失败{center_position_failed} ({center_position_rate:.1f}%)")
                    
                    # 最终通过统计
                    final_passed = self._stage2_stats['final_passed']
                    final_rate = (final_passed / total_detections * 100) if total_detections > 0 else 0
                    self.logger.info(f"    最终通过: {final_passed} ({final_rate:.1f}%)")
                
                if self.stats['stage2_geometry_assessment'] > 0:
                    stage3_rate = (self.stats['stage3_mediapipe_detection'] / self.stats['stage2_geometry_assessment'] * 100)
                    self.logger.info(f"  🚪 关卡3 MediaPipe检测: {self.stats['stage3_mediapipe_detection']} ({stage3_rate:.1f}%)")
                    
                    # 🔍 添加关卡3详细统计
                    if hasattr(self, '_stage3_stats') and self._stage3_stats['total_checks'] > 0:
                        self.logger.info("  📊 关卡3详细统计:")
                        total_checks = self._stage3_stats['total_checks']
                        
                        # 上半身关键点检查统计
                        upper_passed = self._stage3_stats['upper_body_check_passed']
                        upper_failed = self._stage3_stats['upper_body_check_failed']
                        upper_rate = (upper_passed / total_checks * 100) if total_checks > 0 else 0
                        self.logger.info(f"    上半身7关键点检查: 通过{upper_passed}, 失败{upper_failed} ({upper_rate:.1f}%)")
                        
                        # 移除重复的肩膀质量检查统计
                        
                        # 下半身排除检查统计
                        lower_passed = self._stage3_stats['lower_body_check_passed']
                        lower_failed = self._stage3_stats['lower_body_check_failed']
                        lower_rate = (lower_passed / total_checks * 100) if total_checks > 0 else 0
                        self.logger.info(f"    下半身零容忍检查: 通过{lower_passed}, 失败{lower_failed} ({lower_rate:.1f}%)")
                        
                        # 最终通过统计
                        final_passed = self._stage3_stats['final_passed']
                        final_rate = (final_passed / total_checks * 100) if total_checks > 0 else 0
                        self.logger.info(f"    最终通过: {final_passed} ({final_rate:.1f}%)")
                    
                    if self.stats['stage3_mediapipe_detection'] > 0:
                        stage4_rate = (self.stats['stage4_pose_quality'] / self.stats['stage3_mediapipe_detection'] * 100)
                        self.logger.info(f"  🚪 关卡4 姿态质量: {self.stats['stage4_pose_quality']} ({stage4_rate:.1f}%)")
                        
                        # 关卡5: DWpose多人检测统计
                        if self.stats['stage4_pose_quality'] > 0:
                            stage5_rate = (self.stats['stage5_dwpose_multi_person'] / self.stats['stage4_pose_quality'] * 100)
                            self.logger.info(f"  🚪 关卡5 DWpose多人检测: {self.stats['stage5_dwpose_multi_person']} ({stage5_rate:.1f}%)")
                            
                            # 🔍 添加关卡5详细统计
                            if self.dwpose_detector:
                                dwpose_stats = self.dwpose_detector.get_statistics()
                                if dwpose_stats['total_frames_processed'] > 0:
                                    self.logger.info("  📊 关卡5详细统计:")
                                    total_dwpose = dwpose_stats['total_frames_processed']
                                    
                                    # 单人检测统计
                                    single_person = dwpose_stats['single_person_frames']
                                    single_rate = (single_person / total_dwpose * 100) if total_dwpose > 0 else 0
                                    self.logger.info(f"    单人检测: {single_person} ({single_rate:.1f}%)")
                                    
                                    # 多人检测统计
                                    multi_person = dwpose_stats['multi_person_frames']
                                    multi_rate = (multi_person / total_dwpose * 100) if total_dwpose > 0 else 0
                                    self.logger.info(f"    多人检测: {multi_person} ({multi_rate:.1f}%)")
                                    
                                    # 无人检测统计
                                    no_person = dwpose_stats['no_person_frames']
                                    no_rate = (no_person / total_dwpose * 100) if total_dwpose > 0 else 0
                                    self.logger.info(f"    未检测到人: {no_person} ({no_rate:.1f}%)")
                                    
                                    # 检测失败统计
                                    failures = dwpose_stats['detection_failures']
                                    failure_rate = (failures / total_dwpose * 100) if total_dwpose > 0 else 0
                                    self.logger.info(f"    检测失败: {failures} ({failure_rate:.1f}%)")
                                    
                                    # 平均人数统计
                                    avg_persons = dwpose_stats['total_persons_detected'] / total_dwpose if total_dwpose > 0 else 0
                                    self.logger.info(f"    平均人数: {avg_persons:.2f}人/帧")
                            
                            # 关卡6: 场景切换检测统计
                            if self.stats['stage5_dwpose_multi_person'] > 0:
                                stage6_rate = (self.stats['stage6_scene_change_detection'] / self.stats['stage5_dwpose_multi_person'] * 100)
                                self.logger.info(f"  🚪 关卡6 场景切换检测: {self.stats['stage6_scene_change_detection']} ({stage6_rate:.1f}%)")
                                
                                # 🔍 添加关卡6详细统计
                                if self.scene_change_detector:
                                    scene_stats = self.scene_change_detector.get_statistics()
                                    if scene_stats['total_frames'] > 0:
                                        self.logger.info("  📊 关卡6详细统计:")
                                        total_scene = scene_stats['total_frames']
                                        
                                        # 场景切换统计
                                        scene_changes = scene_stats.get('scene_changes_detected', 0)
                                        stable_frames = total_scene - scene_changes
                                        
                                        stable_rate = (stable_frames / total_scene * 100) if total_scene > 0 else 0
                                        change_rate = (scene_changes / total_scene * 100) if total_scene > 0 else 0
                                        
                                        self.logger.info(f"    场景稳定: {stable_frames} ({stable_rate:.1f}%)")
                                        self.logger.info(f"    场景切换: {scene_changes} ({change_rate:.1f}%)")
                                        
                                        # 平均相似度统计
                                        if 'average_similarity' in scene_stats:
                                            avg_similarity = scene_stats['average_similarity']
                                            self.logger.info(f"    平均相似度: {avg_similarity:.3f}")
                                            self.logger.info(f"    场景切换次数: {scene_changes}次")
                                
                                final_rate = (self.stats['stage6_scene_change_detection'] / total * 100)
                                self.logger.info(f"  🎯 最终通过率: {final_rate:.2f}%")
                            else:
                                self.logger.info("  📊 关卡6未执行（关卡5无通过帧）")
                                final_rate = (self.stats['stage5_dwpose_multi_person'] / total * 100)
                                self.logger.info(f"  🎯 最终通过率: {final_rate:.2f}%")
                        else:
                            self.logger.info("  📊 关卡5未执行（关卡4无通过帧）")
                            final_rate = (self.stats['stage4_pose_quality'] / total * 100)
                            self.logger.info(f"  🎯 最终通过率: {final_rate:.2f}%")
    
    def _process_frame(self, frame: np.ndarray, frame_idx: int, fps: int) -> Dict:
        """处理单帧"""
        result = {
            'frame_idx': frame_idx,
            'timestamp': frame_idx / fps,
            'is_valid': False,
            'person_detected': False,
            'pose_detected': False,
            'quality_score': 0.0,
            'pose_landmarks': None,
            'person_bbox': None
        }
        
        # 第一步：YOLO人体检测
        try:
            yolo_results = self.yolo_model(frame, verbose=False)
            person_boxes = []
            
            for r in yolo_results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        # 检查是否为人体 (class_id=0) 且置信度足够
                        if class_id == 0 and confidence > 0.5:  # 提高YOLO置信度阈值
                            person_boxes.append({
                                'bbox': box.xyxy[0].cpu().numpy(),
                                'confidence': confidence
                            })
            
            if person_boxes:
                result['person_detected'] = True
                self.stats['person_detected_frames'] += 1
                
                # 选择置信度最高的人体框
                best_person = max(person_boxes, key=lambda x: x['confidence'])
                result['person_bbox'] = best_person['bbox']
                
                # 第二步：MediaPipe姿态估计
                pose_result = self._detect_pose_with_mediapipe(frame, best_person['bbox'])
                if pose_result:
                    result['pose_detected'] = True
                    result['pose_landmarks'] = pose_result['landmarks']
                    self.stats['pose_detected_frames'] += 1
                    
                    # 第三步：评估姿态质量
                    quality_score = self._evaluate_pose_quality(pose_result['landmarks'], frame.shape)
                    result['quality_score'] = quality_score
                    
                    # 判断是否为有效演讲姿态 (提高质量阈值)
                    if quality_score > self.config.get('quality_threshold', 0.6):
                        result['is_valid'] = True
                        self.stats['valid_pose_frames'] += 1
        
        except Exception as e:
            self.logger.warning(f"处理帧 {frame_idx} 时出错: {e}")
        
        return result
    
    def _detect_pose_with_mediapipe(self, frame: np.ndarray, person_bbox: np.ndarray) -> Optional[Dict]:
        """使用MediaPipe检测姿态"""
        try:
            # 转换为RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 可选：裁剪到人体区域以提高精度
            if self.config['crop_to_person']:
                x1, y1, x2, y2 = person_bbox.astype(int)
                # 扩展边界框
                h, w = frame.shape[:2]
                margin = 0.1
                x1 = max(0, int(x1 - (x2-x1) * margin))
                y1 = max(0, int(y1 - (y2-y1) * margin))
                x2 = min(w, int(x2 + (x2-x1) * margin))
                y2 = min(h, int(y2 + (y2-y1) * margin))
                
                cropped_frame = rgb_frame[y1:y2, x1:x2]
                if cropped_frame.size == 0:
                    return None
                
                results = self.pose.process(cropped_frame)
                
                # 调整关键点坐标到原图
                if results.pose_landmarks:
                    for landmark in results.pose_landmarks.landmark:
                        landmark.x = landmark.x * (x2 - x1) / w + x1 / w
                        landmark.y = landmark.y * (y2 - y1) / h + y1 / h
            else:
                results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                return {
                    'landmarks': results.pose_landmarks,
                    'visibility': [lm.visibility for lm in results.pose_landmarks.landmark]
                }
            
        except Exception as e:
            self.logger.warning(f"MediaPipe姿态检测失败: {e}")
        
        return None
    
    def _evaluate_pose_quality(self, landmarks, frame_shape) -> float:
        """评估姿态质量"""
        if not landmarks:
            return 0.0
        
        scores = {
            'upper_body': 0.0,
            'face_orientation': 0.0,
            'hand_gesture': 0.0,
            'body_ratio': 0.0,
            'stability': 0.0
        }
        
        # 上半身可见性 (40%) - 提高可见性要求
        upper_body_landmarks = [
            self.mp_pose.PoseLandmark.NOSE,
            self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            self.mp_pose.PoseLandmark.LEFT_ELBOW,
            self.mp_pose.PoseLandmark.RIGHT_ELBOW,
            self.mp_pose.PoseLandmark.LEFT_WRIST,
            self.mp_pose.PoseLandmark.RIGHT_WRIST,
        ]
        
        visible_count = 0
        total_visibility = 0
        for landmark_id in upper_body_landmarks:
            landmark = landmarks.landmark[landmark_id]
            if landmark.visibility > 0.7:  # 提高可见性阈值
                visible_count += 1
                total_visibility += landmark.visibility
        
        # 要求至少5个关键点可见
        if visible_count >= 5:
            scores['upper_body'] = (visible_count / len(upper_body_landmarks)) * (total_visibility / visible_count)
        else:
            scores['upper_body'] = 0.0
        
        # 面部朝向 (30%)
        nose = landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
        left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        if (nose.visibility > 0.5 and 
            left_shoulder.visibility > 0.5 and 
            right_shoulder.visibility > 0.5):
            
            # 计算面部是否朝向前方
            shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
            nose_deviation = abs(nose.x - shoulder_center_x)
            scores['face_orientation'] = max(0, 1.0 - nose_deviation * 5)
        
        # 手势活跃度 (20%)
        left_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        
        if left_wrist.visibility > 0.5 and right_wrist.visibility > 0.5:
            # 检查手部是否在合理的演讲区域
            wrist_y_avg = (left_wrist.y + right_wrist.y) / 2
            shoulder_y_avg = (left_shoulder.y + right_shoulder.y) / 2
            
            # 手部在肩膀附近活动得分较高
            if wrist_y_avg < shoulder_y_avg + 0.3:  # 手部不能太低
                scores['hand_gesture'] = min(1.0, (left_wrist.visibility + right_wrist.visibility) / 2)
        
        # 身体比例 (10%)
        h, w = frame_shape[:2]
        body_landmarks = [lm for lm in landmarks.landmark if lm.visibility > 0.5]
        if body_landmarks:
            # 计算身体占画面的比例
            y_coords = [lm.y for lm in body_landmarks]
            body_height = max(y_coords) - min(y_coords)
            scores['body_ratio'] = min(1.0, body_height * 2)  # 理想情况下身体占画面一半
        
        # 加权计算总分
        weights = {
            'upper_body': 0.4,
            'face_orientation': 0.3,
            'hand_gesture': 0.2,
            'body_ratio': 0.1
        }
        
        total_score = sum(scores[key] * weights[key] for key in weights)
        return total_score
    
    def _finalize_segment(self, segment_frames: List[Dict], fps: float = 30.0, video_path: str = None) -> Optional[Dict]:
        """完成片段处理 - 只提取连续的有效帧，并进行场景变化检测"""
        if not segment_frames:
            return None
        
        # 计算片段质量
        quality_scores = [f.get('pose_quality_score', f.get('quality_score', 0)) for f in segment_frames]
        avg_quality = np.mean(quality_scores)
        
        # 质量过滤 - 提高片段质量阈值
        if avg_quality < self.config.get('segment_quality_threshold', 0.7):
            return None
        
        # 🔧 简化逻辑：直接检查总检测帧数是否足够（按用户要求8秒=39帧）
        frame_indices = [f['frame_idx'] for f in segment_frames]
        
        # 直接检查是否有足够的检测帧数
        if len(frame_indices) < self.config['min_segment_length']:
            self.logger.debug(f"❌ 检测帧数不足: {len(frame_indices)} < {self.config['min_segment_length']}")
            return None
        
        # 使用所有检测帧的范围
        longest_segment = frame_indices
        
        # 🔧 改进的帧范围计算：减少安全边距，只保存真正的检测帧范围
        # 不再添加安全边距，直接使用检测帧的范围
        segment_start_frame = longest_segment[0]
        segment_end_frame = longest_segment[-1]
        
        self.logger.info(f"🔧 严格模式: 只保存检测帧范围{segment_start_frame}-{segment_end_frame}")
        
        # 🎬 场景变化检测：检查保存范围内是否有镜头切换
        if video_path and self.config.get('enable_scene_change_detection', True):
            scene_changes = self._detect_scene_changes(video_path, segment_start_frame, segment_end_frame, fps)
            if scene_changes:
                self.logger.warning(f"❌ 检测到场景变化，拒绝该片段（包含镜头切换）")
                self.logger.warning(f"   场景变化帧: {scene_changes}")
                self.logger.warning(f"   片段范围: {segment_start_frame}-{segment_end_frame}")
                return None  # 直接拒绝包含场景变化的片段
        
        # 转换为时间戳
        segment_start_time = segment_start_frame / fps
        segment_end_time = segment_end_frame / fps
        
        # 🔍 详细的帧和时间信息日志
        self.logger.info(f"📊 片段详细信息:")
        self.logger.info(f"   🎯 检测到的有效帧: {len(longest_segment)}帧")
        self.logger.info(f"   📏 帧范围: {longest_segment[0]} - {longest_segment[-1]} (共{len(longest_segment)}帧)")
        self.logger.info(f"   🎬 最终保存帧范围: {segment_start_frame} - {segment_end_frame} (共{segment_end_frame - segment_start_frame + 1}帧)")
        self.logger.info(f"   ⏰ 检测时间范围: {longest_segment[0]/fps:.2f}s - {longest_segment[-1]/fps:.2f}s")
        self.logger.info(f"   ⏰ 最终视频时间范围: {segment_start_time:.2f}s - {segment_end_time:.2f}s")
        self.logger.info(f"   ⏱️ 视频片段时长: {segment_end_time - segment_start_time:.2f}s")
        
        # 确保片段时长合理
        duration = segment_end_time - segment_start_time
        min_duration_seconds = self.config['min_segment_length'] / fps
        if duration < min_duration_seconds:
            self.logger.debug(f"❌ 连续片段时长不足: {duration:.1f}s < {min_duration_seconds:.1f}s")
            return None
        
        self.logger.info(f"✅ 有效片段: {len(longest_segment)}帧 (帧{segment_start_frame}-{segment_end_frame})")
        
        return {
            'start_frame': segment_start_frame,
            'end_frame': segment_end_frame,
            'start_time': segment_start_time,
            'end_time': segment_end_time,
            'duration': duration,
            'frame_count': len(segment_frames),
            'detection_frame_count': len(segment_frames),
            'actual_video_frames': segment_end_frame - segment_start_frame,
            'avg_quality': avg_quality,
            'max_quality': max(quality_scores),
            'quality_scores': quality_scores,
            'detection_frames': [f['frame_idx'] for f in segment_frames]
        }
    
    def _print_statistics(self):
        """打印统计信息"""
        total = self.stats['total_frames']
        person_rate = (self.stats['person_detected_frames'] / total * 100) if total > 0 else 0
        pose_rate = (self.stats['pose_detected_frames'] / total * 100) if total > 0 else 0
        valid_rate = (self.stats['valid_pose_frames'] / total * 100) if total > 0 else 0
        
        self.logger.info("📊 YOLO+MediaPipe 处理统计:")
        self.logger.info(f"  总帧数: {total}")
        self.logger.info(f"  检测到人体的帧数: {self.stats['person_detected_frames']} ({person_rate:.1f}%)")
        self.logger.info(f"  检测到姿态的帧数: {self.stats['pose_detected_frames']} ({pose_rate:.1f}%)")
        self.logger.info(f"  有效姿态帧数: {self.stats['valid_pose_frames']} ({valid_rate:.1f}%)")
    
    def cleanup(self):
        """清理资源"""
        if hasattr(self, 'pose'):
            self.pose.close()
        
        # 清理DWpose检测器
        if hasattr(self, 'dwpose_detector') and self.dwpose_detector:
            self.dwpose_detector.cleanup()
        
        # 清理场景切换检测器
        if hasattr(self, 'scene_change_detector') and self.scene_change_detector:
            self.scene_change_detector.cleanup()
        
        self.logger.info("🧹 资源清理完成") 

    def _detect_scene_changes(self, video_path: str, start_frame: int, end_frame: int, fps: float) -> List[int]:
        """
        检测场景变化/镜头切换
        返回发生镜头切换的帧索引列表
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
        
        scene_change_frames = []
        prev_hist = None
        
        # 场景变化检测参数
        hist_threshold = self.config.get('scene_change_hist_threshold', 0.3)  # 直方图相似度阈值
        edge_threshold = self.config.get('scene_change_edge_threshold', 0.4)   # 边缘密度变化阈值
        
        self.logger.info(f"🔍 场景变化检测: 帧范围 {start_frame}-{end_frame}")
        
        for frame_idx in range(start_frame, end_frame + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            
            # 方法1: 颜色直方图比较
            hist_score = 1.0
            if prev_hist is not None:
                # 计算RGB直方图
                current_hist = []
                for channel in range(3):
                    hist = cv2.calcHist([frame], [channel], None, [32], [0, 256])
                    current_hist.append(hist)
                
                # 比较直方图相似度
                hist_similarities = []
                for i in range(3):
                    similarity = cv2.compareHist(prev_hist[i], current_hist[i], cv2.HISTCMP_CORREL)
                    hist_similarities.append(similarity)
                
                hist_score = np.mean(hist_similarities)
                prev_hist = current_hist
            else:
                # 第一帧，计算初始直方图
                prev_hist = []
                for channel in range(3):
                    hist = cv2.calcHist([frame], [channel], None, [32], [0, 256])
                    prev_hist.append(hist)
                continue
            
            # 方法2: 边缘密度变化检测
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            if not hasattr(self, '_prev_edge_density'):
                self._prev_edge_density = edge_density
                continue
            
            edge_change_ratio = abs(edge_density - self._prev_edge_density) / (self._prev_edge_density + 1e-6)
            self._prev_edge_density = edge_density
            
            # 检测场景变化
            is_scene_change = (hist_score < hist_threshold) or (edge_change_ratio > edge_threshold)
            
            if is_scene_change:
                time_stamp = frame_idx / fps
                self.logger.warning(f"🎬 检测到场景变化 - 帧{frame_idx} ({time_stamp:.1f}s): 直方图相似度={hist_score:.3f}, 边缘变化={edge_change_ratio:.3f}")
                scene_change_frames.append(frame_idx)
        
        cap.release()
        
        if scene_change_frames:
            self.logger.info(f"⚠️ 共检测到 {len(scene_change_frames)} 个场景变化点: {scene_change_frames}")
        else:
            self.logger.info(f"✅ 场景连续性良好，无明显镜头切换")
        
        return scene_change_frames

    def _split_segment_by_scene_changes(self, segment: Dict, scene_changes: List[int]) -> List[Dict]:
        """根据场景变化点分割片段"""
        if not scene_changes:
            return [segment]
        
        # 找到在片段范围内的场景变化点
        start_frame = segment['start_frame']
        end_frame = segment['end_frame']
        
        relevant_changes = [sc for sc in scene_changes if start_frame < sc < end_frame]
        
        if not relevant_changes:
            return [segment]
        
        # 分割片段
        split_segments = []
        current_start = start_frame
        
        for change_point in relevant_changes:
            # 创建一个子片段（到场景变化点前）
            if change_point - current_start > 30:  # 至少30帧才保留
                sub_segment = segment.copy()
                sub_segment['start_frame'] = current_start
                sub_segment['end_frame'] = change_point - 1
                sub_segment['start_time'] = current_start / 24.0  # 使用估计FPS
                sub_segment['end_time'] = (change_point - 1) / 24.0
                sub_segment['duration'] = sub_segment['end_time'] - sub_segment['start_time']
                split_segments.append(sub_segment)
                
                self.logger.info(f"🔪 场景分割: 帧{current_start}-{change_point-1} (时长{sub_segment['duration']:.1f}s)")
            
            current_start = change_point
        
        # 处理最后一段
        if end_frame - current_start > 30:
            sub_segment = segment.copy()
            sub_segment['start_frame'] = current_start
            sub_segment['end_frame'] = end_frame
            sub_segment['start_time'] = current_start / 24.0
            sub_segment['end_time'] = end_frame / 24.0
            sub_segment['duration'] = sub_segment['end_time'] - sub_segment['start_time']
            split_segments.append(sub_segment)
            
            self.logger.info(f"🔪 场景分割: 帧{current_start}-{end_frame} (时长{sub_segment['duration']:.1f}s)")
        
        self.logger.info(f"✂️ 原片段分割为 {len(split_segments)} 个子片段")
        return split_segments 