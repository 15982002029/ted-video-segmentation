#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DWPose ONNX检测器 - 基于ONNX格式的DWPose模型进行多人检测
基于真实项目成功经验的实现
"""

import os
import logging
import numpy as np
from typing import Dict, Optional, List, Any, Tuple
import cv2

# 尝试导入ONNX Runtime依赖
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
    print("✅ ONNX Runtime检测成功")
except ImportError as e:
    ONNX_AVAILABLE = False
    print(f"警告: ONNX Runtime未安装 - {e}")
    print("请运行 pip install onnxruntime-gpu==1.20.1")

# 尝试导入torch用于一些工具函数
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class DWPoseONNXDetector:
    """
    DWPose ONNX检测器
    
    基于ONNX格式的DWPose模型，提供高效的人体姿态检测和多人场景过滤
    专为六关卡筛选系统设计
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化DWPose ONNX检测器
        
        Args:
            config: 配置字典，包含dwpose_multi_person部分
        """
        self.logger = logging.getLogger(__name__)
        
        # 加载配置
        self.dwpose_config = config.get('dwpose_multi_person', {})
        self.enable_detection = self.dwpose_config.get('enable_dwpose_detection', True)
        
        # 模型路径配置
        self.model_dir = self.dwpose_config.get('dwpose_model_path', 'models/dwpose')
        self.det_model_path = os.path.join(self.model_dir, 'yolox_l.onnx')
        self.pose_model_path = os.path.join(self.model_dir, 'dw-ll_ucoco_384.onnx')
        
        # 设备配置
        self.device = self.dwpose_config.get('dwpose_device', 'cuda')
        
        # 检测参数（从配置文件读取，提供默认值）
        self.max_persons_per_frame = self.dwpose_config.get('max_persons_per_frame', 1)
        self.person_confidence_threshold = self.dwpose_config.get('person_confidence_threshold', 0.3)  # 降低阈值提高灵敏度
        self.keypoint_confidence_threshold = self.dwpose_config.get('keypoint_confidence_threshold', 0.3)  # 关键点置信度阈值
        self.nms_threshold = self.dwpose_config.get('nms_threshold', 0.45)  # NMS阈值
        
        # 统计相关参数
        self.enable_detailed_stats = self.dwpose_config.get('enable_detailed_stats', True)
        self.log_detection_results = self.dwpose_config.get('log_detection_results', False)
        
        # 初始化统计信息
        self.stats = {
            'total_frames_processed': 0,
            'single_person_frames': 0,
            'multi_person_frames': 0,
            'no_person_frames': 0,
            'total_persons_detected': 0,
            'detection_failures': 0
        }
        
        # ONNX会话
        self.det_session = None
        self.pose_session = None
        
        # 初始化检测器
        self._initialize_models()
        
        self.logger.info(f"🤖 DWPose ONNX检测器初始化完成")
        self.logger.info(f"   启用状态: {self.enable_detection}")
        self.logger.info(f"   最大人数: {self.max_persons_per_frame}")
        self.logger.info(f"   人体检测阈值: {self.person_confidence_threshold}")
        self.logger.info(f"   关键点阈值: {self.keypoint_confidence_threshold}")
        self.logger.info(f"   NMS阈值: {self.nms_threshold}")
        self.logger.info(f"   ONNX可用: {ONNX_AVAILABLE}")
        self.logger.info(f"   设备: {self.device}")
    
    def _initialize_models(self):
        """初始化ONNX模型"""
        if not self.enable_detection:
            self.logger.info("DWPose检测已禁用")
            return
            
        if not ONNX_AVAILABLE:
            self.logger.warning("ONNX Runtime不可用，请安装: pip install onnxruntime-gpu==1.20.1")
            return
        
        try:
            # 检查模型文件是否存在
            if not os.path.exists(self.det_model_path):
                self.logger.warning(f"检测模型文件不存在: {self.det_model_path}")
                self.logger.info("请运行 python download_dwpose_onnx.py 下载模型")
                return
                
            if not os.path.exists(self.pose_model_path):
                self.logger.warning(f"姿态模型文件不存在: {self.pose_model_path}")
                self.logger.info("请运行 python download_dwpose_onnx.py 下载模型")
                return
            
            # 设置ONNX执行提供者
            if self.device == 'cpu':
                providers = ['CPUExecutionProvider']
                provider_options = [{}]
            else:
                # 检查CUDA可用性
                available_providers = ort.get_available_providers()
                if 'CUDAExecutionProvider' in available_providers:
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                    provider_options = [
                        {'device_id': 0, 'arena_extend_strategy': 'kNextPowerOfTwo'},
                        {}
                    ]
                    self.logger.info("✅ 使用CUDA执行提供者")
                else:
                    providers = ['CPUExecutionProvider']
                    provider_options = [{}]
                    self.logger.warning("CUDA不可用，使用CPU执行")
            
            # 初始化ONNX会话
            self.logger.info("🔄 正在加载DWPose ONNX模型...")
            
            # 检测模型会话
            self.det_session = ort.InferenceSession(
                self.det_model_path,
                providers=providers,
                provider_options=provider_options
            )
            
            # 姿态估计模型会话
            self.pose_session = ort.InferenceSession(
                self.pose_model_path,
                providers=providers,
                provider_options=provider_options
            )
            
            self.logger.info("✅ DWPose ONNX模型加载成功")
            
            # 打印模型信息
            det_inputs = self.det_session.get_inputs()
            pose_inputs = self.pose_session.get_inputs()
            self.logger.info(f"   检测模型输入: {[inp.name + str(inp.shape) for inp in det_inputs]}")
            self.logger.info(f"   姿态模型输入: {[inp.name + str(inp.shape) for inp in pose_inputs]}")
            
        except Exception as e:
            self.logger.error(f"DWPose ONNX模型初始化失败: {e}")
            self.det_session = None
            self.pose_session = None
    
    def detect_upper_body_keypoints(self, frame: np.ndarray, yolo_detections: List[Dict] = None) -> Dict[str, Any]:
        """
        检测帧中的人数并返回检测结果
        
        Args:
            frame: 输入视频帧 (H, W, C) BGR格式
            yolo_detections: 外部YOLO检测结果列表 (可选)
            
        Returns:
            Dict: 包含检测结果的字典
                - is_single_person: bool, 是否为单人
                - person_count: int, 检测到的人数
                - detection_confidence: float, 检测置信度
                - message: str, 检测信息
                - dwpose_result: 具体的检测结果
        """
        self.stats['total_frames_processed'] += 1
        
        if not self.enable_detection:
            return {
                'is_single_person': True,
                'person_count': 1,
                'detection_confidence': 1.0,
                'message': "DWPose检测已禁用",
                'dwpose_result': None
            }
        
        # 严格模式：模型必须可用
        if not ONNX_AVAILABLE:
            raise RuntimeError("ONNX Runtime不可用，无法进行DWPose检测")
        
        if self.det_session is None or self.pose_session is None:
            raise RuntimeError("DWPose模型未正确加载，请检查模型文件")
        
        try:
            # 使用外部YOLO检测结果或内部检测
            if yolo_detections is not None:
                # 使用外部YOLO检测结果进行姿态估计
                dwpose_result = self._run_dwpose_with_external_detections(frame, yolo_detections)
            else:
                # 执行完整的DWPose检测（包含内部YOLOX检测）
                dwpose_result = self._run_dwpose_detection(frame)
            
            # 严格模式：检测失败时立即报错
            if dwpose_result is None:
                self.stats['detection_failures'] += 1
                raise RuntimeError("DWPose检测失败，无法获取检测结果")
            
            # 分析检测结果 - 统计有效人数
            valid_persons = dwpose_result.get('bodies', {}).get('valid_persons', [])
            person_count = len(valid_persons)
            self.stats['total_persons_detected'] += person_count
            
            # 判断是否为单人
            is_single_person = (person_count == 1)
            
            # 更新统计
            if person_count == 0:
                self.stats['no_person_frames'] += 1
                message = "未检测到人体"
            elif person_count == 1:
                self.stats['single_person_frames'] += 1
                message = "检测到单人，符合要求"
            else:
                self.stats['multi_person_frames'] += 1
                message = f"检测到{person_count}人，超出限制"
            
            if self.log_detection_results:
                self.logger.debug(f"DWPose检测: {message}")
            
            return {
                'is_single_person': is_single_person,
                'person_count': person_count,
                'detection_confidence': dwpose_result.get('confidence', 0.8),
                'message': message,
                'dwpose_result': dwpose_result
            }
            
        except Exception as e:
            self.stats['detection_failures'] += 1
            self.logger.error(f"DWPose检测过程出错: {e}")
            # 严格模式：不允许默认通过，直接抛出异常
            raise RuntimeError(f"DWPose检测异常: {e}")
    
    def _run_dwpose_with_external_detections(self, frame: np.ndarray, yolo_detections: List[Dict]) -> Optional[Dict]:
        """
        使用外部YOLO检测结果进行DWpose姿态估计
        
        Args:
            frame: 输入图像
            yolo_detections: YOLO检测结果，格式为 [{'bbox': [x1,y1,x2,y2], 'confidence': float, 'class_id': int}]
            
        Returns:
            DWPose检测结果字典
        """
        try:
            # 过滤出人体检测（class_id == 0）
            person_detections = []
            for det in yolo_detections:
                if det.get('class_id', 0) == 0:  # 人体类别
                    # 转换YOLO格式到DWpose格式
                    x1, y1, x2, y2 = det['bbox']
                    person_detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': det.get('confidence', 0.8),
                        'class_id': 0
                    })
            
            if len(person_detections) == 0:
                self.logger.debug("外部检测未发现人体")
                return {
                    'bodies': {'valid_persons': []},
                    'confidence': 0.0,
                    'detection_source': 'external_yolo'
                }
            
            # 对检测到的每个人体进行姿态估计
            pose_results = self._inference_pose(frame, person_detections)
            
            # 构建结果
            valid_persons = []
            total_confidence = 0.0
            
            for i, pose_result in enumerate(pose_results):
                if pose_result and len(pose_result.get('keypoints', [])) > 0:
                    valid_persons.append({
                        'person_id': i,
                        'bbox': person_detections[i]['bbox'],
                        'keypoints': pose_result['keypoints'],
                        'confidence': pose_result.get('confidence', 0.8)
                    })
                    total_confidence += pose_result.get('confidence', 0.8)
            
            avg_confidence = total_confidence / len(valid_persons) if valid_persons else 0.0
            
            return {
                'bodies': {
                    'valid_persons': valid_persons,
                    'total_detected': len(person_detections),
                    'successful_poses': len(valid_persons)
                },
                'confidence': avg_confidence,
                'detection_source': 'external_yolo'
            }
            
        except Exception as e:
            self.logger.error(f"外部检测结果处理失败: {e}")
            return None

    def _run_dwpose_detection(self, frame: np.ndarray) -> Optional[Dict]:
        """
        运行完整的DWPose检测流程
        
        Args:
            frame: 输入图像 (H, W, C) BGR格式
            
        Returns:
            Dict: 检测结果，包含bodies等信息
        """
        try:
            # Step 1: 人体检测
            det_results = self._inference_detection(frame)
            if not det_results:
                return None
            
            # Step 2: 姿态估计
            pose_results = self._inference_pose(frame, det_results)
            if not pose_results:
                return None
            
            # Step 3: 后处理和组织结果
            final_result = self._process_detection_results(pose_results, det_results)
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"DWPose检测流程失败: {e}")
            return None
    
    def _inference_detection(self, frame: np.ndarray) -> List[Dict]:
        """
        执行人体检测推理
        
        Args:
            frame: 输入图像
            
        Returns:
            List[Dict]: 检测到的人体边界框列表
        """
        try:
            # 图像预处理
            input_size = (640, 640)  # YOLOX标准输入尺寸
            processed_img, scale, pad = self._preprocess_detection_input(frame, input_size)
            
            # ONNX推理
            input_name = self.det_session.get_inputs()[0].name
            outputs = self.det_session.run(None, {input_name: processed_img})
            
            # 后处理检测结果
            if self.log_detection_results:
                self.logger.debug(f"YOLOX原始输出形状: {outputs[0].shape}")
                self.logger.debug(f"输出数据范围: min={outputs[0].min():.4f}, max={outputs[0].max():.4f}")
            detections = self._postprocess_detection_output(outputs[0], scale, pad, frame.shape[:2])
            
            # 过滤人体检测 (class_id = 0) 并应用NMS
            person_detections = []
            total_person_candidates = 0
            for det in detections:
                if det['class_id'] == 0:  # 人体类别
                    total_person_candidates += 1
                    if det['confidence'] > self.person_confidence_threshold:
                        person_detections.append(det)
            
            # 添加调试日志
            if self.log_detection_results:
                self.logger.debug(f"DWPose检测: 原始检测{len(detections)}个, 人体候选{total_person_candidates}个, 通过阈值{len(person_detections)}个")
            
            # 应用非极大值抑制
            if len(person_detections) > 1:
                person_detections = self._apply_nms(person_detections)
            
            return person_detections
            
        except Exception as e:
            self.logger.error(f"人体检测推理失败: {e}")
            return []
    
    def _inference_pose(self, frame: np.ndarray, detections: List[Dict]) -> List[Dict]:
        """
        执行姿态估计推理
        
        Args:
            frame: 输入图像
            detections: 人体检测结果
            
        Returns:
            List[Dict]: 姿态估计结果列表
        """
        pose_results = []
        
        try:
            for i, det in enumerate(detections):
                # 提取人体区域
                bbox = det['bbox']
                person_img = self._crop_person_region(frame, bbox)
                
                if person_img is None or person_img.size == 0:
                    continue
                
                # 姿态估计预处理
                processed_img = self._preprocess_pose_input(person_img)
                
                # ONNX推理
                input_name = self.pose_session.get_inputs()[0].name
                outputs = self.pose_session.run(None, {input_name: processed_img})
                
                # 后处理姿态结果
                keypoints = self._postprocess_pose_output(outputs, bbox, frame.shape[:2])
                
                pose_result = {
                    'person_id': i,
                    'bbox': bbox,
                    'keypoints': keypoints,
                    'confidence': det['confidence']
                }
                
                pose_results.append(pose_result)
                
        except Exception as e:
            self.logger.error(f"姿态估计推理失败: {e}")
            
        return pose_results
    
    def _preprocess_detection_input(self, frame: np.ndarray, input_size: Tuple[int, int]) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        检测模型输入预处理
        
        Args:
            frame: 输入图像 (H, W, C)
            input_size: 目标尺寸 (width, height)
            
        Returns:
            Tuple: (预处理后的图像, 缩放比例, 填充值)
        """
        h, w = frame.shape[:2]
        input_w, input_h = input_size
        
        # 计算缩放比例，保持长宽比
        scale = min(input_w / w, input_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # 缩放图像
        resized = cv2.resize(frame, (new_w, new_h))
        
        # 创建填充后的图像
        padded = np.full((input_h, input_w, 3), 114, dtype=np.uint8)
        
        # 计算填充位置（居中）
        pad_x = (input_w - new_w) // 2
        pad_y = (input_h - new_h) // 2
        padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        
        # 转换为CHW格式并归一化
        input_img = padded.transpose(2, 0, 1).astype(np.float32) / 255.0
        input_img = np.expand_dims(input_img, axis=0)  # 添加batch维度
        
        return input_img, scale, (pad_x, pad_y)
    
    def _postprocess_detection_output(self, output: np.ndarray, scale: float, pad: Tuple[int, int], original_shape: Tuple[int, int]) -> List[Dict]:
        """
        后处理检测输出
        
        Args:
            output: ONNX模型原始输出
            scale: 缩放比例
            pad: 填充值 (pad_x, pad_y)
            original_shape: 原始图像尺寸 (height, width)
            
        Returns:
            List[Dict]: 解析后的检测结果
        """
        detections = []
        pad_x, pad_y = pad
        
        try:
            if len(output.shape) == 3:
                output = output[0]  # 移除batch维度
            
            # 应用置信度阈值 (使用动态阈值避免过早过滤)
            # YOLOX输出需要sigmoid激活
            obj_conf = 1 / (1 + np.exp(-output[:, 4]))  # sigmoid激活
            # 使用更低的预处理阈值，让更多候选进入后续处理
            pre_threshold = min(0.01, self.person_confidence_threshold * 0.1)
            valid_mask = obj_conf > pre_threshold
            valid_output = output[valid_mask]
            
            # 调试信息
            if hasattr(self, 'log_detection_results') and self.log_detection_results:
                self.logger.debug(f"后处理: 输入shape={output.shape}, 置信度范围={obj_conf.min():.4f}-{obj_conf.max():.4f}")
                self.logger.debug(f"预处理阈值: {pre_threshold:.4f}, 通过候选: {len(valid_output)}")
            
            if len(valid_output) == 0:
                return detections
            
            # 解析边界框和类别信息
            boxes = valid_output[:, :4]
            # 重新计算置信度（从有效输出中）
            confidences = 1 / (1 + np.exp(-valid_output[:, 4]))  # sigmoid激活
            class_probs = valid_output[:, 5:] if valid_output.shape[1] > 5 else np.ones((len(valid_output), 1))
            
            # 对类别概率也进行sigmoid激活
            if class_probs.shape[1] > 1:
                class_probs = 1 / (1 + np.exp(-class_probs))  # sigmoid激活
            
            # 转换bbox格式 (center_x, center_y, width, height -> x1, y1, x2, y2)
            x_center, y_center, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            
            # 转换到原图坐标
            x1 = (x_center - width / 2 - pad_x) / scale
            y1 = (y_center - height / 2 - pad_y) / scale
            x2 = (x_center + width / 2 - pad_x) / scale
            y2 = (y_center + height / 2 - pad_y) / scale
            
            # 获取类别和最终置信度
            if class_probs.shape[1] > 1:
                class_ids = np.argmax(class_probs, axis=1)
                class_confidences = np.max(class_probs, axis=1)
                final_confidences = confidences * class_confidences
            else:
                class_ids = np.zeros(len(confidences), dtype=int)
                final_confidences = confidences
            
            # 构建检测结果
            for i in range(len(x1)):
                # 确保边界框在图像范围内
                x1_clipped = max(0, min(float(x1[i]), original_shape[1] - 1))
                y1_clipped = max(0, min(float(y1[i]), original_shape[0] - 1))
                x2_clipped = max(x1_clipped + 1, min(float(x2[i]), original_shape[1]))
                y2_clipped = max(y1_clipped + 1, min(float(y2[i]), original_shape[0]))
                
                detections.append({
                    'bbox': [x1_clipped, y1_clipped, x2_clipped, y2_clipped],
                    'confidence': float(final_confidences[i]),
                    'class_id': int(class_ids[i])
                })
            
        except Exception as e:
            self.logger.error(f"检测输出后处理失败: {e}")
            
        return detections
    
    def _apply_nms(self, detections: List[Dict]) -> List[Dict]:
        """
        应用非极大值抑制
        
        Args:
            detections: 检测结果列表
            
        Returns:
            List[Dict]: NMS后的检测结果
        """
        if len(detections) <= 1:
            return detections
        
        try:
            # 提取边界框和置信度
            boxes = np.array([det['bbox'] for det in detections])
            scores = np.array([det['confidence'] for det in detections])
            
            # 计算面积
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            
            # 按置信度排序
            order = scores.argsort()[::-1]
            
            keep = []
            while len(order) > 0:
                i = order[0]
                keep.append(i)
                
                if len(order) == 1:
                    break
                
                # 计算IoU
                xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
                yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
                xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
                yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
                
                w = np.maximum(0, xx2 - xx1)
                h = np.maximum(0, yy2 - yy1)
                intersection = w * h
                
                union = areas[i] + areas[order[1:]] - intersection
                iou = intersection / (union + 1e-6)
                
                # 保留IoU小于阈值的检测
                inds = np.where(iou <= self.nms_threshold)[0]
                order = order[inds + 1]
            
            return [detections[i] for i in keep]
            
        except Exception as e:
            self.logger.error(f"NMS处理失败: {e}")
            return detections
    
    def _crop_person_region(self, frame: np.ndarray, bbox: List[float]) -> Optional[np.ndarray]:
        """
        根据边界框裁剪人体区域
        
        Args:
            frame: 输入图像
            bbox: 边界框 [x1, y1, x2, y2]
            
        Returns:
            Optional[np.ndarray]: 裁剪后的人体区域图像
        """
        try:
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = bbox
            
            # 确保坐标在有效范围内
            x1 = max(0, min(int(x1), w - 1))
            y1 = max(0, min(int(y1), h - 1))
            x2 = max(x1 + 1, min(int(x2), w))
            y2 = max(y1 + 1, min(int(y2), h))
            
            # 检查裁剪区域是否有效
            if x2 <= x1 or y2 <= y1:
                return None
            
            return frame[y1:y2, x1:x2]
            
        except Exception as e:
            self.logger.error(f"人体区域裁剪失败: {e}")
            return None
    
    def _preprocess_pose_input(self, person_img: np.ndarray) -> np.ndarray:
        """
        姿态估计模型输入预处理
        
        Args:
            person_img: 人体区域图像
            
        Returns:
            np.ndarray: 预处理后的图像
        """
        # DWPose姿态模型输入尺寸：从模型signature得知 [batch, 3, 384, 288]
        # 即 height=384, width=288
        input_height, input_width = 384, 288
        
        # 缩放图像 (cv2.resize参数顺序为 (width, height))
        resized = cv2.resize(person_img, (input_width, input_height))
        
        # 归一化到[-1, 1]
        normalized = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
        
        # 转换为CHW格式
        input_img = normalized.transpose(2, 0, 1)
        input_img = np.expand_dims(input_img, axis=0)  # 添加batch维度
        
        # 调试信息
        if hasattr(self, 'log_detection_results') and self.log_detection_results:
            self.logger.debug(f"姿态输入预处理: 原始{person_img.shape} -> 缩放{resized.shape} -> 最终{input_img.shape}")
        
        return input_img
    
    def _postprocess_pose_output(self, outputs: List[np.ndarray], bbox: List[float], original_shape: Tuple[int, int]) -> Dict:
        """
        后处理姿态估计输出
        
        Args:
            outputs: ONNX模型输出列表
            bbox: 人体边界框
            original_shape: 原始图像尺寸
            
        Returns:
            Dict: 解析后的关键点信息
        """
        try:
            # 调试信息
            if hasattr(self, 'log_detection_results') and self.log_detection_results:
                self.logger.debug(f"姿态输出后处理: outputs={len(outputs)}, bbox={bbox}, bbox_type={type(bbox)}")
                if len(outputs) > 0:
                    self.logger.debug(f"关键点数据形状: {outputs[0].shape}")
            
            # DWPose输出通常包含身体关键点
            if len(outputs) == 0:
                return {'body': np.array([]), 'bbox': bbox}
            
            # 获取关键点数据
            keypoints_data = outputs[0]
            if len(keypoints_data.shape) == 3:
                keypoints_data = keypoints_data[0]  # 移除batch维度 -> shape: (133, 576)
            
            # DWPose格式：133个关键点，每个关键点可能有多维数据
            # 通常前17个是身体关键点，我们主要关注这些
            num_body_keypoints = min(17, keypoints_data.shape[0])
            
            body_points = []
            
            # 确保bbox是列表或有足够的元素
            if len(bbox) >= 4:
                x1, y1, x2, y2 = bbox[:4]
                bbox_w, bbox_h = x2 - x1, y2 - y1
            else:
                # 如果bbox格式不正确，使用原始坐标
                x1, y1, x2, y2 = 0, 0, 1, 1
                bbox_w, bbox_h = 1, 1
            
            # 解析身体关键点（前17个）
            for i in range(num_body_keypoints):
                try:
                    # DWPose可能输出热图或直接坐标
                    # 检查实际的数据格式
                    if hasattr(self, 'log_detection_results') and self.log_detection_results and i == 0:
                        self.logger.debug(f"关键点{i}原始数据前10个值: {keypoints_data[i, :10]}")
                    
                    # 假设前两个值是x, y坐标，第三个是置信度
                    if keypoints_data.shape[1] >= 3:
                        x = float(keypoints_data[i, 0])
                        y = float(keypoints_data[i, 1])
                        conf = float(keypoints_data[i, 2]) if keypoints_data.shape[1] > 2 else 1.0
                        
                        # 确保置信度在合理范围内
                        if conf < 0:
                            conf = abs(conf)  # 取绝对值
                        if conf > 100:
                            conf = conf / 100  # 可能是百分比格式
                    else:
                        # 如果不是标准格式，跳过
                        continue
                    
                    # 关键点坐标通常是相对于输入图像的归一化坐标
                    # 转换到原图坐标
                    abs_x = x * bbox_w + x1
                    abs_y = y * bbox_h + y1
                    
                    body_points.append([abs_x, abs_y, conf])
                    
                    if hasattr(self, 'log_detection_results') and self.log_detection_results and i < 3:
                        self.logger.debug(f"关键点{i}: 原始({x:.3f}, {y:.3f}, {conf:.3f}) -> 转换({abs_x:.3f}, {abs_y:.3f}, {conf:.3f})")
                    
                except Exception as e:
                    if hasattr(self, 'log_detection_results') and self.log_detection_results:
                        self.logger.debug(f"关键点{i}解析失败: {e}")
                    continue
            
            return {
                'body': np.array(body_points),
                'bbox': bbox
            }
            
        except Exception as e:
            self.logger.error(f"姿态输出后处理失败: {e}")
            return {
                'body': np.array([]),
                'bbox': bbox
            }
    
    def _process_detection_results(self, pose_results: List[Dict], det_results: List[Dict]) -> Dict:
        """
        处理和组织最终检测结果
        
        Args:
            pose_results: 姿态估计结果
            det_results: 人体检测结果
            
        Returns:
            Dict: 组织后的最终结果
        """
        valid_persons = []
        
        for pose_result in pose_results:
            keypoints = pose_result['keypoints']
            
            # 检查关键点质量
            if 'body' in keypoints and len(keypoints['body']) > 0:
                body_points = keypoints['body']
                
                # 调试信息
                if hasattr(self, 'log_detection_results') and self.log_detection_results:
                    self.logger.debug(f"姿态质量检查: 总关键点{len(body_points)}, 阈值{self.keypoint_confidence_threshold}")
                    if len(body_points) > 0:
                        confidences = body_points[:, 2] if body_points.shape[1] > 2 else []
                        self.logger.debug(f"置信度范围: {confidences.min():.3f}-{confidences.max():.3f}")
                
                # 检查有效关键点数量
                valid_points = body_points[body_points[:, 2] > self.keypoint_confidence_threshold]
                
                if hasattr(self, 'log_detection_results') and self.log_detection_results:
                    self.logger.debug(f"有效关键点: {len(valid_points)}/{len(body_points)}")
                
                # 如果有足够的有效关键点，认为是有效人体
                if len(valid_points) >= 5:  # 至少5个有效关键点
                    valid_persons.append({
                        'person_id': pose_result['person_id'],
                        'bbox': pose_result['bbox'],
                        'keypoints': keypoints,
                        'confidence': pose_result['confidence'],
                        'valid_keypoints_count': len(valid_points)
                    })
                    if hasattr(self, 'log_detection_results') and self.log_detection_results:
                        self.logger.debug(f"✅ 人体{pose_result['person_id']}通过质量检查")
                else:
                    if hasattr(self, 'log_detection_results') and self.log_detection_results:
                        self.logger.debug(f"❌ 人体{pose_result['person_id']}未通过质量检查: 有效关键点{len(valid_points)}<5")
        
        return {
            'bodies': {
                'valid_persons': valid_persons,
                'total_detections': len(det_results)
            },
            'confidence': np.mean([p['confidence'] for p in valid_persons]) if valid_persons else 0.0,
            'detection_info': {
                'total_detected': len(det_results),
                'valid_poses': len(valid_persons)
            }
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取检测统计信息
        
        Returns:
            Dict: 统计信息字典
        """
        stats = self.stats.copy()
        
        if stats['total_frames_processed'] > 0:
            stats['single_person_rate'] = stats['single_person_frames'] / stats['total_frames_processed']
            stats['multi_person_rate'] = stats['multi_person_frames'] / stats['total_frames_processed']
            stats['no_person_rate'] = stats['no_person_frames'] / stats['total_frames_processed']
            stats['failure_rate'] = stats['detection_failures'] / stats['total_frames_processed']
            stats['average_persons_per_frame'] = stats['total_persons_detected'] / stats['total_frames_processed']
        else:
            stats['single_person_rate'] = 0.0
            stats['multi_person_rate'] = 0.0
            stats['no_person_rate'] = 0.0
            stats['failure_rate'] = 0.0
            stats['average_persons_per_frame'] = 0.0
        
        return stats
    
    def print_statistics(self):
        """打印检测统计信息"""
        if not self.enable_detailed_stats:
            return
        
        stats = self.get_statistics()
        
        self.logger.info("=" * 50)
        self.logger.info("📊 DWPose ONNX检测统计")
        self.logger.info("=" * 50)
        self.logger.info(f"处理帧数: {stats['total_frames_processed']}")
        self.logger.info(f"单人帧数: {stats['single_person_frames']} ({stats['single_person_rate']:.2%})")
        self.logger.info(f"多人帧数: {stats['multi_person_frames']} ({stats['multi_person_rate']:.2%})")
        self.logger.info(f"无人帧数: {stats['no_person_frames']} ({stats['no_person_rate']:.2%})")
        self.logger.info(f"检测失败: {stats['detection_failures']} ({stats['failure_rate']:.2%})")
        self.logger.info(f"检测到的总人数: {stats['total_persons_detected']}")
        self.logger.info(f"平均每帧人数: {stats['average_persons_per_frame']:.2f}")
        self.logger.info("=" * 50)
    
    def reset_statistics(self):
        """重置统计信息"""
        self.stats = {
            'total_frames_processed': 0,
            'single_person_frames': 0,
            'multi_person_frames': 0,
            'no_person_frames': 0,
            'total_persons_detected': 0,
            'detection_failures': 0
        }
        self.logger.info("📊 DWPose ONNX检测统计信息已重置")
    
    def cleanup(self):
        """清理资源"""
        if hasattr(self, 'det_session') and self.det_session is not None:
            del self.det_session
            self.det_session = None
        
        if hasattr(self, 'pose_session') and self.pose_session is not None:
            del self.pose_session
            self.pose_session = None
        
        self.logger.info("🧹 DWPose ONNX检测器资源已清理")