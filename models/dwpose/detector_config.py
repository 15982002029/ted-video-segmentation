
# 简化的YOLOv8检测器配置
model = dict(
    type='YOLOX',
    backbone=dict(type='CSPDarknet'),
    neck=dict(type='YOLOXPAFPN'),
    bbox_head=dict(type='YOLOXHead')
)
