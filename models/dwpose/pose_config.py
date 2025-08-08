
# 简化的RTMPose配置
model = dict(
    type='TopdownPoseEstimator',
    backbone=dict(type='CSPNeXt'),
    head=dict(type='RTMCCHead')
)
