_base_ = 'faster-rcnn_r50-caffe_fpn_ms-1x_coco.py'

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[60000, 80000])

# Runner type
runner = dict(_delete_=True, type='IterBasedRunner', max_iters=90000)

checkpoint_config = dict(interval=10000)
evaluation = dict(interval=10000, metric='bbox')
