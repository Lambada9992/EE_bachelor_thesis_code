auto_scale_lr = dict(base_batch_size=96)
data_preprocessor = dict(
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    num_classes=1000,
    std=[
        58.395,
        57.12,
        57.375,
    ],
    to_rgb=True)
dataset_type = 'ImageNet'
default_hooks = dict(
    checkpoint=dict(_scope_='mmcls', interval=1, type='CheckpointHook'),
    logger=dict(_scope_='mmcls', interval=100, type='LoggerHook'),
    param_scheduler=dict(_scope_='mmcls', type='ParamSchedulerHook'),
    sampler_seed=dict(_scope_='mmcls', type='DistSamplerSeedHook'),
    timer=dict(_scope_='mmcls', type='IterTimerHook'),
    visualization=dict(
        _scope_='mmcls', enable=False, type='VisualizationHook'))
default_scope = 'mmcls'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
find_unused_parameters = True
launcher = 'none'
load_from = None
log_level = 'INFO'
model = dict(
    _scope_='mmrazor',
    architecture=dict(
        _scope_='mmcls',
        backbone=dict(type='MobileNetV2', widen_factor=1.0),
        head=dict(
            in_channels=1280,
            loss=dict(loss_weight=1.0, type='CrossEntropyLoss'),
            num_classes=1000,
            topk=(
                1,
                5,
            ),
            type='LinearClsHead'),
        neck=dict(type='GlobalAveragePooling'),
        type='ImageClassifier'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='ImgDataPreprocessor'),
    distiller=dict(
        distill_losses=dict(
            loss_kl=dict(loss_weight=3, tau=1, type='KLDivergence')),
        loss_forward_mappings=dict(
            loss_kl=dict(
                preds_S=dict(from_student=True, recorder='fc'),
                preds_T=dict(from_student=False, recorder='fc'))),
        student_recorders=dict(
            fc=dict(source='head.fc', type='ModuleOutputs')),
        teacher_recorders=dict(
            fc=dict(source='head.fc', type='ModuleOutputs')),
        type='ConfigurableDistiller'),
    teacher=dict(
        cfg_path='mmcls::densenet/densenet201_4xb256_in1k.py',
        pretrained=False),
    teacher_ckpt=
    'https://download.openmmlab.com/mmclassification/v0/densenet/densenet201_4xb256_in1k_20220426-05cae4ef.pth',
    type='SingleTeacherDistill')
optim_wrapper = dict(
    optimizer=dict(
        _scope_='mmcls',
        lr=0.045,
        momentum=0.9,
        type='SGD',
        weight_decay=4e-05))
param_scheduler = dict(
    _scope_='mmcls', by_epoch=True, gamma=0.98, step_size=1, type='StepLR')
randomness = dict(deterministic=False, seed=1234)
resume = True
student = dict(
    _scope_='mmcls',
    backbone=dict(type='MobileNetV2', widen_factor=1.0),
    head=dict(
        in_channels=1280,
        loss=dict(loss_weight=1.0, type='CrossEntropyLoss'),
        num_classes=1000,
        topk=(
            1,
            5,
        ),
        type='LinearClsHead'),
    neck=dict(type='GlobalAveragePooling'),
    type='ImageClassifier')
teacher_ckpt = 'https://download.openmmlab.com/mmclassification/v0/densenet/densenet201_4xb256_in1k_20220426-05cae4ef.pth'
test_cfg = dict()
test_dataloader = dict(
    batch_size=96,
    dataset=dict(
        _scope_='mmcls',
        ann_file='meta/val.txt',
        data_prefix='val',
        data_root='data/imagenet',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(backend='pillow', edge='short', scale=256, type='ResizeEdge'),
            dict(crop_size=224, type='CenterCrop'),
            dict(type='PackClsInputs'),
        ],
        type='ImageNet'),
    num_workers=10,
    sampler=dict(_scope_='mmcls', shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    _scope_='mmcls', topk=(
        1,
        5,
    ), type='Accuracy')
test_pipeline = [
    dict(_scope_='mmcls', type='LoadImageFromFile'),
    dict(
        _scope_='mmcls',
        backend='pillow',
        edge='short',
        scale=256,
        type='ResizeEdge'),
    dict(_scope_='mmcls', crop_size=224, type='CenterCrop'),
    dict(_scope_='mmcls', type='PackClsInputs'),
]
train_cfg = dict(by_epoch=True, max_epochs=300, val_interval=1)
train_dataloader = dict(
    batch_size=96,
    dataset=dict(
        _scope_='mmcls',
        ann_file='meta/train.txt',
        data_prefix='train',
        data_root='data/imagenet',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(backend='pillow', scale=224, type='RandomResizedCrop'),
            dict(direction='horizontal', prob=0.5, type='RandomFlip'),
            dict(type='PackClsInputs'),
        ],
        type='ImageNet'),
    num_workers=10,
    sampler=dict(_scope_='mmcls', shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(_scope_='mmcls', type='LoadImageFromFile'),
    dict(
        _scope_='mmcls', backend='pillow', scale=224,
        type='RandomResizedCrop'),
    dict(_scope_='mmcls', direction='horizontal', prob=0.5, type='RandomFlip'),
    dict(_scope_='mmcls', type='PackClsInputs'),
]
val_cfg = dict(type='mmrazor.SingleTeacherDistillValLoop')
val_dataloader = dict(
    batch_size=96,
    dataset=dict(
        _scope_='mmcls',
        ann_file='meta/val.txt',
        data_prefix='val',
        data_root='data/imagenet',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(backend='pillow', edge='short', scale=256, type='ResizeEdge'),
            dict(crop_size=224, type='CenterCrop'),
            dict(type='PackClsInputs'),
        ],
        type='ImageNet'),
    num_workers=10,
    sampler=dict(_scope_='mmcls', shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    _scope_='mmcls', topk=(
        1,
        5,
    ), type='Accuracy')
vis_backends = [
    dict(_scope_='mmcls', type='LocalVisBackend'),
]
visualizer = dict(
    _scope_='mmcls',
    type='ClsVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = 'my_kd'
