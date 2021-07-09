# optimizer
optimizer = dict(type='AdamW', lr=1e-4,  betas=(0.9, 0.999), weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=4,
    warmup_ratio=0.1,
    min_lr=1e-6,
    warmup_by_epoch=True)
runner = dict(type='EpochBasedRunner', max_epochs=100)
