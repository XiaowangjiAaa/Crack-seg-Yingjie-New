"""Fail installation if binary ops or optional model registries cannot load."""
import os
import site
import sys


def main():
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError('The supported installation environment is Python 3.10')
    import torch
    if getattr(site, 'ENABLE_USER_SITE', False):
        user_site = os.path.realpath(site.getusersitepackages())
        if user_site in os.path.realpath(torch.__file__):
            raise RuntimeError(
                'torch was imported from the user site-packages ({}) instead of this environment. '
                'A stale user-level install shadows the environment. Run with PYTHONNOUSERSITE=1, '
                'or remove the torch/torchvision copy under that directory.'.format(user_site)
            )
    import torchvision
    import mmcv
    import mmengine
    import mmseg
    import mmpretrain
    import mmdet
    import tensorboard
    from mmcv.ops import nms, point_sample, MultiScaleDeformableAttention
    from mmseg.utils import register_all_modules
    register_all_modules()
    import mmpretrain.models
    import mmdet.models
    import simple_seg.dataset
    import simple_seg.hooks
    import simple_seg.logging_hooks
    boxes = torch.tensor([[0., 0., 4., 4.], [0., 0., 4., 4.]])
    _, keep = nms(boxes, torch.tensor([0.9, 0.8]), 0.5)
    assert keep.numel() == 1, 'MMCV NMS smoke check failed'
    for module in (torch, torchvision, mmcv, mmengine, mmseg, mmpretrain, mmdet, tensorboard):
        print('{}: {}'.format(module.__name__, module.__version__))
    print('CUDA available: {}; GPU count: {}'.format(torch.cuda.is_available(), torch.cuda.device_count()))
    print('Model registrations and compiled MMCV NMS: OK')


if __name__ == '__main__':
    main()
