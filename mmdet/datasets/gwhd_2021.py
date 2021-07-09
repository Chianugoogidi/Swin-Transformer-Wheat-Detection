from .builder import DATASETS
from .coco import CocoDataset


@DATASETS.register_module()
class WheatDataset(CocoDataset):

    CLASSES = ('wheat_head', )
