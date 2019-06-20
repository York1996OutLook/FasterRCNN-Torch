import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
import pylab

coco=COCO("/public/home/G19940018/3DGroup/Yaochun/FasterRCNN/coco_dataset/stuff_annotations_trainval2017/annotations/stuff_val2017.json")
catIds=coco.getCatIds(catNms=['dog','person'])
print(catIds)