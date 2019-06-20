import torch
import torch.nn as nn
from vgg16model import RPN
import Utils
from torch.utils.data import DataLoader
from coco_utils import Coco,CocoData

import random
if __name__ == '__main__':
    CUDA_INDEX=1
    evaluation="val"
    ann_path = "e:/python/coco_dataset/annotations_trainval2017/annotations/instances_%s2017.json"%evaluation
    imgs_path = "e:/python/coco_dataset/%s2017"%evaluation
    batch_size=1
    coco = Coco(ann_path, imgs_path)
    coco_data=CocoData(coco)
    coco_loader=DataLoader(coco_data,batch_size,True)
    rpn_backbone = RPN().cuda(CUDA_INDEX)

    optimizer = torch.optim.Adam(rpn_backbone.parameters())
    cls_loss_func = torch.nn.MSELoss()
    reg_loss_func = nn.MSELoss()
    EPOCHES=50
    for eps in range(EPOCHES):
        for raw_img,gt_bboxes in coco_data:
            img = Utils.process_img(raw_img) # 减去均值操作 # 427,640,3 宽 长 通道
            img_width=img.shape[1]
            img_height=img.shape[0]
            tensor=torch.Tensor(img).cuda(CUDA_INDEX).unsqueeze(0).permute(0,3,1,2) # 1,3,427,640

            f_width,f_height,cls,reg=rpn_backbone(tensor) # 40,26, (1,18,26,40),(1,36,26,40)
            label_cls,label_reg=cls.clone(),reg.clone() # 先用回归值填充真值，再改变其中一些值(1,18,26,40),(1,36,26,40)

            anchors_dict=Utils.get_anchors_dict(f_width,f_height,16)#得到所有的anchor (26,40,9,4),numpy 格式
            gt_anchors, pos_diff_dict,pos_dict, ign_dict, neg_dict = Utils.filter_anchors(anchors_dict, gt_bboxes, 0.7, 0.3)

            # Utils.anchors_visualization(img_width,img_height,anchors_dict,show_center=True)
            # Utils.show_img_anchors(raw_img, gt_anchors, (0, 255, 0))
            # Utils.show_img_anchors(raw_img, pos_dict, (0, 255, 0))
            # Utils.show_img_anchors(raw_img, ign_dict, (0, 255, 0))
            # Utils.show_img_anchors(raw_img, neg_dict, (0, 255, 0))

            len_pos=len(pos_dict)
            print(len_pos)
            neg_potisions=list(ign_dict.keys())
            random.shuffle(neg_potisions)

            c,numbers,height,width=label_cls.size()
            label_cls=label_cls.view(c,9,2,height,width)
            label_reg=label_reg.view(c,9,4,height,width)
            for h,w,num in pos_dict:
                label_cls[0,num,:,h,w]=torch.tensor([1,0])
                label_reg[0,num,:,h,w]=torch.Tensor(pos_diff_dict[h,w,num]) #anchor-ground truth

            for h,w,num in neg_potisions[:len_pos]:
                label_cls[0,num,:,h,w]=torch.Tensor([0,1])


            label_cls=label_cls.view(1,18,f_height,f_width)
            label_reg=label_reg.view(1,36,f_height,f_width)

            cls_loss=cls_loss_func(cls,label_cls)
            reg_loss=reg_loss_func(reg,label_reg)
            lamd=1
            total_loss=cls_loss+lamd*reg_loss
            print(cls_loss.item(),reg_loss.item())

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        torch.save(rpn_backbone,"./rpn_pkl/%drpn.pkl"%eps)

