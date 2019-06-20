import torchvision,torch,torch.nn as nn
import Utils,coco_utils
from torch.utils.data import DataLoader

class  RPN(nn.Module):

    def __init__(self):
        super(RPN, self).__init__()
        vgg = torchvision.models.vgg16(pretrained=True)
        self.features=vgg.features[:30]
        self.conv3by3=nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
        )

        self.bin_class_conv=nn.Sequential(
            nn.Conv2d(512, 18, 1, 1),
            nn.Softmax(dim=1))
        self.reg_conv=nn.Conv2d(512,36,1,1)

        self.roi_pooling=nn.AdaptiveAvgPool2d((7,7))
    def forward(self, img):#,gt_anchors):#img是tensor格式的，c,w,h

        features=self.features(img)
        features=self.conv3by3(features)
        f_size=features.size()

        f_width = f_size[3]
        f_height = f_size[2]
        cls=self.bin_class_conv(features)
        reg=self.reg_conv(features)

        return  f_width,f_height,cls,reg


        # f_size = features.size()
        #
        # f_width = f_size[3]
        # f_height = f_size[2]
        #
        #
        # anchors = Utils.get_anchors(f_width, f_height, 16)
        # gt_anchors, max_iou_anchors, pos_anchors, ign_anchors, neg_anchors = Utils.filter_anchors(anchors, gt_anchors,0.7, 0.3)
        # pos_anchors_data = Utils.AnchorData(img, pos_anchors.extend(max_iou_anchors)[:256], 1)
        # pos_len=len(pos_anchors_data)
        #
        #
        # neg_anchors_data=Utils.AnchorData(img,neg_anchors[:pos_len],2)
        #
        # pos_loader=DataLoader(pos_anchors_data,shuffle=True,batch_size=2)
        # neg_loader=DataLoader(neg_anchors_data,shuffle=True,batch_size=2)
        # for idx,(pos,neg) in zip(pos_loader,neg_loader):
        #     pos=self.roi_pooling(pos)
        #     neg=self.roi_pooling(neg)

        return features
if __name__ == '__main__':
    print( torchvision.models.vgg16())