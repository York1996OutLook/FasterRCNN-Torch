import cv2,numpy as np,torch
from torch.utils.data import Dataset
from visdom import Visdom
from collections import defaultdict
import random
# import torch
def deformation(top_left,down_sample_ratio,ratio,scale):
    '''
    把anchor根据ratio和scale变换
    :param top_left: x,y,w,h
    :param down_sample_ratio 降采样的倍数
    :param ratio: 长比宽
    :param scale: 缩放比例
    :return: 变换后的anchor
    '''
    x,y=top_left
    w=down_sample_ratio
    h=down_sample_ratio
    center_x=x*down_sample_ratio+down_sample_ratio/2
    center_y=y*down_sample_ratio+down_sample_ratio/2

    return [center_x-w*ratio**0.5/2*scale,center_y-h/ratio**0.5/2*scale,w*ratio**0.5*scale,h/ratio**0.5*scale]


def get_rect(anchor):
    '''
    把anchor变成rect
    :param anchor: x,y,w,h
    :return: x,y,x+w,x+h
    '''
    x,y,w,h=anchor
    return int(x),int(y),int(x+w),int(y+h)
def get_center(anchor):
    '''
    找到anchor的中心位置
    :param anchor:
    :return:
    '''
    x,y,w,h=anchor
    center=int(x+w/2),int(y+h/2)
    return center
def anchors_visualization(width,height,anchors_dict,show_center):
    '''
    可视化anchor们
    :param width: 图片宽
    :param height: 图片长
    :param anchors_dict: anchor们
    :param show_center :是显示anchor的中心还是显示anchor
    :return:
    '''
    image=np.zeros((height,width,3),np.uint8)
    last_h, last_w=-1,-1
    for h,w,num in anchors_dict:
        anchor=anchors_dict[h,w,num]
        if show_center:
            if (h,w) == (last_h,last_w):
                continue#num 的值影响不了画图的中心位置
            center = get_center(anchor)
            image=cv2.rectangle(image,(center[0]-1,center[1]-1),(center[0]+1,center[1]+1),[0,255,0],1)
            #这里有冗余
            last_h,last_w=h,w
        else:
            rect = get_rect(anchor)
            image=cv2.rectangle(image,(rect[0],rect[1]),(rect[2],rect[3]),[255,0,0],1)
    vis = Visdom(env="img")
    vis.image(image.transpose(2,0,1)[::-1,...])

def process_img(raw_img):

    _R_MEAN = 123.68
    _G_MEAN = 116.78
    _B_MEAN = 103.94
    mean =[_R_MEAN, _G_MEAN, _B_MEAN]
    #转换成bgr
    return raw_img.astype(np.float)-mean
def get_anchors_dict(width,height,down_sample_ratio,ratios=[0.5,1,2],scales=[4,9,16]):
    '''
    获取所有的anchor
    :param width: feature map的宽
    :param height: feature map的高度
    :param down_sample_ratio 降采样的倍数
    :param ratios: 长宽比
    :param scales: 缩放比例
    :return:
    '''

    results_anchors_dict={}
    for h in  range(height):
        for w in range(width):
            for s_idx,scale in enumerate(scales):
                for r_idx,ratio in enumerate(ratios):
                    results_anchors_dict[h,w,s_idx*3+r_idx]=deformation([w,h] ,down_sample_ratio, ratio, scale)
    return results_anchors_dict

def filter_anchors(anchors_dict,gt_bboxes,upper_iou,lower_iou):
    '''

    :param anchors_dict: 所有的anchors
    :param gt_bboxes: 所有的真实框
    :param upper_iou: 大于upper iou的是真实框
    :param lower_iou: 在upper和lower之间的忽略样本，小于lower的是负样本
    :return: 返回值是索引，【h,w,num】这些位置是正样本框或者负样本框。
    '''

    # pos_anchors=[]     #h,w,num:anchor-gt_box

    ign_dict={}  #dict of [h,w,num]:anchor x,y,w,h
    neg_dict={}  #dict of [h,w,num]:anchor x,y,w,h
    pos_dict={}  #dict of [h,w,num]:anchor x,y,w,h

    pos_diff_dict={}


    # max_gt_iou_anchors=[]#list of [h,w,num]
    max_gt_iou_positions_dict={}#idx:[h,w,num]
    max_gt_iou_diff_dict={}#h,w,num:anchor-gt_box
    max_gt_iou_dict={} #h,w,num:anchor

    all_gt_bboxes_dict = defaultdict(list)
    idx=0
    for key in gt_bboxes:  # 11
        bboxes = gt_bboxes[key]
        for box in bboxes:

            all_gt_bboxes_dict[idx, idx, idx]=box  # 19
            idx+=1

    #找出每个gt最大的iou的样本
    for _,_,idx in all_gt_bboxes_dict:
        gt_box=all_gt_bboxes_dict[idx,idx,idx]
        max_iou = 0

        for h ,w,num in anchors_dict:

            anchor=anchors_dict[h,w,num]
            iou=cal_iou(anchor,gt_box)#x,y,w,h格式

            if max_iou<iou:
                # print(iou)
                max_iou=iou
                max_gt_iou_positions_dict[idx]=[h,w,num] # 每个gt都有一个最大iou的正样本
    #找出每个iou最大的框的位置和与真值框之间的差距
    for idx in max_gt_iou_positions_dict:

        h,w,num=max_gt_iou_positions_dict[idx]#最接近的anchor的索引，格式是h,w,num

        gt_box=all_gt_bboxes_dict[idx,idx,idx]
        gt_x,gt_y,gt_w,gt_h=gt_box
        anchor=anchors_dict[h,w,num]
        an_x,an_y,an_w,an_h=anchor

        max_gt_iou_diff_dict[h,w,num]=[(gt_x-an_x)/an_w,(gt_y-an_y)/an_h,np.log(gt_w/an_w),np.log(gt_h/an_h)]


        max_gt_iou_dict[h,w,num]=anchors_dict[h,w,num],gt_box
    #找出每个iou最大的框的位置和与真值框之间的差距

    for h, w, num in anchors_dict:

        anchor = anchors_dict[h, w, num]
        if [h,w,num] in max_gt_iou_positions_dict.values():
            continue
        max_iou = -1
        max_idx = -1
        for _,_,idx in all_gt_bboxes_dict:

            gt_box=all_gt_bboxes_dict[idx, idx, idx]
            iou = cal_iou(anchor, gt_box)

            if iou>max_iou:
                max_iou=iou
                max_idx=idx

        if max_iou>upper_iou:
            gt_box=all_gt_bboxes_dict[max_idx,max_idx,max_idx]

            gt_x, gt_y, gt_w, gt_h = gt_box
            an_x, an_y, an_w, an_h = anchor

            pos_diff_dict[h, w, num] = [(gt_x - an_x) / an_w, (gt_y - an_y) / an_h, np.log(gt_w / an_w),
                                               np.log(gt_h / an_h)]

            #正样本需要记录下来对应的anchor-ground truth
            pos_dict[h,w,num]=anchor,gt_box
            #anchor和gt的差距用来构造target
        elif max_iou>lower_iou:#负样本和忽略样本不需要进行位置的回归
            ign_dict[h,w,num]=anchor

        else:
            neg_dict[h,w,num]=anchor

    # pos_anchors.extend(max_gt_iou_anchor.values())
    pos_diff_dict.update(max_gt_iou_diff_dict)
    pos_dict.update(max_gt_iou_dict)
    return all_gt_bboxes_dict,pos_diff_dict,pos_dict,ign_dict,neg_dict

def show_img_anchors(image,anchors_dict,show_gt=False):
    img_cp=image.copy()

    for h,w,num in anchors_dict:
        color = [random.randint(0, 255)for _ in range(3)]
        if show_gt:
            anchor,gt_anchor = anchors_dict[h, w, num]
            x, y, w, h = gt_anchor
            pt1 = (int(x), int(y))
            pt2 = (int(x + w), int(y + h))
            cv2.rectangle(img_cp, pt1, pt2, color)
        else:
            anchor = anchors_dict[h, w, num]

        x, y, w, h = anchor
        pt1 = (int(x), int(y))
        pt2 = (int(x + w), int(y + h))
        cv2.rectangle(img_cp, pt1, pt2, color)


    vis = Visdom(env="img")
    vis.image(img_cp.transpose(2,0,1))

def cal_iou(rect1,rect2):
    '''
    计算两个矩形的交并比
    :param rect1:第一个矩形框。表示为x,y,w,h，其中x,y表示矩形右上角的坐标
    :param rect2:第二个矩形框。
    :return:返回交并比，也就是交集比并集
    '''

    x1,y1,w1,h1=rect1
    x2,y2,w2,h2=rect2

    inter_w=(w1+w2)-(max(x1+w1,x2+w2)-min(x1,x2))
    inter_h=(h1+h2)-(max(y1+h1,y2+h2)-min(y1,y2))

    if inter_h<=0 or inter_w<=0:#代表相交区域面积为0
        return 0
    #往下进行应该inter 和 union都是正值
    inter=inter_w * inter_h

    union=w1*h1+w2*h2-inter
    return  inter/union

if __name__ == '__main__':
    # my_anchors=get_anchors(60,40)
    # anchors_visualization(1000,600,my_anchors)
    pass
# near=max_gt_iou_anchor[18]
# gt=all_gt_bboxes[18]
# print(near,gt)
# image = np.zeros((422, 640), np.uint8)
# image = cv2.rectangle(image, (int(gt[0]), int(gt[1])), (int(gt[2])+int(gt[0]), int(gt[3])+int(gt[1])), 255, 1)
# image = cv2.rectangle(image, (int(near[0]), int(near[1])), (int(near[2])+int(near[0]), int(near[3])+int(near[1])), 255, 1)
# vis=Visdom(env="img")
# vis.image(image)