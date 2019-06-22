import json,time
import numpy as np
import os ,cv2
from torch.utils.data import Dataset
# from PIL import Image
from collections import defaultdict
# import Utils

from  visdom import Visdom
class CocoData(Dataset):
    def __init__(self,coco):
        #img的格式是c,w,h
        super(CocoData, self).__init__()
        # self.imgs=coco.imgs
        self.imgs=coco.imgs
        self.gt_boxes=coco.gt_boxes
        self.length=coco.length
    def __getitem__(self,index):

        return self.imgs,self.gt_boxes

    def __len__(self):
        return self.length

class Coco:
    def __init__(self,annotation_file_path,image_path):
        self.imgs_path=image_path #图片的根目录
        self.dataset_dict=None #数据集dict
        self.info={} #信息
        self.images={} #图片信息
        self.licenses={} #凭证
        self.categories={} #所有的类别
        self.annotations={} #所有的标注
        self.img2anns=defaultdict(list) #图片的标注
        self.cat2imgs=defaultdict(list) #类别的图片
        self.annotations_file=annotation_file_path #标记文件路径
        self.imgs=[]#图片内容
        self.gt_boxes=[]#真实框 list of list
        self.length=0
        # self.imgs_gt_boxes=None
        self.load_jsons() #读取json文件
        self.create_index() #分析json文件

    def get_gt_bboxes(self,img_id):
        #获取一个图片中的所有的真实框们,和对应的类别

        anns=self.get_anns(img_id)
        bboxes=defaultdict(list)
        for ann in anns:
            bbox=ann["bbox"]
            b_class=ann['category_id']
            bboxes[b_class].append(bbox)
        return bboxes
    def load_jsons(self):
        #加载json文件,显示加载时间
        start=time.time()
        print("loading annotation file into memory!")
        self.dataset_dict=json.load(open(self.annotations_file, 'r'))
        interval=time.time()-start
        print("load success,time-consuming %d"%interval)

    def create_index(self):

        self.info=self.dataset_dict["info"]

        for ann in self.dataset_dict["annotations"]:
            self.annotations[ann["id"]]=ann
            self.img2anns[ann["image_id"]].append(ann)

        for image in self.dataset_dict["images"]:
            self.images[image["id"]]=image

        for lic in self.dataset_dict["licenses"]:
            self.licenses[lic["id"]]=lic

        for cat in self.dataset_dict["categories"]:
            self.categories[cat["id"]]=cat
            self.cat2imgs[ann["category_id"]].append(ann["image_id"])

        # self.imgs_gt_boxes = self.get_all_img_and_box()
        self.get_all_img_and_box()
        print("index created!")
    def get_img_size(self,img_id):
        #获取某个img的width 和 height
        image=self.images[img_id]

        width=image["width"]
        height=image["height"]
        return width,height
    def get_anns(self,img_id):
        # 获取某个图片的标注们

        return self.img2anns[img_id]
    def get_all_img_ids(self):
        #获取json文件中所有的图片的id
        ids=[]
        for image in self.images:
            ids.append(image)
        return ids
    def get_img_by_id(self,img_id):
        #根据id获取图片的np array形式，RGB格式
        img_dict=self.images[img_id]
        file_name=os.path.join(self.imgs_path,img_dict["file_name"])
        image=cv2.imread( file_name)
        return image[...,::-1].copy()
    def get_all_img_and_box(self):

        img_ids=self.get_all_img_ids()
        self.length=len(img_ids)
        count=0
        for img_id in img_ids:
            count+=1
            if count%100==0:
                print(count)

            # yield self.get_img_by_id(img_id),self.get_gt_bboxes(img_id)
            # yield

            self.imgs.append(self.get_img_by_id(img_id))
            self.gt_boxes.append(self.get_gt_bboxes(img_id))


    def show_gt_bbox(self,img_id,with_img):
        #显示bbox,是否显示原始图片
        if not with_img:

            width,height=self.get_img_size(img_id)
            image=np.zeros((height,width,3),np.uint8)
        else:
            img_path=os.path.join(self.imgs_path,self.images[img_id]['file_name'])
            image=cv2.imread(img_path)
        anns=self.get_anns(img_id)

        for ann in anns:
            bbox=ann["bbox"]
            pt1=(int(bbox[0]),int(bbox[1]))
            pt2=(int(bbox[2]+bbox[0]),int(bbox[3]+bbox[1]))
            cv2.rectangle(image,pt1,pt2,(255,0,0))
            cv2.putText(image,self.categories[ann["category_id"]]["name"],pt1,1,1,(0,0,255))
        vis=Visdom(env="img")
        vis.image(image.transpose(2, 0, 1)[::-1,...])

if __name__ == '__main__':
    print("????")
    # ann_path="i:/cocodata/MS_COCO/annotations_trainval2017/annotations/instances_val2017.json"
    # imgs_path="I:/cocodata/MS_COCO/val2017"
    # coco=Coco(ann_path,imgs_path)
    # id_list=coco.get_all_img_ids()
    #
    # width_list=[]
    # height_list=[]
    # for img_id in id_list:
    #     boxes=coco.get_gt_bboxes(img_id)
    #     for box in boxes.values():
    #         width_list.append(box[0][2])
    #         height_list.append(box[0][3])

# print("1")

    # img=coco.get_img_by_id(id_list[0])
    # img=Utils.process_img(img)





