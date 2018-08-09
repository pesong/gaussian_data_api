import cv2
import json
import os

import yaml
from pycocotools.coco import COCO
import numpy as np
import pycocotools.mask as maskUtils
import skimage.io as io
import matplotlib.pyplot as plt
import pylab


class GetAnn():
    '''
    get map of coco imgids and annotations which fit our gaussian dataset
    '''

    def __init__(self):
        # our defined detection object
        self.gaussian_things_list = [
            'lane markings', 'person', 'rider', 'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'robot',
            'pole', 'traffic light', 'traffic sign', 'dustbin', 'billboard', 'traffic_cone', 'road_pile',
            'garbage', 'slops', 'ladder', 'pack', 'face', 'chair', 'table', 'carpet', 'potted plant'
            ]

        # our defined stuff object
        self.gaussian_stuff_list = [
            'road', 'pavement', 'sidewalk', 'fence',
            'sky', 'vegetation', 'building', 'wall'
            ]

        # stuff defined in coco which fit our stuff object but not in gaussian_stuff_list above
        self.coco_fitted_stuff_list = [
            'bush', 'grass', 'tree', 'carpet', 'rug', 'sky-other', 'clouds', 'table',
            'wall-stone', 'wall-brick'
            ]

        self.target_stuff_list = self.gaussian_stuff_list + self.coco_fitted_stuff_list

        self.coco_data_dir = '/dl/data/coco'

    def get_gaussian_imgIds(self, data_type):
        '''
        get target img list and annotations from coco things and stuff
        :param data_type: train2017/val2017
        :return: set of coco imgIds
        '''

        img_ids = []

        # get coco things category(subcategory)
        annFile_instance = '{}/annotations/instances_{}.json'.format(self.coco_data_dir, data_type)
        self.thing_coco = COCO(annFile_instance)
        coco_things_cats = self.thing_coco.loadCats(self.thing_coco.getCatIds())
        coco_things_catnms = [cat['name'] for cat in coco_things_cats]

        # find target things in coco, return imgid
        for thing in self.gaussian_things_list:
            if thing in coco_things_catnms:
                # get catid list with given category name
                catIds = self.thing_coco.getCatIds(catNms=[thing])
                img_ids.extend(self.thing_coco.getImgIds(catIds=catIds))


        # get coco stuff category(subcategory)
        annFile_stuff = '{}/annotations/stuff_{}.json'.format(self.coco_data_dir, data_type)
        self.stuff_coco = COCO(annFile_stuff)
        coco_stuff_cats = self.stuff_coco.loadCats(self.stuff_coco.getCatIds())
        coco_stuff_catnms = [cat['name'] for cat in coco_stuff_cats]

        # find target stuff in coco, return imgid
        for stuff in self.target_stuff_list:
            if stuff in coco_stuff_catnms:
                # get catid list with given category name
                catIds = self.stuff_coco.getCatIds(catNms=[stuff])
                img_ids.extend(self.stuff_coco.getImgIds(catIds=catIds))

        # # mini version
        # img_ids = img_ids[0:10]
        # for img_id in img_ids:
        #     print(img_id)

        return set(img_ids)



    def get_img_ann_list(self, img_ids, gs_category_dict=False):
        '''
        get imgs list and annotations from coco label json
        :param img_ids:
        :return: filtered imgs and annotations
        '''

        if gs_category_dict:
            gs_cat_map = gs_category_dict
        else:
            # get gaussian category map {cat_name:id}
            gs_cat_map = {}
            f = open('../../gaussian_categories.yml', 'r')
            catseqs = yaml.load(f)
            for super, seqs in catseqs.items():
                for cat_name, id in seqs.items():
                    gs_cat_map[cat_name] = id

        anns_list = []
        img_list = []
        anns_thing = []
        anns_stuff = []

        for img_id in img_ids:
            img_list.append(self.thing_coco.loadImgs(img_id)[0])
            # get mapped thing annotations
            coco_annIds_thing = self.thing_coco.getAnnIds(img_id)
            coco_anns_thing = self.thing_coco.loadAnns(coco_annIds_thing)

            # get mapped thing annotations
            for ann in coco_anns_thing:
                thing_name = self.thing_coco.loadCats(ann['category_id'])[0]['name']
                if thing_name in self.gaussian_things_list:
                    ann['category_id'] = gs_cat_map[thing_name]

                    anns_thing.append(ann)


            # get mapped stuff annotations
            coco_annIds_stuff = self.stuff_coco.getAnnIds(img_id)
            coco_anns_stuff = self.stuff_coco.loadAnns(coco_annIds_stuff)

            for ann in coco_anns_stuff:
                stuff_name = self.stuff_coco.loadCats(ann['category_id'])[0]['name']
                if stuff_name in self.target_stuff_list:
                    if stuff_name in ['bush', 'grass', 'tree']:
                        ann['category_id'] = gs_cat_map['vegetation']
                    elif stuff_name in ['carpet', 'rug']:
                        ann['category_id'] = gs_cat_map['carpet']
                    elif stuff_name in ['sky-other', 'clouds']:
                        ann['category_id'] = gs_cat_map['sky']
                    elif stuff_name in ['wall-stone', 'wall-brick']:
                        ann['category_id'] = gs_cat_map['wall']
                    else:
                        ann['category_id'] = gs_cat_map[stuff_name]

                    anns_stuff.append(ann)

            anns_list = anns_thing + anns_stuff

        return anns_list, img_list



    def mask2polys(self, anns_list):
        '''
        convert segmentation with mask to ploys
        :param anns_dict: imgs annotations filtered by gaussian target object
        :return: converted annotations
        '''

        for ann in anns_list:
            if "segmentation" in ann and type(ann['segmentation']) != list:
                # mask
                t = self.stuff_coco.imgs[ann['image_id']]
                if type(ann['segmentation']['counts']) == list:
                    rle = maskUtils.frPyObjects([ann['segmentation']], t['height'], t['width'])
                else:
                    rle = [ann['segmentation']]
                mask = maskUtils.decode(rle)

                mask = np.squeeze(mask, 2).astype(np.uint8).copy()

                _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                segmentation = []

                for contour in contours:
                    contour = contour.flatten().tolist()
                    segmentation.append(contour)
                    if len(contour) > 4:
                        segmentation.append(contour)
                if len(segmentation) == 0:
                    continue

                ann['segmentation'] = segmentation

        return anns_list



if __name__ == "__main__":

    coco_ann = GetAnn()
    img_ids_val = coco_ann.get_gaussian_imgIds('val2017')
    anns_val, img_val = coco_ann.get_img_ann_list(img_ids_val)

    refine_anns_val = coco_ann.mask2polys(anns_val)

    print("")

# person bicycle car motorcycle airplane bus train truck boat traffic light fire
# hydrant stop sign parking meter bench bird cat dog horse sheep cow elephant bear
# zebra giraffe backpack umbrella handbag tie suitcase frisbee skis snowboard sports
# ball kite baseball bat baseball glove skateboard surfboard tennis racket bottle
# wine glass cup fork knife spoon bowl banana apple sandwich orange broccoli carrot
#  hot dog pizza donut cake chair couch potted plant bed dining table toilet tv laptop
#  mouse remote keyboard cell phone microwave oven toaster sink refrigerator book
# clock vase scissors teddy bear hair drier toothbrush

# banner blanket branch bridge building-other bush cabinet cage cardboard carpet
#  ceiling-other ceiling-tile cloth clothes clouds counter cupboard curtain
# desk-stuff dirt door-stuff fence floor-marble floor-other floor-stone floor-tile
#  floor-wood flower fog food-other fruit furniture-other grass gravel ground-other
#  hill house leaves light mat metal mirror-stuff moss mountain mud napkin net paper
#  pavement pillow plant-other plastic platform playingfield railing railroad river
#  road rock roof rug salad sand sea shelf sky-other skyscraper snow solid-other stairs
#  stone straw structural-other table tent textile-other towel tree vegetable wall-brick
#  wall-concrete wall-other wall-panel wall-stone wall-tile wall-wood water-other
#  waterdrops window-blind window-other wood other










