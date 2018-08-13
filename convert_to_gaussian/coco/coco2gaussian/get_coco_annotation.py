'''
This scripts
1. merge the things and stuff task and filter the target imgids

    ref: http://cocodataset.org/#stuff-2018
    things: instance segmentation, provide bbox and polygon/RLE segmentation label
    stuff:  segmentation, provide bbox and RLE segmentation label

we should convert stuff label to things_like json format

2. get the annotations according to imgids

3. map the classification to our gaussian format
    for example, carpet and rug exist in coco can map to carpet

'''

import cv2
import yaml
from pycocotools.coco import COCO
import numpy as np
import pycocotools.mask as maskUtils


class GetCocoAnn():
    '''
    get map of coco imgids and annotations which fit our gaussian dataset
    '''

    def __init__(self, coco_data_dir, target_obj_list):

        # define the dict of the interested items which in coco but not in our yaml
        # such as coco's 'tree'/'bush' not in our categories yaml but belongs to the item 'vegetation'
        self.cats_expand_dict = {
            'vegetation': ['tree', 'grass', 'bush'],
            'sky': ['sky-other', 'clouds'],
            'wall': ['wall-stone', 'wall-brick'],
            'carpet': ['carpet', 'rug'],
        }

        # get coco's object list based on our gaussian yaml
        target_obj_expand_list = []
        for item in target_obj_list:
            if item in self.cats_expand_dict.keys():
                target_obj_expand_list.extend(self.cats_expand_dict[item])
            else:
                target_obj_expand_list.append(item)

        self.target_obj_expand_list = target_obj_expand_list


        self.coco_data_dir = coco_data_dir

    def get_gaussian_imgIds(self, data_type):
        '''
        get target img list from coco things and stuff
        :param data_type: train2017/val2017
        :return: set of coco imgIds
        '''

        img_ids = []

        # get coco things category(subcategory)
        annFile_instance = '{}/annotations/instances_{}.json'.format(self.coco_data_dir, data_type)
        self.thing_coco = COCO(annFile_instance)
        coco_things_cats = self.thing_coco.loadCats(self.thing_coco.getCatIds())
        coco_things_catnms = [cat['name'] for cat in coco_things_cats]

        # get coco stuff category(subcategory)
        annFile_stuff = '{}/annotations/stuff_{}.json'.format(self.coco_data_dir, data_type)
        self.stuff_coco = COCO(annFile_stuff)
        coco_stuff_cats = self.stuff_coco.loadCats(self.stuff_coco.getCatIds())
        coco_stuff_catnms = [cat['name'] for cat in coco_stuff_cats]


        # get catid list with given category name
        for obj in self.target_obj_expand_list:
            if obj in coco_things_catnms:
                catIds = self.thing_coco.getCatIds(catNms=[obj])
                img_ids.extend(self.thing_coco.getImgIds(catIds=catIds))
            elif obj in coco_stuff_catnms:
                catIds = self.stuff_coco.getCatIds(catNms=[obj])
                img_ids.extend(self.stuff_coco.getImgIds(catIds=catIds))


        # mini version
        # img_ids = img_ids[0:100]
        # for img_id in img_ids:
        #     print(img_id)

        return set(img_ids)



    def get_img_ann_list(self, img_ids, gs_category_dict):
        '''
        get imgs list and annotations from coco label json
        :param img_ids: get by function get_gaussian_imgIds()
        :param gs_category_dict: target category dict {category_name: id} read form category yaml
        :param target_obj_list: target object read form category yaml
        :return: filtered imgs and annotations
        '''

        gs_cat_map = gs_category_dict
        anns_list = []
        img_list = []
        anns_thing = []
        anns_stuff = []
        cats_expand_list = []
        cats_expand_dict_verse = {}

        # get cats expand reversed dict: eg. given 'tree', return 'vegetation'
        for key, val_list in self.cats_expand_dict.items():
            cats_expand_list.extend(val_list)
            for val in val_list:
                cats_expand_dict_verse[val] = key

        for img_id in img_ids:

            img_list.append(self.thing_coco.loadImgs(img_id)[0])
            # get mapped thing annotations
            coco_annIds_thing = self.thing_coco.getAnnIds(img_id)
            coco_anns_thing = self.thing_coco.loadAnns(coco_annIds_thing)

            # get mapped thing annotations
            for ann in coco_anns_thing:
                thing_name = self.thing_coco.loadCats(ann['category_id'])[0]['name']
                if thing_name in self.target_obj_expand_list:
                    ann['category_id'] = gs_cat_map[thing_name]

                    anns_thing.append(ann)


            # get mapped stuff annotations
            coco_annIds_stuff = self.stuff_coco.getAnnIds(img_id)
            coco_anns_stuff = self.stuff_coco.loadAnns(coco_annIds_stuff)

            for ann in coco_anns_stuff:
                stuff_name = self.stuff_coco.loadCats(ann['category_id'])[0]['name']

                if stuff_name in self.target_obj_expand_list:

                    if stuff_name in cats_expand_list:
                        ann['category_id'] = gs_cat_map[cats_expand_dict_verse[stuff_name]]
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

        self.ann_id = 0
        gs_anns_list = []

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

                for contour in contours:
                    # contour = contour.flatten().tolist()

                    # select the contour with at least 2 points
                    if len(contour) > 2:
                        gs_ann = {}

                        # discard the contour too small
                        area = cv2.contourArea(contour)
                        if area < 500:
                            continue

                        x, y, w, h = cv2.boundingRect(contour)

                        gs_ann['area'] = area
                        gs_ann['bbox'] = [x, y, w, h]
                        gs_ann['category_id'] = ann['category_id']
                        self.ann_id += 1
                        gs_ann['id'] = self.ann_id
                        gs_ann['image_id'] = ann['image_id']
                        gs_ann['segmentation'] = [contour.flatten().tolist()]
                        gs_ann['iscrowd'] = 0

                        gs_anns_list.append(gs_ann)
            else:
                self.ann_id += 1
                ann['id'] = self.ann_id
                gs_anns_list.append(ann)

        return gs_anns_list


# coco things and stuff

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










