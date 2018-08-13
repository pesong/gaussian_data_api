"""
This scripts generate our gaussian lable json file
ref: https://gaussian.yuque.com/perception/documents/pktva3

{
"info":             info,
"images":           {image}, {image}, {image}
"annotations":      {annotation}, {annotation}, {annotation}
"categories":       {category}, {category}
}


"""
import collections
import json
import os
import random
import shutil

import yaml
from convert_to_gaussian.coco.coco2gaussian.get_coco_annotation import GetCocoAnn

class GaussianJsonCoco():
    '''
    convert coco annotations to our Gaussian Json format
    '''

    def __init__(self, out_dir):
        self.out_dir = out_dir

    def generate_gaussian_json(self, data_type, category_yaml, coco_data_dir, iscopy=False):
        '''
        merge json items and generate the label json file according to data_type
        :param data_type: val2017/train2017
        :param category_yaml: path of gaussian category yaml file
        :param coco_data_dir: absolute path of coco dataset
        :return: single json file
        '''

        self.__get_basic_info__()
        self.__get_categories__(category_yaml)
        self.__get_img_ann__(data_type, coco_data_dir, iscopy)

        json_data = {
            "info": self.info,
            "vehicle_info": self.vehicle_info,
            "log_info": self.log_info,
            "categories": self.categories,
            "images": self.imgs,
            "annotations": self.anns
        }

        if not os.path.exists(os.path.join(self.out_dir, 'annotations')):
            os.makedirs(os.path.join(self.out_dir, 'annotations'))
        with open(os.path.join(self.out_dir, 'annotations', "instances_%s.json" % data_type), 'w') as jsonfile:
            jsonfile.write(json.dumps(json_data, sort_keys=True))


    def __get_basic_info__(self, ros_info=False):
        '''
        # get basic info such as description and vehicle/log info .etc
        :param ros_info: receive the external info such as rosbag
        '''

        self.info = {
            "year": 2018,
            "version": 'test_api_v1',
            "description": 'convert  coco2017 to gaussian dataset merging the thing & stuff tasks',
            "contributor": 'gaussian dl team',
            "device": 'coco images',
            "date_created": '2018-07-28'
        }

        self.vehicle_info = [{
            "id": '0',
            "hardware_version": '',
            "software_version": '',
            "sensor_list": [],
            "sensor_frequency": 0
        }]

        self.log_info = [{
            "id": '0',
            "type": '',
            "vehicle_id": 0,
            "location": '',
            "starting time": '',
            "end time": ''
        }]


    def __get_categories__(self, category_yaml):
        '''
        read target category yaml file, and get our target categories
        :return: dict and list of categories
        '''


        id_cat = {}             # {id:{}}
        self.categories = []
        self.category_dict = {}
        self.target_obj_list = []
        id_list = []

        f = open(category_yaml, 'r')
        catseqs = yaml.load(f)
        for super, seqs in catseqs.items():
            for name, id in seqs.items():
                id_cat[id] = {"supercategory": super, "name": name, "id": id}
                self.category_dict[name] = id
                self.target_obj_list.append(name)
                id_list.append(id)

        id_list.sort()

        for id in id_list:
            self.categories.append(id_cat[id])



    def __get_img_ann__(self, data_type, coco_data_dir, iscopy=False):
        '''
        call get_coco_annotation function and get related imgs and annotations
        :param data_type: train2017/val2017
        :param coco_data_dir: absolute path of coco dataset
        :return:
        '''

        # init GetCocoAnn Object and get COCO Annotations
        coco_ann = GetCocoAnn(coco_data_dir, self.target_obj_list)
        img_ids = coco_ann.get_gaussian_imgIds(data_type)
        anns_list, img_list = coco_ann.get_img_ann_list(img_ids, self.category_dict)

        self.anns = coco_ann.mask2polys(anns_list)

        # get gaussian imgs
        gs_imgs = []

        # define copy file path
        src_dir = os.path.join(coco_data_dir, 'images', data_type)
        target_dir = os.path.join(self.out_dir, 'images', data_type)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        for img in img_list:
            gs_img = {}

            # construct images
            gs_img['id'] = img['id']

            file_name = img['file_name']
            gs_img['file_name'] = file_name
            gs_img['coco_url'] = img['coco_url']

            gs_img['width'] = img['width']
            gs_img['height'] = img['height']
            gs_img['depth'] = 3
            gs_img['device'] = 'camera'
            gs_img['date_captured'] = img['date_captured']
            gs_img['rosbag_name'] = ''
            gs_img['encode_type'] = 'rgb'
            gs_img['is_synthetic'] = 'no'
            gs_img['vehicle_info_id'] = '0'
            gs_img['log_info_id'] = '0'
            gs_img['weather'] = ''
            gs_imgs.append(gs_img)


            # copy target image file to outdir
            if iscopy:
                # print('copying files from source dataset')
                shutil.copyfile(os.path.join(src_dir, file_name), os.path.join(target_dir, file_name))


        self.imgs = gs_imgs


if __name__ == "__main__":

    coco_data_dir = '/dl/data/coco'
    out_dir = '/media/pesong/e/dl_gaussian/data/gaussian'
    data_type = 'val2017'
    category_yaml = '../../gaussian_categories_test.yml'

    # generate gaussian json
    gs_json = GaussianJsonCoco(out_dir)
    gs_json.generate_gaussian_json(data_type, category_yaml, coco_data_dir)






