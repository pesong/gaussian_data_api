import collections
import json
import os
import shutil
import numpy as np
import cv2
import yaml

import convert_to_gaussian.cityscapes.cityscapesscripts.evaluation.instances2dict_with_polygons as cs
import convert_to_gaussian.cityscapes.utils.boxes as bboxs_util
import convert_to_gaussian.cityscapes.utils.segms as segms_util


class GSJsonFromCityscapes():
    '''
    convert cityscapes to gaussian json format
    '''

    def __init__(self, city_data_dir, out_dir):
        self.img_id = 1
        self.ann_id = 1
        self.city_data_dir = city_data_dir
        self.out_dir = out_dir
        self.gaussian_stuff_list = \
            ['road', 'pavement', 'fence', 'sky', 'vegetation', 'building', 'wall', 'sidewalk', 'terrain']


    def generate_gaussian_json(self, data_type, category_yaml, iscopy = False):
        self.__get_basic_info__()
        self.__get_categories__(category_yaml)
        self.__get_img_ann__(data_type, iscopy)

        json_data = {
            "info": self.info,
            "vehicle_info": self.vehicle_info,
            "log_info": self.log_info,
            "categories": self.categories,
            "images": self.images,
            "annotations": self.annotations
        }

        # write json to out_dir
        if not os.path.exists(os.path.join(self.out_dir, 'annotations')):
            os.makedirs(os.path.join(self.out_dir, 'annotations'))

        with open(os.path.join(self.out_dir, 'annotations', "instances_%s2017.json" % data_type), 'w') as jsonfile:
            jsonfile.write(json.dumps(json_data, sort_keys=True))

    def __get_basic_info__(self, ros_info=False):
        """
        # get basic info such as description and vehicle/log infos .etc
        :param self:
        :param ros_info:
        :return:
        """

        self.info = {
            "year": 2018,
            "version": 'test_api_v1',
            "description": 'convert  cityscapes to gaussian dataset merging the thing & stuff tasks',
            "contributor": 'gaussian dl team',
            "device": 'cityscapes images',
            "date_created": '2018-09-26'
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

    def __get_img_ann__(self, data_type, iscopy=False):
        """
        convert from cityscapes format to gaussian format
        :param data_type: val/train/test
        :return:
        """

        sets = ['gtFine_val']
        ann_dirs = ['gtFine_trainvaltest/gtFine/%s' % data_type]           # data_type: val/train/test
        json_name = ''
        ends_in = '%s_polygons.json'

        category_list = self.target_obj_list


        # define copy file path

        src_dir = os.path.join(self.city_data_dir, 'leftImg8bit', data_type)
        target_dir = os.path.join(self.out_dir, 'images', data_type+'2017')
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)


        for data_set, ann_dir in zip(sets, ann_dirs):
            print('Starting %s' % data_set)
            ann_dict = {}
            self.images = []
            self.annotations = []
            ann_dir = os.path.join(self.city_data_dir, ann_dir)
            for root, _, files in os.walk(ann_dir):
                for filename in files:
                    if filename.endswith(ends_in % data_set.split('_')[0]):
                        if len(self.images) % 50 == 0:
                            print("Processed %s images, %s annotations" % (
                                len(self.images), len(self.annotations)))
                        json_ann = json.load(open(os.path.join(root, filename)))


                        seg_file_name = filename[:-len(
                            ends_in % data_set.split('_')[0])] + \
                                                 '%s_labelIds.png' % data_set.split('_')[0]

                        fullname = os.path.join(root, seg_file_name)
                        objects = cs.instances2dict_with_polygons(
                            [fullname], verbose=False)[fullname]

                        for object_cls in objects:
                            if object_cls not in category_list:
                                continue  # skip non-instance categories

                            for obj in objects[object_cls]:
                                if obj['contours'] == []:
                                    print('Warning: empty contours.')
                                    continue  # skip non-instance categories

                                len_p = [len(p) for p in obj['contours']]
                                if min(len_p) <= 4:
                                    print('Warning: invalid contours.')
                                    continue  # skip non-instance categories

                                for seg in obj['contours']:
                                    ann = {}
                                    ann['id'] = self.ann_id
                                    self.ann_id += 1

                                    ann['image_id'] = self.img_id

                                    # map cityscapes categoty to gaussian category_dict
                                    if object_cls == 'sidewalk':
                                        ann['category_id'] = self.category_dict['pavement']
                                    elif object_cls == 'terrain':
                                        ann['category_id'] = self.category_dict['vegetation']
                                    else:
                                        ann['category_id'] = self.category_dict[object_cls]

                                    ann['iscrowd'] = 0
                                    seg_reshape = np.array(seg).reshape((len(seg)//2), 2)
                                    aera = cv2.contourArea(seg_reshape)
                                    # remove small object in stuff list
                                    if object_cls in self.gaussian_stuff_list and aera < 3000:
                                        continue
                                    ann['area'] = aera
                                    ann['bbox'] = bboxs_util.xyxy_to_xywh(
                                        segms_util.polys_to_boxes(
                                            [[seg]])).tolist()[0]
                                    ann['segmentation'] = [seg]
                                    self.annotations.append(ann)

                        #  add image info to our annotations
                        image = {}
                        image['id'] = self.img_id
                        self.img_id += 1
                        image['width'] = json_ann['imgWidth']
                        image['height'] = json_ann['imgHeight']
                        image['depth'] = 3
                        image['device'] = 'camera'
                        image['date_captured'] = ''
                        image['rosbag_name'] = ''
                        image['encode_type'] = 'rgb'
                        image['is_synthetic'] = 'no'
                        image['vehicle_info_id'] = '0'
                        image['log_info_id'] = '0'
                        image['weather'] = ''
                        file_name = filename[:-len(
                            ends_in % data_set.split('_')[0])] + 'leftImg8bit.png'
                        image['file_name'] = file_name
                        self.images.append(image)

                        # copy target image file to outdir
                        if iscopy:
                            print('copying files form source dataset')
                            src_dir_suffix = root.split('/')[-1]
                            shutil.copyfile(os.path.join(src_dir, src_dir_suffix, file_name),
                                            os.path.join(target_dir, file_name))

                        # # control dataset size
                        # if len(self.images) >= 50:
                        #     break


if __name__ == '__main__':
    data_dir = '/media/pesong/e/dl_gaussian/data/cityscapes/cityscapes_mini'
    # out_dir = '/media/pesong/e/dl_gaussian/data/cityscapes/4detectron/annotations'

    out_dir = '../../test_cs'
    category_yaml = '../../gaussian_categories_test.yml'

    # init
    gs_json_from_city = GSJsonFromCityscapes(data_dir, out_dir)

    # generate val json
    gs_json_from_city.generate_gaussian_json('val', category_yaml, iscopy=True)

    # # generate train json
    # gs_json_from_city.generate_gaussian_json('train', category_yaml)
