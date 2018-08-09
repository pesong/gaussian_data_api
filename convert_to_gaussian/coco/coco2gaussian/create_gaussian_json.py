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
import json
import os
import yaml

from get_coco_annotation import GetAnn

class GaussianJson():
    '''
    convert coco annotations to our Gaussian Json format
    '''

    def __init__(self, target_json_dir):
        self.target_json_dir = target_json_dir


    def generate_gaussian_json(self, data_type, category_yaml):
        '''
        merge json items and generate the label json file according to data_type
        :param data_type: val2017/train2017
        :return: single json file
        '''

        self.__get_basic_info__()
        self.__get_categories__(category_yaml)
        self.__get_img_ann__(data_type)

        json_data = {
            "info": self.info,
            "vehicle_info": self.vehicle_info,
            "log_info": self.log_info,
            "categories": self.categories,
            "images": self.imgs,
            "annotations": self.anns
        }

        with open(os.path.join(self.target_json_dir, "instances_%s.json" % data_type), 'w') as jsonfile:
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

        self.categories = []
        self.category_dict = {}
        self.target_object = []

        f = open(category_yaml, 'r')
        catseqs = yaml.load(f)
        for super, seqs in catseqs.items():
            for name, id in seqs.items():
                self.categories.append({"supercategory": super, "name": name, "id": id})
                self.category_dict[name] = id
                self.target_object.append(name)


    def __get_img_ann__(self, data_type):
        '''
        call get_coco_annotation function and get related imgs and annotations
        :param data_type: train2017/val2017
        :return:
        '''

        # init GetAnn Object and get COCO Annotations
        coco_ann = GetAnn(coco_data_dir)
        img_ids = coco_ann.get_gaussian_imgIds(data_type, self.target_object)
        anns_list, img_list = coco_ann.get_img_ann_list(img_ids,  self.target_object, self.category_dict)

        self.anns = coco_ann.mask2polys(anns_list)

        # get gaussian imgs
        gs_imgs = []

        for img in img_list:
            gs_img = {}

            # construct images
            gs_img['id'] = img['id']
            gs_img['file_name'] = img['file_name']
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

        self.imgs = gs_imgs


if __name__ == "__main__":

    coco_data_dir = '/dl/data/coco'
    target_json_dir = './'
    data_type = 'val2017'
    category_yaml = '../../gaussian_categories_test.yml'

    # generate gaussian json
    gs_json = GaussianJson(target_json_dir)
    gs_json.generate_gaussian_json(data_type=data_type, category_yaml=category_yaml)






