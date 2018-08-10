'''

This scripts aimed to merge the gaussian formatted images and labels converted from
the other dataset such as coco/cityscapes .etc.

We intend to:
 1. unify the imgid, file_name with different sources
 2. merge the label json file
 3. copy the sourced images to target dir

'''
import json
import os
import random

from convert_to_gaussian.coco.coco2gaussian.create_gaussian_json import GaussianJsonCoco
from convert_to_gaussian.cityscapes.cityscapes2gaussian.cityscapes_to_gaussian import GSJsonFromCityscapes


class GaussianData():

    def __init__(self, out_dir):
        self.out_root_dir = out_dir

    def convert_source(self, data_source, source_data_dir, data_type, category_yaml):
        '''
        convert the source data by given data and data_dir
        :param data_source: data source name
        :param source_data_dir: the path of the data
        :param data_type: val/train/test
        :param category_yaml: the path of gaussian categories file
        :return:
        '''

        # make dir of saved path
        outdir = os.path.join(self.out_root_dir, data_source)
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        # generate json file
        if data_source == 'coco':
            data_type = data_type + '2017'
            gs_json_from_coco = GaussianJsonCoco(outdir)
            print('---------------start convert coco---------------------')
            gs_json_from_coco.generate_gaussian_json(data_type, category_yaml, source_data_dir)

        elif data_source == 'cityscapes':
            gs_json_from_city = GSJsonFromCityscapes(source_data_dir, outdir)
            print('---------------start convert cityscapes----------------')
            gs_json_from_city.generate_gaussian_json(data_type, category_yaml)


    def merge_json(self, data_type, json_file1, jsonfile2):
        '''
        merge the label json files and shuffle the images and annotations
        :param data_type: val/gaussian
        :param json_file1:
        :param jsonfile2:
        :return:
        '''
        self.info = {
            "year": 2018,
            "version": 'test_api_v2',
            "description": 'convert  coco2017 && cityscapes to gaussian dataset',
            "contributor": 'gaussian dl team',
            "device": 'coco & cityscapes images',
            "date_created": '2018-07-28'
        }

        with open(json_file1) as f:
            line = f.readline()
            d = json.loads(line)
            self.imgs = d['images']
            self.anns = d['annotations']
            self.vehicle_info = d['vehicle_info']
            self.log_info = d['log_info']
            self.categories = d['categories']

        with open(jsonfile2) as f:
            line = f.readline()
            d = json.loads(line)
            print('-----------extend list-------------')
            self.imgs.extend(d['images'])
            self.anns.extend(d['annotations'])

        print('-----------shuffle-------------')
        random.shuffle(self.imgs)
        random.shuffle(self.anns)

        json_data = {
            "info": self.info,
            "vehicle_info": self.vehicle_info,
            "log_info": self.log_info,
            "categories": self.categories,
            "images": self.imgs,
            "annotations": self.anns
        }

        if not os.path.exists(os.path.join(self.out_root_dir, 'annotations')):
            os.makedirs(os.path.join(self.out_root_dir, 'annotations'))
        with open(os.path.join(self.out_root_dir, 'annotations', "instances_%s2017.json" % data_type), 'w') as jsonfile:
            jsonfile.write(json.dumps(json_data))


def convert():
    # define source data list
    source_data_dict = [
        {
            'data_source': 'coco',
            'source_data_dir': '/media/pesong/e/dl_gaussian/data/coco'
        },
        {
            'data_source': 'cityscapes',
            'source_data_dir': '/media/pesong/e/dl_gaussian/data/cityscapes/cityscapes_ori'
        }
    ]

    # define paremeters
    out_dir = '/media/pesong/e/dl_gaussian/data/gaussian'
    data_types = ['train', 'val']
    category_yaml = './gaussian_categories_test.yml'

    # init Object
    gs_data = GaussianData(out_dir)

    # generate json
    for data_type in data_types:
        for source in source_data_dict:
            data_source = source['data_source']
            source_data_dir = source['source_data_dir']
            gs_data.convert_source(data_source, source_data_dir, data_type, category_yaml)


def merge():
    out_dir = '/media/pesong/e/dl_gaussian/data/gaussian'
    data_types = ['val', 'train']

    gs_data = GaussianData(out_dir)
    for data_type in data_types:
        json_file1 = '/media/pesong/e/dl_gaussian/data/gaussian/coco/annotations/instances_{}2017.json'.format(data_type)
        json_file2 = '/media/pesong/e/dl_gaussian/data/gaussian/cityscapes/annotations/instances_{}2017.json'.format(data_type)
        gs_data.merge_json(data_type, json_file1, json_file2)



if __name__ == "__main__":
    # convert()
    merge()



