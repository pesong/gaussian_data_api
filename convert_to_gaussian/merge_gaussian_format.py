'''

This scripts aimed to merge the gaussian formatted images and labels converted from
the other dataset such as coco/cityscapes .etc.

We intend to:
 1. unify the imgid, file_name with different sources
 2. merge the label json file
 3. copy the sourced images to target dir

'''
import os

from convert_to_gaussian.coco.coco2gaussian.create_gaussian_json import GaussianJsonCoco

class GaussianData():

    def __init__(self, out_dir):
        self.out_root_dir = out_dir


    def convert_coco(self, coco_data_dir, data_type, category_yaml):

        # saved path
        target_json_dir = os.path.join(self.out_root_dir, 'coco')
        # make dir
        if not os.path.exists(target_json_dir):
            os.mkdirs(target_json_dir)
        # generate json file
        gs_json = GaussianJsonCoco(target_json_dir)
        gs_json.generate_gaussian_json(data_type, category_yaml, coco_data_dir)


    def




if __name__ == "__main__":

    coco_data_dir = '/dl/data/coco'
    out_dir = './'
    data_type = 'train2017'
    category_yaml = './gaussian_categories_test.yml'

    gs_data = GaussianData(out_dir)
    gs_data.convert_coco(coco_data_dir, data_type, category_yaml)