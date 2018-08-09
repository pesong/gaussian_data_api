'''

This scripts aimed to merge the gaussian formatted images and labels converted from
the other dataset such as coco/cityscapes .etc.

We intend to:
 1. unify the imgid, file_name with different sources
 2. merge the label json file
 3. copy the sourced images to target dir

'''
from convert_to_gaussian.coco.coco2gaussian.create_gaussian_json import GaussianJson

class GaussianData():

    def __init__(self):
        # to avoid the conflict of img_id
        self.img_id = 0

        print()

    def convert_coco(self):

