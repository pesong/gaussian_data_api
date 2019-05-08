"""
This scripts generate the VOC xml format from bdd100k dataset
the user should define the object list and output_dir, etc.
"""

import argparse, json
import cytoolz
from lxml import etree, objectify
import os, re
import shutil
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET

### define variables

dataType = 'train'

target_classes = ['person', 'car', 'bike', 'bus', 'rider']
output_dir = '/media/pesong/e/dl_gaussian/data/backbone_seg_ssd/bdd100k_ssd_seg/'
dataDir = '/media/pesong/e/dl_gaussian/data/backbone_seg_ssd/bdd100k_ssd_seg/seg_image/{}'.format(dataType)
img_list_file = '/media/pesong/e/dl_gaussian/data/backbone_seg_ssd/bdd100k_ssd_seg/ImageSets/Main/{}.txt'.format(dataType)
fw = open(img_list_file, 'w')

def write_xml(target_img_file, xml_path, label_dicts):
    target_imgs = open(target_img_file, 'r')
    target_imgs_num = 0
    for img in label_dicts.keys():
        root = ET.Element("annotation")
        ET.SubElement(root, "filename").text = str(img)
        sizes = ET.SubElement(root, 'size')
        ET.SubElement(sizes, 'width').text = '1280'
        ET.SubElement(sizes, 'height').text = '720'
        ET.SubElement(sizes, 'depth').text = '3'

        # img = img.rstrip('\n')
        # if img not in label_dicts.keys():
        #     continue
        print(img)
        anns = label_dicts[img]
        target_obj_nums = 0
        for ann in anns:
            if ann['category'] in target_classes:
                target_obj_nums += 1
                objects = ET.SubElement(root, 'object')
                ET.SubElement(objects, 'name').text = ann['category']
                ET.SubElement(objects, 'pose').text = 'Unspecified'
                ET.SubElement(objects, 'truncated').text = '0'
                ET.SubElement(objects, 'difficult').text = '0'

                bndbox = ET.SubElement(objects, 'bndbox')
                ET.SubElement(bndbox, 'xmin').text = str(int(ann['bbox'][0]))
                ET.SubElement(bndbox, 'ymin').text = str(int(ann['bbox'][1]))
                ET.SubElement(bndbox, 'xmax').text = str(int(ann['bbox'][2]))
                ET.SubElement(bndbox, 'ymax').text = str(int(ann['bbox'][3]))

        if target_obj_nums == 0:
            continue
        target_imgs_num += 1
        tree = ET.ElementTree(root)
        tree.write(xml_path + '/' + img.rstrip('.jpg') + '.xml', encoding='utf-8')
        fw.write(img.rstrip('.jpg') + '\n')
        if target_imgs_num == 20000:
            break




def label2dics():
    labeldicts = {}
    anns = []

    for json_type in ['train']:
        json_file = '/dl/data/backbone_seg_ssd/bdd100k_ssd_seg/detection_labels_ori_json/detection_{}.json'.format(json_type)
        labels = json.load(open(json_file, 'r'))
        for label in labels:
            img_name = label['name']
            if img_name not in labeldicts.keys():
                anns = []
            anns.append(label)
            labeldicts[img_name] = anns

    return labeldicts


def main():

    label_dicts = label2dics()

    # create path
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    img_path = os.path.join(output_dir, 'images')
    ann_path = os.path.join(output_dir, 'Annotations')
    if not os.path.exists(img_path): os.makedirs(img_path)
    if not os.path.exists(ann_path): os.makedirs(ann_path)

    write_xml(img_list_file, ann_path, label_dicts)


if __name__ == "__main__":

    main()