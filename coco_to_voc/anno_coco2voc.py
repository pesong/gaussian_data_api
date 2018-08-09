import argparse, json
import cytoolz
from lxml import etree, objectify
import os, re
import shutil

# define variables

dataType = 'train'
target_classes = ['person']     # list of target object to convert
output_dir = '/media/pesong/e/dl_gaussian/data/detect_voc'

dataDir = '/media/pesong/e/dl_gaussian/data/coco/images/{}2017'.format(dataType)
anno_file = '/media/pesong/e/dl_gaussian/data/coco/annotations/instances_{}2017.json'.format(dataType)
type = 'instance'            # annotation file for object instance/keypoint

def instance2xml_base(anno):
    E = objectify.ElementMaker(annotate=False)
    anno_tree = E.annotation(
        E.folder('VOC2017_instance/{}'.format(anno['category_id'])),
        E.filename(anno['file_name']),
        E.source(
            E.database('MS COCO 2017'),
            E.annotation('MS COCO 2017'),
            E.image('Flickr'),
            E.url(anno['coco_url'])
        ),
        E.size(
            E.width(anno['width']),
            E.height(anno['height']),
            E.depth(3)
        ),
        E.segmented(0),
    )
    return anno_tree


def instance2xml_bbox(anno, bbox_type='xyxy'):
    """bbox_type: xyxy (xmin, ymin, xmax, ymax); xywh (xmin, ymin, width, height)"""
    assert bbox_type in ['xyxy', 'xywh']
    if bbox_type == 'xyxy':
        xmin, ymin, w, h = anno['bbox']
        xmax = xmin+w
        ymax = ymin+h
    else:
        xmin, ymin, xmax, ymax = anno['bbox']
    E = objectify.ElementMaker(annotate=False)
    anno_tree = E.object(
        E.name(anno['category_id']),
        E.bndbox(
            E.xmin(round(xmin)),
            E.ymin(round(ymin)),
            E.xmax(round(xmax)),
            E.ymax(round(ymax))
        ),
        E.difficult(anno['iscrowd'])
    )
    return anno_tree


def parse_instance(content, outdir):
    categories = {d['id']: d['name'] for d in content['categories']}
    # merge images and annotations: id in images vs image_id in annotations
    merged_info_list = list(map(cytoolz.merge, cytoolz.join('id', content['images'], 'image_id', content['annotations'])))

    filtered_info_list = []
    # convert category id to name && get target object info

    for instance in merged_info_list:
        cat_name = categories[instance['category_id']]
        if cat_name in target_classes:
            instance['category_id'] = cat_name
            filtered_info_list.append(instance)


    # group by filename to pool all bbox in same file
    target_images = []
    for name, groups in cytoolz.groupby('file_name', filtered_info_list).items():
        anno_tree = instance2xml_base(groups[0])
        # if one file have multiple different objects, save it in each category sub-directory
        filenames = []
        for group in groups:
            filenames.append(os.path.join(outdir, re.sub(" ", "_", group['category_id']),
                                          'annotations', os.path.splitext(name)[0] + ".xml"))
            anno_tree.append(instance2xml_bbox(group, bbox_type='xyxy'))
        for filename in filenames:
            etree.ElementTree(anno_tree).write(filename, pretty_print=True)

        print("Formating instance xml file {} done!".format(name))


        # copy target image file to outdir
        if name not in target_images:
            target_images.append(name)
            src_dir = os.path.join(dataDir, name)
            target_dir = os.path.join(output_dir, re.sub(" ", "_", group['category_id']), 'images', name)
            shutil.copyfile(src_dir, target_dir)

    print(len(target_images))



def main():
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    content = json.load(open(anno_file, 'r'))
    if type == 'instance':
        # make subdirectories
        sub_dirs = [re.sub(" ", "_", cate['name']) for cate in content['categories']]
        for sub_dir in sub_dirs:
            if sub_dir in target_classes:
                sub_dir = os.path.join(output_dir, str(sub_dir))
                anno_dir = os.path.join(sub_dir, 'annotations')
                img_dir = os.path.join(sub_dir, 'images')
                if not os.path.exists(anno_dir):
                    os.makedirs(anno_dir)
                if not os.path.exists(img_dir):
                    os.makedirs(img_dir)
        parse_instance(content, output_dir)


if __name__ == "__main__":

    main()