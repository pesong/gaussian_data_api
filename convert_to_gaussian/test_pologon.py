from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab


def main():
    pylab.rcParams['figure.figsize'] = (8.0, 10.0)



    # dataDir='/media/pesong/e/dl_gaussian/data/gaussian/images'

    dataDir='/home/pesong/Documents/mini_gaussian_json/images'
    dataType='val'
    # annFile='/media/pesong/e/dl_gaussian/data/gaussian/annotations/instances_{}2017.json'.format(dataType)
    annFile='/home/pesong/Documents/mini_gaussian_json/annotations/instances_{}2017.json'.format(dataType)



    # initialize COCO api for instance annotations
    coco=COCO(annFile)

    # display COCO categories and supercategories
    cats = coco.loadCats(coco.getCatIds())
    nms=[cat['name'] for cat in cats]
    print('COCO categories: \n{}\n'.format('.  '.join(nms)))

    # nms = set([cat['supercategory'] for cat in cats])
    # print('COCO supercategories: \n{}'.format(' '.join(nms)))

    # get all images containing given categories, select one at random
    catIds = coco.getCatIds(catNms=['vegetation'])
    print(catIds)
    imgIds = coco.getImgIds(catIds=catIds)
    imgIds = coco.getImgIds(imgIds = [532481])
    img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
    print(str(img['id']))

    # load and display image
    print(img['file_name'])
    I = io.imread('%s/%s2017/%s'%(dataDir, dataType ,img['file_name']))
    # use url to load image
    # I = io.imread(img['coco_url'])
    plt.axis('off')
    plt.imshow(I)
    plt.show()

    # load and display instance annotations
    plt.imshow(I); plt.axis('off')
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds)
    anns = coco.loadAnns(annIds)
    print(len(anns))
    print([ann['category_id'] for ann in anns])
    # print([ann['bbox'] for ann in anns ])

    coco.showAnns(anns)


if __name__ == "__main__":
    main()