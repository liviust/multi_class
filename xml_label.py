# Wirte by Julius@lenovo R&T Shenzhen
"""
    Convent voc xml file to txt label file


This script convents xml file in to txt file with the following format:

    xxx.jpg class1 class3 class4
    xxx.jpg class3 class20

Each row contains two parts:
    1. the file name the image
    2. string specifying the class names
"""

from xml.dom import minidom
import os.path as osp
import numpy as np


def load_pascal_annotation(index, pascal_root, with_diff=False):
    """
    Load the pascal annotation from xml file
    Args:
        index: integer specifying the index of the xml/image file name
        pascal_root: the root directory of the dataset
        with_diff: specify whether to use difficult objects in xml file,
                   default: False

    Returns:
        cls: string, the class names the objects in responding xml file
    """
    filename = osp.join(pascal_root, 'Annotations', index + '.xml')
    # print 'Loading: {}'.format(filename)

    def get_data_from_tag(node, tag):
        return node.getElementsByTagName(tag)[0].childNodes[0].data

    with open(filename) as f:
        data = minidom.parseString(f.read())

    objs = data.getElementsByTagName('object')
    cls = ''

    for obj in objs:
        is_diff = get_data_from_tag(obj, 'difficult')
        if with_diff or is_diff == '0':
          cls += str(get_data_from_tag(obj, "name")).lower().strip() + ' '

    return cls


def xml_2_txt(pascal_root, input_file, out_file):
    """ Convent a dataset from input_file to out_file

    Args:
        pascal_root: root directory
        input_file: txt of the dataset, train, val, test
        out_file: txt with label

    Returns:

    """
    indexlist = [line.rstrip('\n') for line in open(osp.join(pascal_root, 'ImageSets/Main', input_file))]
    f = open(out_file, 'w')
    for i, index in enumerate(indexlist):
        cls = load_pascal_annotation(index, pascal_root, with_diff=False)
        label = index + '.jpg ' + cls
        label = label.rstrip(' ') + '\n'
        f.write(label)

        if not i % 1000:
            print ("Conventing %d xml to txt" % i)
    f.close()

if __name__ == '__main__':
    pascal_root = '/home/julius/py-faster-rcnn/data/VOCdevkit2007/VOC2007'
    input_file = 'test.txt'
    out_file = 'data/test.txt'
    xml_2_txt(pascal_root, input_file, out_file)
