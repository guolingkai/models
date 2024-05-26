# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
import os
from os import getcwd

# sets = ['train', 'val', 'test']
sets = ['train', 'val']
# classes = ['immature Koroneiki',
# 'half-matured Koroneiki',
# 'fully mature Koroneiki',
# 'immature Giarffa',
# 'half-matured Giarffa',
# 'fully mature Giarffa',
# 'immature Leucoarpa Round',
# 'half-matured Leucoarpa Round',
# 'fully mature Leucoarpa Round']


classes=['Bollworm', 'Meadow borer', 'Gryllotalpa orientalis', 'Little Gecko',
         'Agriotes fuscicollis Miwa', 'Nematode trench','Athetis lepigone', 'Scotogramma trifolii Rottemberg',
         'Armyworm', 'Spodoptera cabbage', 'Anomala corpulenta','Spodoptera exigua',
         'Plutella xylostella', 'holotrichia parallela', 'Rice planthopper', 'Yellow tiger',
        'Land tiger', 'eight-character tiger', 'holotrichia oblita', 'Stem borer',
         'Striped rice bore','Rice Leaf Roller', 'Spodoptera litura', 'Melahotus']
#
# pest24_names =  {
#         'Rice planthopper': 1,  'Rice Leaf Roller': 2, 'Striped rice bore':3,  'Armyworm': 5,  'Bollworm' : 6,
#         'Meadow borer': 7,  'Athetis lepigone': 8, 'Spodoptera litura': 10,   'Spodoptera exigua': 11,
#         'Stem borer':12,  'Little Gecko': 13,  'Plutella xylostella': 14, 'Spodoptera cabbage': 15,
#         'Scotogramma trifolii Rottemberg': 16,  'Yellow tiger': 24, 'Land tiger': 25,  'eight-character tiger': 28,
#          'holotrichia oblita':29,  'holotrichia parallela': 31,   'Anomala corpulenta': 32, 'Gryllotalpa orientalis': 34,
#         'Nematode trench': 35, 'Agriotes fuscicollis Miwa': 36, 'Melahotus': 37
# }

# classes=['red', 'yellow', 'green', 'off']


abs_path = os.getcwd()
print(abs_path)


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h

'''

def convert_annotation(image_id):
    in_file = open('D:/projects/Pest24/VOCdevkit/voc2007/Annotations/%s.xml' % (image_id), encoding='UTF-8')  # D:\projects\Pest24\VOCdevkit\voc2007
    out_file = open('D:/projects/Pest24/VOCdevkit/voc2007/labels/%s.txt' % (image_id), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        # difficult = obj.find('Difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        b1, b2, b3, b4 = b
        # 标注越界修正
        if b2 > w:
            b2 = w
        if b4 > h:
            b4 = h
        b = (b1, b2, b3, b4)
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


wd = getcwd()
for image_set in sets:
    if not os.path.exists('D:/projects/Pest24/VOCdevkit/voc2007/labels/'):
        os.makedirs('D:/projects/Pest24/VOCdevkit/voc2007/labels/')
    image_ids = open('D:/projects/Pest24/VOCdevkit/voc2007/ImageSets/%s.txt' % (image_set)).read().strip().split()

    if not os.path.exists('D:/projects/Pest24/VOCdevkit/voc2007/yolo_path/'):
        os.makedirs('D:/projects/Pest24/VOCdevkit/voc2007/yolo_path/')

    list_file = open('D:/projects/Pest24/VOCdevkit/voc2007/yolo_path/%s.txt' % (image_set), 'w')  # 相对路径
    for image_id in image_ids:
        list_file.write('D:/projects/Pest24/VOCdevkit/voc2007/images/%s.jpg\n' % (image_id))
        convert_annotation(image_id)
    list_file.close()

'''

set_path = '/Users/liyuhai/Documents/Code/pytool/suining_cqcb/'

def convert_annotation(image_id):
    in_file = open(set_path + 'Annotations/%s.xml' % (image_id), encoding='UTF-8')  # D:\projects\Pest24\VOCdevkit\voc2007
    out_file = open(set_path + 'labels/%s.txt' % (image_id), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        # difficult = obj.find('Difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        b1, b2, b3, b4 = b
        # 标注越界修正
        if b2 > w:
            b2 = w
        if b4 > h:
            b4 = h
        b = (b1, b2, b3, b4)
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


wd = getcwd()
for image_set in sets:
    if not os.path.exists(set_path + 'labels/'):
        os.makedirs(set_path + 'labels/')
    image_ids = open(set_path + 'ImageSets/%s.txt' % (image_set)).read().strip().split()

    if not os.path.exists(set_path + 'yolo_path/'):
        os.makedirs(set_path + 'yolo_path/')

    list_file = open(set_path + 'yolo_path/%s.txt' % (image_set), 'w')  # 相对路径
    for image_id in image_ids:
        list_file.write(set_path + 'images/%s.jpg\n' % (image_id))
        convert_annotation(image_id)
    list_file.close()



