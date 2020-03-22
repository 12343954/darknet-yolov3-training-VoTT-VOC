"""
convert VoTT Pascal Voc format to darknet text label format

smith
2020-3-8


VOC structure
│  tree.txt
│  
└─1
    │  coco_custom.names
    │  yolov3_custom.cfg
    │  
    └─Pascal_VOC_dataset
        │  pascal_label_map.pbtxt
        │  
        ├─Annotations
        │      IMG_1512.xml
        │      IMG_1513.xml
        │      
        ├─ImageSets
        │  └─Main
        │          peanut_train.txt
        │          peanut_val.txt
        │          sunflower seed_train.txt
        │          sunflower seed_val.txt
        │          
        └─JPEGImages
                IMG_1512.png
                IMG_1513.png
                
"""
import xml.etree.ElementTree as ET
import pickle
import os, shutil
from os import listdir, getcwd
from os.path import join

import time
import hashlib
from absl import app, flags
from absl.flags import FLAGS
import lxml.etree
import tqdm
import coloredlogs
import logging
import json, re

import pickle
import numpy as np
from PIL import Image, ImageDraw
import imgaug as ia
from imgaug import augmenters as iaa

logger = logging.getLogger(__name__)
coloredlogs.install(level="DEBUG", logger=logger)

flags.DEFINE_string(
    "data_dir", "./training/pascal_voc", "path to raw PASCAL VOC dataset"
)
flags.DEFINE_string("output", None, "output label text path")
flags.DEFINE_boolean("imgaug", None, "wether image aug")
flags.DEFINE_integer("imgaugloop", 0, "aug times")
flags.DEFINE_boolean("augcheck", None, "wether generate validation augment images")


# /ImageSets/Main
FILE_TYPES = ["train", "val"]


def convert(size, box):
    dw = 1.0 / (size[0])
    dh = 1.0 / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


# convert_annotation()
# in_file: input file path
# in_name: input file name
# class_id: id in classes
def convert_annotation(data_dir, out_dir, out_label_dir, in_file, in_name, classes):
    # in_file = open("VOCdevkit/VOC%s/Annotations/%s.xml" % (year, image_id))
    # name, _ = in_name.split(".")
    # out_file = os.path.join(out_label_dir, "%s.txt" % (name))

    # print(os.path.basename(out_file))
    # return
    annotation_xml = lxml.etree.fromstring(open(in_file).read())
    root = parse_xml(annotation_xml)["annotation"]
    filename = root["filename"]

    name, _ = filename.split(".")
    out_file = os.path.join(out_label_dir, "%s.txt" % (name))

    size = root["size"]
    w = int(size["width"])
    h = int(size["height"])
    if "object" in root:
        for obj in root["object"]:
            difficult = obj["difficult"]
            if int(difficult) == 1:
                continue

            name = obj["name"]
            cls_id = classes.index(name)

            xmlbox = obj["bndbox"]
            b = (
                float(xmlbox["xmin"]),
                float(xmlbox["xmax"]),
                float(xmlbox["ymin"]),
                float(xmlbox["ymax"]),
            )
            bb = convert((w, h), b)

            if os.path.exists(out_file):
                shutil.copy2(
                    os.path.join(data_dir, "JPEGImages", filename), out_label_dir
                )
                with open(out_file, "a") as f:
                    f.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + "\n")
            else:
                with open(out_file, "w") as f:
                    f.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + "\n")
                # print(str(cls_id) + " " + " ".join([str(a) for a in bb]) + "\n")


def parse_xml(xml):
    if not len(xml):
        return {xml.tag: xml.text}
    result = {}
    for child in xml:
        child_result = parse_xml(child)
        if child.tag != "object":
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}


# 读取xml 的 object 节点
def read_xml_annotation(root, image_id):
    # in_file = open(os.path.join(root, image_id))
    in_file = open(root)
    tree = ET.parse(in_file)
    root = tree.getroot()
    bndboxlist = []

    for object in root.findall("object"):  # 找到root节点下的所有country节点
        bndbox = object.find("bndbox")  # 子节点下节点rank的值
        # print(bndbox)

        xmin = int(float(bndbox.find("xmin").text))
        xmax = int(float(bndbox.find("xmax").text))
        ymin = int(float(bndbox.find("ymin").text))
        ymax = int(float(bndbox.find("ymax").text))
        # print(xmin,ymin,xmax,ymax)
        bndboxlist.append([xmin, ymin, xmax, ymax])
        # print(bndboxlist)

    bndbox = root.find("object").find("bndbox")
    return bndboxlist


# 修改&创建xml
def change_xml_list_annotation(root, image_id, new_target, saveroot, id):

    # in_file = open(os.path.join(root, str(image_id) + ".xml"))  # 这里root分别由两个意思
    in_file = open(os.path.join(root))  # 这里root分别由两个意思
    tree = ET.parse(in_file)
    xmlroot = tree.getroot()
    index = 0

    for object in xmlroot.findall("object"):  # 找到root节点下的所有country节点
        bndbox = object.find("bndbox")  # 子节点下节点rank的值

        # xmin = int(bndbox.find('xmin').text)
        # xmax = int(bndbox.find('xmax').text)
        # ymin = int(bndbox.find('ymin').text)
        # ymax = int(bndbox.find('ymax').text)

        new_xmin = new_target[index][0]
        new_ymin = new_target[index][1]
        new_xmax = new_target[index][2]
        new_ymax = new_target[index][3]

        xmin = bndbox.find("xmin")
        xmin.text = str(new_xmin)
        ymin = bndbox.find("ymin")
        ymin.text = str(new_ymin)
        xmax = bndbox.find("xmax")
        xmax.text = str(new_xmax)
        ymax = bndbox.find("ymax")
        ymax.text = str(new_ymax)

        index = index + 1

    new_name = str(image_id) + "_aug_" + str(id)  # IMG_1544_aug_0
    xmlroot.find("filename").text = xmlroot.find("filename").text.replace(
        image_id, new_name
    )
    xmlroot.find("path").text = xmlroot.find("path").text.replace(image_id, new_name)

    saved_path = os.path.join(saveroot, new_name + ".xml")
    tree.write(saved_path)
    return saved_path


# 图像增强主函数
# FLAGS.data_dir, out_dir, out_label_dir, annotation_xml, name, classes
def image_augment(seq, data_dir, out_dir, out_label_dir, annotation_xml, name, classes):
    boxes_img_aug_list = []
    imgaug_files = []
    new_bndbox_list = []

    bndbox = read_xml_annotation(annotation_xml, name)
    for epoch in range(FLAGS.imgaugloop):
        seq_det = seq.to_deterministic()  # 保持坐标和图像同步改变，而不是随机

        # 读取图片
        img = Image.open(os.path.join(data_dir, "JPEGImages", name[:-4] + ".jpg"))
        img = np.array(img)

        # bndbox 坐标增强
        for i in range(len(bndbox)):
            temp_bndbox_list = []
            bbs = ia.BoundingBoxesOnImage(
                [
                    ia.BoundingBox(
                        x1=bndbox[i][0],
                        y1=bndbox[i][1],
                        x2=bndbox[i][2],
                        y2=bndbox[i][3],
                    ),
                ],
                shape=img.shape,
            )

            bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
            boxes_img_aug_list.append(bbs_aug)

            # new_bndbox_list:[[x1,y1,x2,y2],...[],[]]
            new_bndbox_list.append(
                [
                    int(bbs_aug.bounding_boxes[0].x1),
                    int(bbs_aug.bounding_boxes[0].y1),
                    int(bbs_aug.bounding_boxes[0].x2),
                    int(bbs_aug.bounding_boxes[0].y2),
                ]
            )

        # 存储变化后的图片
        image_aug = seq_det.augment_images([img])[0]
        auged_name = str(name[:-4]) + "_aug_" + str(epoch) + ".jpg"

        path = os.path.join(data_dir, "JPEGImages", auged_name)
        image_auged = bbs.draw_on_image(image_aug, thickness=0)
        image_saved = Image.fromarray(image_auged)
        image_saved.save(path)

        # 画验证框
        if FLAGS.augcheck:
            draw = ImageDraw.Draw(image_saved)
            for rect in new_bndbox_list:
                draw.rectangle(rect, outline=(0, 255, 0), width=2)

            val_path = os.path.join(
                data_dir,
                "JPEGImages",
                str(name[:-4]) + "_aug_" + str(epoch) + "_val.jpg",
            )
            image_saved.save(val_path)

        # 存储变化后的XML
        auged_xml = change_xml_list_annotation(
            annotation_xml,
            name[:-4],
            new_bndbox_list,
            os.path.join(data_dir, "Annotations"),
            epoch,
        )
        # print(auged_name)

        convert_annotation(
            data_dir, out_dir, out_label_dir, auged_xml, auged_name, classes
        )

        # 加进组imgaug_files
        imgaug_files.append(os.path.relpath(os.path.join(out_label_dir, auged_name)))

        new_bndbox_list = []

    return imgaug_files


def main(_argv):
    if not os.path.exists(FLAGS.data_dir):
        raise ValueError(
            "Invalid Pascal Voc dataset dir path `"
            + os.path.abspath(FLAGS.data_dir)
            + "`"
        )

    output = "darknet_traning_label.txt"

    if FLAGS.output:
        if os.path.isdir(FLAGS.output):
            output = os.path.join(FLAGS.output, output)
        else:
            # os.path.isfile(FLAGS.output):
            output = os.path.abspath(FLAGS.output)

    pbtxt = "pascal_label_map.pbtxt"
    if not os.path.exists(os.path.join(FLAGS.data_dir, pbtxt)):
        raise ValueError(
            "NOT exist `pascal_label_map.pbtxt` under `"
            + os.path.abspath(FLAGS.data_dir)
            + "`"
        )

    pbtxt = os.path.join(FLAGS.data_dir, pbtxt)
    pb_text = open(pbtxt).read().splitlines()

    # for t in filter(lambda x: x.strip().startswith("name:"), pb_text):
    #     print(re.findall(r"name: '(.*)'", t.strip())[0])

    # create output dir
    out_dir = os.path.relpath(FLAGS.data_dir)
    out_dir = out_dir[: out_dir.rfind(os.sep)]
    out_dir = os.path.join(out_dir, "voc_custom")
    out_label_dir = os.path.join(out_dir, "labels")
    out_train_txt = os.path.join(out_dir, "train.txt")
    out_val_txt = os.path.join(out_dir, "val.txt")
    out_data_txt = os.path.join(out_dir, "voc_custom.data")
    out_backup_dir = os.path.join(out_dir, "backup")

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir, ignore_errors=True)

    os.mkdir(out_dir)
    os.mkdir(out_label_dir)
    os.mkdir(out_backup_dir)
    open(out_train_txt, "a").close()
    open(out_val_txt, "a").close()

    pb_text = filter(lambda x: x.strip().startswith("name:"), pb_text)
    classes = list()
    for t in pb_text:
        classes.append(re.findall(r"name: '(.*)'", t.strip())[0])

    coco_names = os.path.join(out_dir, "coco_custom.names")

    # create coco.names
    with open(coco_names, "w") as f:
        # f.writelines(list("%s\n" % item for item in classes))
        f.write("\n".join(classes))

    logger.debug("\n`coco.names` created : {}".format(os.path.relpath(coco_names)))

    # map `/ImageSets/Main/*.txt`, ["train","val"]
    train_list = list()
    val_list = list()

    # 1. 把文件里的内容合并到2个list中
    for t in FILE_TYPES:
        for index, name in enumerate(classes):
            txt_file = os.path.relpath(
                os.path.join(
                    FLAGS.data_dir, "ImageSets", "Main", ("%s_%s.txt" % (name, t))
                )
            )
            # print(index, txt_file)
            # print(name)
            image_list = open(txt_file).read().splitlines()
            if t == "train":
                train_list.extend(image_list)
            else:
                val_list.extend(image_list)

    # 2. 去重复
    train_list = [x.split(" ")[0] for x in train_list]
    train_list = list(set(train_list))

    val_list = [x.split(" ")[0] for x in val_list]
    val_list = list(set(val_list))

    # print("train %d", len(train_list))
    # print("train %d", len(val_list))

    # 3. 开始生成 train text
    logger.info("\nmapping [%s] files %d …………" % ("train", len(train_list)))

    if FLAGS.imgaugloop > 0:
        # 影像增强
        seq = iaa.Sequential(
            [
                iaa.Flipud(0.5),  # 对50%的图片进行垂直镜像翻转vertically flip 20% of all images
                iaa.Fliplr(0.5),  # 对50%的图片进行水平镜像翻转
                iaa.Multiply((1.2, 1.5)),  # 更改图像的亮度（原始值的50-150％）。
                # iaa.GaussianBlur(sigma=(0, 3.0)),  # iaa.GaussianBlur(0.5),
                # 使用来模糊每个图像的强度＃高斯模糊（0和3.0之间的sigma），
                # #平均/均匀模糊（内核大小在2x2和7x7之间）
                # #中值模糊（内核大小在3x3和11x11之间）。
                iaa.OneOf(
                    [
                        iaa.GaussianBlur((0, 3.0)),
                        iaa.AverageBlur(k=(2, 7)),
                        iaa.MedianBlur(k=(3, 11)),
                    ]
                ),
                # 给某些图像添加高斯噪声。
                # 在这些情况的50％中，噪声是根据频道和像素。
                # 在所有其他情况的50％中，每个采样一次＃像素（即亮度变化）。
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255)),
                iaa.Affine(
                    translate_px={"x": 15, "y": 15}, scale=(0.8, 0.95), rotate=(-30, 30)
                ),  # translate by 40/60px on x/y axis, and scale to 50-70%, affects BBs
            ]
        )

    list_train = []
    list_val = []
    for name in tqdm.tqdm(train_list):
        list_temp = []
        annotation_xml = os.path.join(
            FLAGS.data_dir,
            "Annotations",
            name.replace(".png", ".xml").replace(".jpg", ".xml"),
        )
        convert_annotation(
            FLAGS.data_dir, out_dir, out_label_dir, annotation_xml, name, classes
        )

        # train_list[i] = "%s%s%s" % (out_label_dir, os.sep, name)
        list_train.append("%s%s%s" % (out_label_dir, os.sep, name))

        # 图像增强，每张10次
        if FLAGS.imgaugloop > 0:
            list_temp = image_augment(
                seq,
                FLAGS.data_dir,
                out_dir,
                out_label_dir,
                annotation_xml,
                name,
                classes,
            )
            list_train.extend(list_temp)

    # 4. 开始生成 val text
    logger.info("\nmapping [%s] files %d …………" % ("val", len(val_list)))
    for name in tqdm.tqdm(val_list):
        annotation_xml = os.path.join(
            FLAGS.data_dir,
            "Annotations",
            name.replace(".png", ".xml").replace(".jpg", ".xml"),
        )
        convert_annotation(
            FLAGS.data_dir, out_dir, out_label_dir, annotation_xml, name, classes
        )
        # val_list[i] = "%s%s%s" % (out_label_dir, os.sep, name)
        list_val.append("%s%s%s" % (out_label_dir, os.sep, name))

        # 图像增强，每张10次
        if FLAGS.imgaugloop > 0:
            list_temp = image_augment(
                seq,
                FLAGS.data_dir,
                out_dir,
                out_label_dir,
                annotation_xml,
                name,
                classes,
            )
            list_val.extend(list_temp)

    with open(out_train_txt, "w") as f:
        f.write("\n".join(list_train))
        logging.info("train text saved: {}".format(out_train_txt))

    with open(out_val_txt, "w") as f:
        f.write("\n".join(list_val))
        logging.info("val text saved: {}".format(out_val_txt))

    out_info = "\n".join(
        [
            "classes=%s" % len(classes),
            "train=%s" % out_train_txt,
            "val=%s" % out_val_txt,
            "names=%s" % coco_names,
            "backup=%s" % out_backup_dir,
            "eval=coco",
        ]
    )
    with open(out_data_txt, "w") as f:
        f.write(out_info)

    # All done!
    print("\n\n")
    print("Finally output:\n\n%s" % out_info)
    print("\n\nAll done!\n\n")
    logger.warning(
        "\nNext, create your own `yolov3.cfg`, and put it in the `voc_custom` folder! \n"
    )


if __name__ == "__main__":
    app.run(main)
