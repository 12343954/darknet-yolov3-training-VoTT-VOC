from ctypes import *
import math
import random
import os
from cv2 import cv2 as cv2
import imutils
import numpy as np
import time
import darknet
from seaborn import color_palette
from PIL import Image, ImageDraw, ImageFont
from absl import app, flags
from absl.flags import FLAGS
import coloredlogs
import logging

flatui = ["#9b59b6", "#3498db", "#0a802e", "#e74c3c", "#34495e", "#fdf7fb"]
YOLOV3_COLORS_LENGHT = 10
# YOLOV3_COLORS = ((np.array(color_palette("Set2", YOLOV3_COLORS_LENGHT)) * 255)).astype(np.uint8)
# YOLOV3_COLORS = ((np.array(color_palette(flatui)) * 255)).astype(np.uint8)


def hex_to_rgb(hex):
    hex = hex.lstrip("#")
    hlen = len(hex)
    return tuple(int(hex[i : i + hlen // 3], 16) for i in range(0, hlen, hlen // 3))


YOLOV3_COLORS = list(map(lambda x: hex_to_rgb(x), flatui))


def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img):
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(
        font="./fonts/MuskBold.otf", size=(img.size[0] + img.size[1]) // 100
    )

    print("Detected {} object(s): ".format(len(detections)))
    for index, detection in enumerate(detections):
        # detection = [class, score, (x, y, w, h)]
        name = detection[0].decode()
        score = detection[1]
        x, y, w, h = (
            int(detection[2][0]),
            int(detection[2][1]),
            int(detection[2][2]),
            int(detection[2][3]),
        )
        xmin, ymin, xmax, ymax = convertBack(float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        # color = YOLOV3_COLORS[int(index % len(flatui))]
        color = YOLOV3_COLORS[int(altNames.index(name) % len(flatui))]

        print(
            "\t{:>3d}  {:<12s}  {:.2%}   {},{},  {}".format(
                index + 1, name, score, pt1, pt2, (w, h)
            )
        )
        # print(type(img))

        thickness = 2
        confidence = "{:.2f}".format(score * 100)

        text = "{} {}".format(name, confidence)
        text_size = draw.textsize(text, font=font)
        text_bg_size = (text_size[0] + thickness, text_size[1] + thickness + thickness)

        # calc contours

        # img2 = np.zeros([w, h, 3])  # create empty rect
        img2 = darknet.make_image(w, h, 3,)
        _img2 = img.crop([pt1[0], pt1[1], pt2[0], pt2[1]])
        rgb_img = _img2.convert("RGB")
        rgb_img.save("./output/d_{}.jpg".format(index))
        # img_np = np.asarray(rgb_img)
        # darknet.copy_image_from_bytes(img2, _img2.tobytes())
        # _img2 = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

        # _img2.save("./output/d_{}.jpg".format(index))
        # cv2.imwrite("./output/d_{}.jpg".format(index), _img2)
        # print(type(_img2))
        # img2 = img[pt1[1] : pt1[1] + h, pt1[0] : pt1[0] + w, :]  # copy to empty rect
        # # 获取(中心点、角度、最大轮廓、宽高)
        # (cnt_center, cnt_angle, cnt_box, cnt_size) = cvContours(pt1, img2)
        # print(cnt_center, cnt_angle, cnt_box, cnt_size)

        # draw rectangle box
        draw.rectangle(
            [pt1[0], pt1[1], pt2[0], pt2[1]], outline=tuple(color), width=thickness
        )

        # draw text background  [(x0, y0), (x1, y1)] or [x0, y0, x1, y1].
        draw.rectangle(
            [
                pt1[0] + thickness,  # x0
                pt1[1],  # y0
                pt1[0] + text_bg_size[0] + 1,  # x1
                pt1[1] + text_bg_size[1],  # y1
            ],
            fill=tuple(color),
        )

        # draw text
        draw.text(
            (pt1[0] + thickness, pt1[1] + thickness), text, fill="black", font=font,
        )

    # rgb_img = img.convert("RGB")
    # img_np = np.asarray(rgb_img)
    # img = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

    return img


def cvDrawBoxes2(detections, img):
    thickness = 2
    margin = 10  # 取边框以外再加10像素

    # 0    1      3     4   5   6       7中心点、角度、最大轮廓、宽高
    # name,cls_id,score,pt1,pt2,(w, h),contours
    # detection,contours
    list_detect = []

    print("Detected {} object(s): ".format(len(detections)))
    for index, detection in enumerate(detections):
        # detection = (class, score, (x, y, w, h))
        name = detection[0].decode()
        score = detection[1]
        x, y, w, h = (
            int(detection[2][0]),
            int(detection[2][1]),
            int(detection[2][2]),
            int(detection[2][3]),
        )
        xmin, ymin, xmax, ymax = convertBack(float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        # color = YOLOV3_COLORS[int(index % len(flatui))]
        # color = YOLOV3_COLORS[int(altNames.index(name) % len(flatui))]
        # color2 = (
        #     255,
        #     0,
        #     0,
        # )  # YOLOV3_COLORS[(int(altNames.index(name) + 1) % len(flatui))]

        # print(
        #     "\n\n\t{:>3d}  {:<12s}  {:.2%}   {},{},  {}".format(
        #         index + 1, name, score, pt1, pt2, (w, h)
        #     )
        # )

        # confidence = "{:.2f}".format(score * 100)

        # text = "{} {}".format(name, confidence)

        # calc contours
        # origin_x = pt1[0]
        # origin_y = pt1[1]
        # origin_w = w
        # origin_h = h

        origin_x = pt1[0] - margin
        origin_y = pt1[1] - margin
        origin_w = w + margin * 2
        origin_h = h + margin * 2
        _x = _y = _w = _h = 0  # 差值
        # 1, x 越界, 不够减
        if origin_x < 0:
            _x = 0 - origin_x  # 左边缩小 _x，宽度也要减小 _x
            origin_x = 0
            origin_w -= _x
        # 2, y 越界, 不够减
        if origin_y < 0:
            _y = 0 - origin_y
            origin_y = 0
            origin_h -= _y

        # 3, x+width 越界, 超出边界
        if origin_x + origin_w > WIDTH:
            _w = origin_x + origin_w - WIDTH
            origin_w -= _w

        # 4, y+height 越界，超出边界
        if origin_y + origin_h > HEIGHT:
            _h = origin_y + origin_h - HEIGHT
            origin_h -= _h

        # img2 = darknet.make_image(w, h, 3,)
        img2 = np.zeros([origin_w, origin_h, 3])  # create empty rect
        # img2 = img[pt1[1] : pt1[1] + h, pt1[0] : pt1[0] + w, :]
        img2 = img[
            origin_y : origin_y + origin_h,  # y:y+h
            origin_x : origin_x + origin_w,  # x:x+w
            :,
        ]

        _save_path = "./upload/{}_d_{}.jpg".format(imageName, index)
        cv2.imwrite(_save_path, img2)
        # # 获取(中心点、角度、最大轮廓、宽高)
        # contours = cvContours((origin_x, origin_y), img2)
        contours = get_contours((origin_x, origin_y), img2)
        # print("\t\t", type(contours))

        # 做数据 (detection,contour)
        list_detect.append([name, score, (pt1, pt2), (w, h), contours])
        # print(
        #     "\n-------------------\n",
        #     [name, score, (pt1, pt2), (w, h), contours],
        #     "\n+++++++++++++++++++\n",
        # )

    # print(list_detect)

    # print(tplt.format(index + 1, name, score, point_xy, angle, mr_w, mr_h))
    # tplt = "{0:>2d}  {1:<14}  {2:>5}  {3:<10}  {4:^2}  {5}x{6}"
    # tplt = "{0:>10}\t{1:^10}\t{2:<10}"

    for index, detect in enumerate(list_detect):
        # print(len(detect), type(detect[0]))
        # print(detect[0], detect[1], detect[2], detect[3], detect[4])

        name, score, xy, size, cnt = (
            detect[0],
            detect[1],
            detect[2],
            detect[3],
            detect[4],
        )
        color = YOLOV3_COLORS[int(altNames.index(name) % len(flatui))]
        color2 = (
            255,
            0,
            0,
        )  # YOLOV3_COLORS[(int(altNames.index(name) + 1) % len(flatui))]

        # 画识别框
        # cv2.rectangle(img, xy[0], xy[1], color, 2)

        # 画轮廓框contours
        if cnt:
            # 中心点、角度、最大轮廓、宽高
            # print(
            #     "\t\tcenter={} , angle={}° , cnt={} , size={}".format(
            #         cnt[0], cnt[1], cnt[2], cnt[3]
            #     )
            # )

            point_xy, angle, box, (mr_w, mr_h) = cnt
            # 画轮廓
            cv2.drawContours(img, [box], 0, color, 2)
            # 画name
            color = YOLOV3_COLORS[int(altNames.index(name) % len(flatui))]
            cv2.putText(
                img,
                name,
                (point_xy[0] - 20, point_xy[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,  # =[255, 255, 255],
                2,
                # 1,
                # 1,
            )
            # 画文字背景
            # cv2.rectangle(
            #     img,
            #     (point_xy[0] - mr_w / 2, point_xy[1] - 10),
            #     (point_xy[0] - mr_w / 2, point_xy[1] - 10),
            #     color,
            #     cv2.FILLED,
            # )

            # 画角度
            # cv2.putText(
            #     img,
            #     angle,
            #     (point_xy[0] - 5, point_xy[1] + 10),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.5,
            #     [255, 255, 255],
            #     # 1,
            #     # 1,
            #     # 1,
            # )
            # cv2.putText(
            #     img,
            #     angle,
            #     (point_xy[0] - 5, point_xy[1] + 10),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     fontScale=0.5,
            #     color=[255, 255, 255],
            #     thickness=1,
            # )

            # for cnt in contours:
            #     # print("\n", cnt[2], "\n")
            #     cv2.drawContours(img, [cnt[2]], 0, color2, 2)

            # 画文字背景
            # (text_width, text_height) = cv2.getTextSize(
            #     name, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1
            # )[0]
            # print(text_width, text_height)

            # 文字背景框 x,y,w,h
            # back_w = text_width + thickness + thickness
            # back_h = text_height + thickness + thickness
            # back_x = pt1[0]
            # back_y = pt1[1] - back_h
            # back_x2 = back_x + back_w
            # back_y2 = back_y + back_h

            # if back_y < 0:
            #     back_y = 0
            #     back_y2 = back_h

            # cv2.rectangle(img, (back_x, back_y), (back_x2, back_y2), color, cv2.FILLED)

            # 画文字
            # cv2.putText(
            #     img,
            #     name,
            #     (pt1[0], pt1[1] - 5),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.5,
            #     [0, 0, 0],  # color,  # [0, 255, 0],
            #     thickness=1,
            # )

            print(
                "\t{:>2d}  {:<16s}  {:.2%}  {},  {}°,  {}x{}".format(
                    index + 1, name, score, point_xy, angle, mr_w, mr_h
                )
            )
            # print(tplt.format(name, score, angle, chr(12288)))

        # print(
        #     "\n\n\t{:>3d}  {:<12s}  {:.2%}   {},{},  {}".format(
        #         index + 1, name, score, pt1, pt2, (w, h)
        #     )
        # )

    return img


def get_contours(origin, img):
    # window_title = "im"

    # 获取轮廓
    # img = cv2.imread(image_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    edges = cv2.Canny(blur, 10, 200)  # Edge detection
    edges = cv2.dilate(edges, None)  # 默认(3x3)
    edges = cv2.erode(edges, None)

    # Find contours in edges（binary image）
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    max_contour = contours[0]

    epsilon = 0.001 * cv2.arcLength(max_contour, True)
    approx = cv2.approxPolyDP(max_contour, epsilon, True)
    # print(approx)
    box = cv2.minAreaRect(approx)
    center, wh, angle = box
    angle = int(abs(angle))
    wh = (int(wh[0]), int(wh[1]))

    center = (int(center[0]), int(center[1]))
    # print(center, wh, angle)
    # cv2.putText(
    #     img, "{}".format(angle), center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
    # )

    box = cv2.boxPoints(box)
    # 坐标转换成大图坐标系
    center = np.array(center) + np.array(origin)
    box = np.array(box) + np.array(origin)
    box = np.int0(box)

    # cv2.drawContours(img, [box], -1, 255, 2)

    # cv2.imshow(window_title, img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # return 中心点、角度、最大轮廓、宽高
    return center, angle, box, wh


def cvContours(origin, image):
    """
    功能：cvContours(原点, image)
    描述：获取(中心点、角度、最大轮廓、宽高)
    """
    # h, w, _ = image.shape
    # print("\t\timage Area: ", h, "x", w, " = ", h * w)
    # return

    b, g, r = np.double(cv2.split(image))
    shadow_ratio = (4 / np.pi) * np.arctan2((b - g), (b + g))
    shadow_ratio = np.uint8(shadow_ratio * 255)

    canny = cv2.Canny(shadow_ratio, 64, 128)
    eroded = cv2.erode(
        canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1)), iterations=1
    )
    dilated = cv2.dilate(
        eroded, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=3
    )
    eroded = cv2.erode(
        dilated, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=3
    )

    """
    cv2.RETR_EXTERNAL   表示只检测外轮廓
    cv2.RETR_LIST       检测的轮廓不建立等级关系
    cv2.RETR_CCOMP      建立两个等级的轮廓，上面的一层为外边界，里面的一层为内孔的边界信息。如果内孔内还有一个连通物体，这个物体的边界也在顶层。
    cv2.RETR_TREE       建立一个等级树结构的轮廓。
    """
    cnts = cv2.findContours(eroded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # thickness = 2  # border width

    # 画中心点
    # point_size = 1
    # point_color = (0, 255, 0)  # BGR
    # point_thickness = 4  # 可以为 0 、4、8

    cnts = imutils.grab_contours(cnts)

    max_index = -1
    max_area = 0.0

    for i, c in enumerate(cnts):
        # print("---------- " * 5)
        area = cv2.contourArea(c)
        # print("\t\t ----------------- cnt area = \t", area)
        if area == 0.0:
            continue

        if area > max_area:
            max_index = i
            max_area = area
            continue

    # print("\t\t", max_index, max_area)

    if max_index == -1:
        return None

    c = cnts[max_index]
    rect = cv2.minAreaRect(c)  # [0]center,[1]width+height,[2]angle,
    mr_w = int(rect[1][0])
    mr_h = int(rect[1][1])

    x, y = rect[0]  # 中心点

    box = cv2.boxPoints(rect)

    # 中心点
    point_xy = (int(x), int(y))

    # 角度
    angle = abs(int(rect[2]))

    # 坐标转换成大图坐标系
    point_xy = np.array(point_xy) + np.array(origin)
    box = np.array(box) + np.array(origin)
    box = np.int0(box)

    # return 中心点、角度、最大轮廓、宽高
    return point_xy, angle, box, (mr_w, mr_h)


netMain = None
metaMain = None
altNames = None
imageName = None  # 输入图片名字


flags.DEFINE_string("config", "./cfg/yolov3.cfg", "path to config file")
flags.DEFINE_string("weight", "./yolov3.weights", "path to weights file")
flags.DEFINE_string("meta", "./cfg/coco.data", "path to weights file")
flags.DEFINE_string("image", "./data/dog.jpg", "path to image file")
flags.DEFINE_string("output", "./output.jpg", "path to output image")
flags.DEFINE_boolean("open", True, "is open output image or not")
flags.DEFINE_integer("size", 416, "resize images to")
flags.DEFINE_float("thresh", 0.5, "confidence, default:50%")
flags.DEFINE_boolean("extra_out", None, "enable console out more information")


def YOLO(args):

    logger = logging.getLogger(__name__)
    coloredlogs.install(level="DEBUG", logger=logger)

    global metaMain, netMain, altNames, imageName, HEIGHT, WIDTH
    # configPath = "./cfg/yolov3.cfg"
    # weightPath = "./yolov3.weights"
    # metaPath = "./cfg/coco.data"
    configPath = FLAGS.config
    weightPath = FLAGS.weight
    metaPath = FLAGS.meta

    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" + os.path.abspath(configPath) + "`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" + os.path.abspath(weightPath) + "`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" + os.path.abspath(metaPath) + "`")
    if netMain is None:
        netMain = darknet.load_net_custom(
            configPath.encode("ascii"), weightPath.encode("ascii"), 0, 1
        )  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re

                match = re.search(
                    "names *= *(.*)$", metaContents, re.IGNORECASE | re.MULTILINE
                )
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass

    while True:
        image_path = input("\ninput a image to detect (press 'q' to exit): ")
        if image_path == "q":
            logger.debug("quit the code 0")
            break

        if image_path == "" and FLAGS.image != "":
            image_path = FLAGS.image
            imageName = os.path.basename(image_path)

        if os.path.isfile(image_path) == False:
            logger.error("NOT exist file: %s" % image_path)
            continue

        logger.debug("YOLO Starts detecting ...")

        prev_time = time.time()

        frame_read = cv2.imread(image_path)
        HEIGHT, WIDTH, channels = frame_read.shape

        # 创建副本
        frame_copy = np.zeros(frame_read.shape, np.uint8)
        # 复制副本, 用于后面的画框和轮廓
        frame_copy = frame_read.copy()
        # cv2.imwrite("./output/frame_copy.jpg", frame_copy)

        #  <class 'numpy.ndarray'>
        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)

        frame_resized = cv2.resize(
            frame_rgb, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR,
        )

        # Create an image we reuse for detect
        darknet_image = darknet.make_image(WIDTH, HEIGHT, 3,)

        # darknet.copy_image_from_bytes(darknet_image, frame_read.tobytes())
        darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())

        # detect image
        detections = darknet.detect_image(
            # netMain, metaMain, darknet_image, thresh=FLAGS.thresh
            netMain,
            metaMain,
            darknet_image,
            thresh=FLAGS.thresh,
        )

        # image = cvDrawBoxes(detections, frame_rgb)
        # image = cvDrawBoxes(detections, frame_resized)
        image = cvDrawBoxes2(detections, frame_copy)

        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Estimated Time of Arrival
        ETA = "ETA: {} ms".format((time.time() - prev_time) * 1000)
        # print(ETA)

        # try:
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(FLAGS.output, image)
        # image.save(FLAGS.output)
        logger.debug("output saved to: {}, {}".format(FLAGS.output, ETA))
        # except Exception as ex:
        #     print(ex)

        if FLAGS.open:
            if 1 == 0:
                logger.error("None detected: 0")
            else:
                img2 = Image.open(FLAGS.output)
                img2.show()


if __name__ == "__main__":
    app.run(YOLO)
