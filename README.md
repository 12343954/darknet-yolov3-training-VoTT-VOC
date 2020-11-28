# darknet yolov3 training (VoTT, VOC, VS2019)
**Not only the object recognition boxes, but also more information (angle, center point)**  


## 1. Installation Environment

|  software   | version  |
|  ----  | ----  |
| CUDA  | 10.1 |
| CUDNN  | 7.6.4 |
| OpenCV  | 4.2.0 |
| VS 2019 | 16.5.0 |
| Python  | 3.7.4 |
| VoTT  | 2.1.0 |

reference: [Win10 + VS2019 配置Darknet YOLOv3](https://www.zhezhi.press/2019/10/17/win10-vs2019-%E9%85%8D%E7%BD%AEdarknet-yolov3/)

GPU version, attention：  
1. installing CUDA , must select vs Compile library `visual_studio_integration`，and copy into vs compilation tools
2. install cudnn，copy cudnn files into CUDA，or else couldn't find `cudnn.h`  
3. opencv 4.2.0 only have opencv2，but does not affect the use，I use vc14's lib  
4. open darknet.sln，Prompt to upgrade–Redirect project，stay default.That is to upgrade to match the compilation environment，otherwise it will not compile，because i only installed vs 2019  
5. vs set up “Maximum number of projects generated in parallel” = 1，otherwise it will not compile
6. 【most important】，if you want to use darknet's python version，Also, compile yolo_cpp_dll.dll according to the step-by-step setting of the above compilation darknet tutorial. Then install the missing library via `pip install`，`darknet.py` has already exist，don't `pip install`  

## 2. download `darknet` souce code，and compile `release x64`
after complication，Download the python code above into `.\darknet\build\darknet\x64`


## 3. VoTT Annotate images
1. Export Settings -> Provider -> Pascal VOC, Asset State -> Only Tagged Assets
2. change xml's path to custom storage folder.

## 4. script auto-generate `voc_custom` folder
use the script `python pascal_voc_to_label.py --data_dir=training\4\4 --imgaugloop=10 --augcheck` to generate `darknet format`.
```
--data_dir #pascal voc directory
--imgaugloop  # image augmentation ，default：0，means no augmentation
--augcheck  # whether generate check image for augmentation
```
Eventually contains:
```
./voc_custom
│  coco_custom.names
│  train.txt
│  tree.txt
│  val.txt
│  voc_custom.data
│  yolov3_custom.cfg
│  
├─backup
│      yolov3_custom_1000.weights
│      yolov3_custom_last.weights
│      
└─labels
        IMG_1544.jpg
        IMG_1544.txt
        IMG_1544_aug_0.jpg
        IMG_1544_aug_0.txt
```
  **attention，change yolov3.cfg to yolov3_custom.cfg，config the cfg，then put it into `voc_custom` folder**

## 5. Training
command：
```
darknet detector train voc_custom\voc_custom.data voc_custom\yolov3_custom.cfg darknet53.conv.74
```

darknet53.conv.74, need to be downloaded separately

## 6. Finally，test your custom weight
command： 
```
python detect.py --config=voc_custom\yolov3_custom.cfg --weight=voc_custom\backup\yolov3_custom_10000.weights 
        --meta=voc_custom\voc_custom.data --image=input\IMG_1876.jpg --output=output\IMG_1876_out.jpg
```

## 7. Known issues
1. After augment the images，bbox's postion is not precise enough. It is best to manually annotate the image to enhance the images.
2. The recognition angle and size are not accurate enough，Need to improve `cvContours` algorithm.
