# fast-detection-YOLO

A real-time object detection app based on lightDenseYOLO. The lightDenseYOLO implementation leverages lightDenseNet as its feature extractor and YOLO v2 as the core detection module.

Fast-detection-YOLO was trained on two renowned object detection datasets (MS COCO and Pascal VOC 07+12).

| CNN architecture               | Training Data  | mAP       | Processing time       |
|---------------------------     |--------------- |---------  |-----------------------|
| **lightDenseYOLO (2 blocks)**  | VOC            | **70.7**  | **20 ms ~ 50 fps**    |
| lightDenseYOLO (4 blocks)      | VOC            | **77.1**  | **28 ms ~ 35.8 fps**  |
| YOLO v2                        | VOC            | 75.4      | 30 ms ~ 33 fps        |
| Faster-RCNN + Resnet 101       | VOC            | 78.9      | 200 ms ~ 5 fps        |
| MobileNets + SSD               | VOC            | 73.9      | 80 ms ~ 12.5 fps      |
| lightDenseYOLO (2 blocks)      | VOC + COCO     | **79.0**  | **20 ms ~ 50 fps**    |
| lightDenseYOLO (4 blocks)      