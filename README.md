# fast-detection-YOLO

A real-time object detection app based on lightDenseYOLO. The lightDenseYOLO implementation leverages lightDenseNet as its feature extractor and YOLO v2 as the core detection module.

Fast-detection-YOLO was trained on two renowned object detection datasets (MS COCO and Pascal VOC 07+12).

| CNN architecture               | Training Data  | mAP       | Processing time       |
|---------------------------     |--------------- |---------  |-----------------------|
| **lightDenseYOLO (2 blocks)**  | VOC            | **70.7**  | **20 ms ~ 50 fps**    |
| lightDenseYOLO (4 block