# YOLO (Standard e Tiny) & SSD - Setup e Testes

Testes para onfigurar e executar  detecção de objetos usando YOLO (Standard e Tiny) e SSD (Single Shot Detector).





### Python

1. Dependências :

   ```bash
   pip install opencv-python opencv-python-headless numpy

   ```

2.   YOLO:

   ```bash
   wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
   wget https://pjreddie.com/media/files/yolov3.weights
   wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names

   ```

   OR
   
   ###    unzip yolo/yolov3.weights.z7 para extrair yolov3.weights

   TINY

   ```bash
   wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg
   wget https://pjreddie.com/media/files/yolov3-tiny.weights

   ```

3.   SSD:

   ```bash

   wget https://github.com/chuanqi305/MobileNet-SSD/raw/master/deploy.prototxt -O MobileNetSSD_deploy.prototxt
   wget https://github.com/chuanqi305/MobileNet-SSD/raw/master/MobileNetSSD_deploy.caffemodel

   ```

4. Utils:


   ls /dev/video*


## Ajustes e Melhorias

1. **Ajuste de Confiança:**

   - Modifica os limites de confiança nos scripts para filtrar detecções:
     ```python
     if confidence > 0.5:  # Ajuste este valor conforme necessário
     ```

2. **Redução de Resolução:**

   - Para aumentar a velocidade, reduzir a resolução dos frames:
     ```python
     blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
     ```



## Referências

- [YOLO Darknet](https://pjreddie.com/darknet/yolo/)
- [YOLO Tiny](https://pjreddie.com/darknet/yolo/)
- [SSD MobileNet](https://github.com/chuanqi305/MobileNet-SSD)
- [OpenCV DNN](https://docs.opencv.org/master/d6/d0f/group__dnn.html)





