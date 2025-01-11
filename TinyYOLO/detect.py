import cv2
import numpy as np


net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

def verificar_objetos(frame):
    height, width, _ = frame.shape

    # Pré-processamento
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    #  resultados
    outs = net.forward(net.getUnconnectedOutLayersNames())
    
 
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                return True
    return False

cap = cv2.VideoCapture(0) # ls /dev/video*
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if verificar_objetos(frame):
        print("Objeto detectado!")
    else:
        print("Nenhum objeto detectado.")

    # para mostrar Frame, no carro nao precisamos
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
