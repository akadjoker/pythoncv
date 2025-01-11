import cv2
import numpy as np

# Carregar as classes do COCO
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
    

# Configurar cores para as classes
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Carregar o modelo Tiny YOLO
net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")

# Identificar as camadas de saída
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Inicializar a câmera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

    # Pré-processar a imagem
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    # Passar pela rede
    outs = net.forward(output_layers)

    # Processar as saídas
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Filtrar por confiança mínima
                # Coordenadas do objeto detectado
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Supressão Não Máxima (NMS)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {int(confidence * 100)}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


    cv2.imshow("Tiny YOLO", frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg
# wget https://pjreddie.com/media/files/yolov3-tiny.weights
# wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
