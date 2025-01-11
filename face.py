import cv2
import os

def detectar_objetos():
    # Inicializa a câmera
    print("Iniciando câmera...")
    cap = cv2.VideoCapture(0)  

    if not cap.isOpened():
        print("Erro: Não foi possível abrir a câmera")
        return

    # Define a resolução (opcional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Largura
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Altura

    # Diretório para salvar imagens
    output_dir = "captured_images"
    os.makedirs(output_dir, exist_ok=True)

    # Carrega o modelo Haar Cascade para detecção de rostos
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Cria uma janela para exibir o feed
    cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)

    image_counter = 0

    try:
        while True:
            # Captura frame
            ret, frame = cap.read()
            if not ret:
                print("Erro ao capturar frame")
                break

            # Converte para tons de cinza
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detecção de rostos
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Desenha retângulos ao redor dos rostos
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Mostra o frame na janela
            cv2.imshow('Camera', frame)

            # Lê a tecla pressionada
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):  # Salvar imagem ao pressionar 's'
                image_path = os.path.join(output_dir, f"image_{image_counter:03d}.jpg")
                cv2.imwrite(image_path, frame)
                print(f"Imagem salva: {image_path}")
                image_counter += 1

            elif key == ord('q'):  # Sair ao pressionar 'q'
                print("Encerrando programa...")
                break

    finally:
        # Garante que a limpeza seja feita corretamente
        print("Liberando recursos...")
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detectar_objetos()
