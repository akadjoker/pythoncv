import cv2
import numpy as np
import json

def salvar_configuracoes(config, arquivo="config.json"):
    with open(arquivo, 'w') as f:
        json.dump(config, f)
    print(f"Configurações salvas em {arquivo}")

def carregar_configuracoes(arquivo="config.json"):
    try:
        with open(arquivo, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Arquivo {arquivo} não encontrado. Usando valores padrão.")
        return {
            'h_min': 0, 'h_max': 179,
            's_min': 0, 's_max': 255,
            'v_min': 0, 'v_max': 255
        }

def detectar_objetos():
    print("Iniciando câmera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Erro: Não foi possível abrir a câmera")
        return

    # Carrega configurações iniciais
    config = carregar_configuracoes()

    # Cria janela e trackbars para ajustar cores
    cv2.namedWindow('Controles')
    cv2.createTrackbar('H Minimo', 'Controles', config['h_min'], 179, lambda x: None)
    cv2.createTrackbar('H Maximo', 'Controles', config['h_max'], 179, lambda x: None)
    cv2.createTrackbar('S Minimo', 'Controles', config['s_min'], 255, lambda x: None)
    cv2.createTrackbar('S Maximo', 'Controles', config['s_max'], 255, lambda x: None)
    cv2.createTrackbar('V Minimo', 'Controles', config['v_min'], 255, lambda x: None)
    cv2.createTrackbar('V Maximo', 'Controles', config['v_max'], 255, lambda x: None)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Erro ao capturar frame")
                break
            
            # Converte para HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Pega valores dos trackbars
            h_min = cv2.getTrackbarPos('H Minimo', 'Controles')
            h_max = cv2.getTrackbarPos('H Maximo', 'Controles')
            s_min = cv2.getTrackbarPos('S Minimo', 'Controles')
            s_max = cv2.getTrackbarPos('S Maximo', 'Controles')
            v_min = cv2.getTrackbarPos('V Minimo', 'Controles')
            v_max = cv2.getTrackbarPos('V Maximo', 'Controles')
            
            # Define range da cor
            cor_min = np.array([h_min, s_min, v_min])
            cor_max = np.array([h_max, s_max, v_max])
            
            # Cria máscara
            mascara = cv2.inRange(hsv, cor_min, cor_max)
            
            # Encontra contornos
            contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Desenha retângulos e calcula propriedades
            for contorno in contornos:
                area = cv2.contourArea(contorno)
                if area > 1000:  # Filtra objetos pequenos
                    x, y, w, h = cv2.boundingRect(contorno)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Calcula o centro do objeto
                    centro_x = x + w // 2
                    centro_y = y + h // 2
                    cv2.circle(frame, (centro_x, centro_y), 5, (0, 0, 255), -1)
                    cv2.putText(frame, f"Area: {int(area)}", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(frame, f"Centro: ({centro_x}, {centro_y})", (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Mostra frames
            cv2.imshow('Original', frame)
            cv2.imshow('Mascara', mascara)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Encerrando programa...")
                break
            elif key == ord('s'):
                # Salva as configurações atuais
                config = {
                    'h_min': h_min, 'h_max': h_max,
                    's_min': s_min, 's_max': s_max,
                    'v_min': v_min, 'v_max': v_max
                }
                salvar_configuracoes(config)

    finally:
        print("Liberando recursos...")
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detectar_objetos()
