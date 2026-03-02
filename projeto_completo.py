from ultralytics import YOLO

def escolher_fonte():
    print("Escolha a fonte de entrada:")
    print("1 - Webcam")
    print("2 - Arquivo de imagem (.jpg ou .png)")
    
    opcao = input("Digite 1 ou 2: ").strip()
    
    if opcao == "1":
        return 0  # webcam
    elif opcao == "2":
        caminho = input("Digite o nome do arquivo (ex: foto.jpg): ").strip()
        return caminho
    else:
        print("Opção inválida.")
        return None

def executar_yolo():
    model = YOLO("yolov8n.pt")
    
    fonte = escolher_fonte()
    
    if fonte is None:
        return
    
    results = model.predict(source=fonte, show=True)
    
    # Mostrar no terminal o que foi detectado
    for r in results:
        for box in r.boxes:
            classe = model.names[int(box.cls)]
            confianca = float(box.conf)
            print(f"Objeto: {classe} | Confiança: {confianca:.2f}")


if __name__ == "__main__":
    executar_yolo()