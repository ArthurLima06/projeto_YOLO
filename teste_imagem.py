from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

results = model("foto.jpg")

for r in results:
    img_com_caixas = r.plot()

    cv2.imshow("Resultado Yolo", img_com_caixas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    