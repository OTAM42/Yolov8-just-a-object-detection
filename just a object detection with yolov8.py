import cv2
from ultralytics import YOLO

# Modelin yüklenmesi
model_path = "best.pt"  # Modelin yolunu buraya girin
model = YOLO(model_path)

# Görüntünün yüklenmesi
image_path = "image.jpg"  # Görüntünün yolunu buraya girin
image = cv2.imread(image_path)

# Modelin görüntü üzerinde çalıştırılması
results = model(image)

# Sonuçların işlenmesi
for result in results:
    boxes = result.boxes  # Bounding boxes
    probs = result.probs  # Class probabilities

    # Sonuçların görüntüye çizilmesi
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy.int().tolist()[0]
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

# Sonuç görüntünün gösterilmesi
cv2.imshow("YOLOv8 Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()