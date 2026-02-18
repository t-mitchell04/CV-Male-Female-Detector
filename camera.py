import cv2
import torch
from model import GenderClassifer
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F

CLASS_NAMES = {0: "Female", 1: "Male"}

def draw_label_box(frame, x1, y1, x2, y2, label, conf):
    text = f"{label} ({conf:.2f})"

    # box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # label background
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    y_text = max(0, y1 - th - 8)
    cv2.rectangle(frame, (x1, y_text), (x1 + tw + 6, y1), (0, 255, 0), -1)

    # label text
    cv2.putText(frame, text, (x1 + 3, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

def clamp(v, lo, hi):
    return max(lo, min(v, hi))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Load your classifier ----
model = GenderClassifer(num_classes=2).to(device)
ckpt = torch.load("model.pth", map_location=device)
model.load_state_dict(ckpt["model_state"])
model.eval()

# ---- Same preprocessing you trained with (plus normalize) ----
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# ---- Face detector ----
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
if face_cascade.empty():
    raise RuntimeError("Could not load Haar cascade for face detection.")

# ---- Webcam ----
vc = cv2.VideoCapture(0)
if not vc.isOpened():
    raise RuntimeError("Could not open webcam.")

while True:
    ok, frame = vc.read()
    if not ok:
        break

    h, w = frame.shape[:2]

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    # For each face, classify gender
    for (x, y, fw, fh) in faces:
        # add a little padding around face
        pad = int(0.15 * max(fw, fh))
        x1 = clamp(x - pad, 0, w - 1)
        y1 = clamp(y - pad, 0, h - 1)
        x2 = clamp(x + fw + pad, 0, w - 1)
        y2 = clamp(y + fh + pad, 0, h - 1)

        face_bgr = frame[y1:y2, x1:x2]
        if face_bgr.size == 0:
            continue

        # BGR -> RGB -> PIL
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        face_img = Image.fromarray(face_rgb)

        input_tensor = transform(face_img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(input_tensor)          # [1,2]
            probs = F.softmax(logits, dim=1)[0]   # [2]
            conf, idx = torch.max(probs, dim=0)

        label = CLASS_NAMES[int(idx.item())]
        conf = float(conf.item())

        draw_label_box(frame, x1, y1, x2, y2, label, conf)

    cv2.imshow("Boy or Girl (Face Box)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vc.release()
cv2.destroyAllWindows()