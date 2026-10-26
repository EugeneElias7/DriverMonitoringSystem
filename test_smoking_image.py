from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "smoking" / "detect.tflite"
IMAGE_PATH = ROOT / "test_image.jpg"

interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH))
interpreter.allocate_tensors()

input_detail = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()

image = cv2.imread(str(IMAGE_PATH))
if image is None:
    raise FileNotFoundError(f"Cannot read image: {IMAGE_PATH}")

_, height, width, _ = input_detail["shape"]
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (width, height))
input_data = np.expand_dims(image, axis=0).astype(input_detail["dtype"])

interpreter.set_tensor(input_detail["index"], input_data)
interpreter.invoke()

outputs = {
    detail["name"]: interpreter.get_tensor(detail["index"])
    for detail in output_details
}

print("\nRAW MODEL OUTPUTS")
for name, values in outputs.items():
    print(f"\n{name}:\n{values}")

print("\nLabel file:")
label_path = ROOT / "models" / "smoking" / "labelmap.txt"
print(label_path.read_text(encoding="utf-8"))

print("\nInspection complete.")
print("We must verify tensor meanings and preprocessing before interpreting")
print("these values as confirmed detections.")