from pathlib import Path
import tensorflow as tf

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "smoking" / "detect.tflite"
SMOKING_DIR = ROOT / "models" / "smoking"

print("Model exists:", MODEL_PATH.exists())
print("Smoking folder:", SMOKING_DIR)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH))
interpreter.allocate_tensors()

print("\nMODEL LOADED SUCCESSFULLY")

print("\nINPUT DETAILS:")
for item in interpreter.get_input_details():
    print("Shape:", item["shape"])
    print("Type:", item["dtype"])

print("\nOUTPUT DETAILS:")
for item in interpreter.get_output_details():
    print("Name:", item["name"], "| Shape:", item["shape"])

print("\nFILES IN SMOKING FOLDER:")
for path in SMOKING_DIR.rglob("*"):
    if path.is_file():
        print(path.name)
        if "label" in path.name.lower() or path.suffix.lower() == ".txt":
            print(path.read_text(encoding="utf-8", errors="replace"))

print("\nINSPECTION COMPLETE")