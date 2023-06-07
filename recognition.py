from pathlib import Path

import face_recognition

DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")

Path("training_data").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)
Path("validation_data").mkdir(exist_ok=True)

def encode_known_faces(
    model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH
) -> None:
    names = []
    encodings = []
    for filepath in Path("training").glob("*/*"):
        name = filepath.parent.name
        image = face_recognition.load_image_file(filepath)