from pathlib import Path

import face_recognition

import argparse

import pickle

from collections import Counter

from PIL import Image, ImageDraw, ImageFont

DEFAULT_FACE_DATA_PATH = Path("output/encodings.pkl")
FACE_BOX_COLOR = "green"
LABEL_COLOR = "white"

arg_parser = argparse.ArgumentParser(description="Facial Recognition Project")
arg_parser.add_argument("--train", action="store_true", help="Train on input data")
arg_parser.add_argument("--validate", action="store_true", help="Validate trained model")
arg_parser.add_argument("--test", action="store_true", help="Test the model with an unknown image")
arg_parser.add_argument("-m", action="store", default="hog", choices=["hog", "cnn"], help="Which model to use for training: hog (CPU), cnn (GPU)")
arg_parser.add_argument("-f", action="store", help="Path to an image with an unknown face")
args = arg_parser.parse_args()

Path("training_data").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)
Path("validation_data").mkdir(exist_ok=True)

def process_known_faces(model: str = "hog", face_data_location: Path = DEFAULT_FACE_DATA_PATH) -> None:
    """
    Processes the known faces for training.

    Args:
        model (str): Model to use for training.
        face_data_location (Path): Path to store the face data.

    Returns:
        None
    """
    individual_names = []
    face_encodings = []
    for file_path in Path("training_data").glob("*/*"):
        individual_name = file_path.parent.name
        img = face_recognition.load_image_file(file_path)

        detected_faces = face_recognition.face_locations(img, model=model)
        encoded_faces = face_recognition.face_encodings(img, detected_faces)

        for encoding in encoded_faces:
            individual_names.append(individual_name)
            face_encodings.append(encoding)

    names_and_encodings = {"names": individual_names, "encodings": face_encodings}
    with face_data_location.open(mode="wb") as f:
        pickle.dump(names_and_encodings, f)

def _identify_face(unknown_encoding, stored_encodings):
    """
    Identifies the face from the unknown encoding using the stored encodings.

    Args:
        unknown_encoding: Encoding of the unknown face.
        stored_encodings: Encodings of the known faces.

    Returns:
        str: Name of the identified face.
    """
    matches = face_recognition.compare_faces(stored_encodings["encodings"], unknown_encoding)
    votes = Counter(name for match, name in zip(matches, stored_encodings["names"]) if match)
    if votes:
        return votes.most_common(1)[0][0]

def _highlight_face(draw, face_coordinates, name, font):
    """
    Highlights the detected face with a bounding box and label.

    Args:
        draw: ImageDraw object to draw on the image.
        face_coordinates: Coordinates of the detected face.
        name: Name of the identified face.
        font: Font object for the label.

    Returns:
        None
    """
    top, right, bottom, left = face_coordinates
    draw.rectangle(((left, top), (right, bottom)), outline=FACE_BOX_COLOR)
    text_left, text_top, text_right, text_bottom = draw.textbbox((left, bottom), name, font=font)
    draw.rectangle(((text_left - 5, text_top - 5), (text_right + 5, text_bottom + 5)), fill="green", outline="green")
    draw.text((text_left, text_top), name, fill="white", font=font)

def detect_faces(image_path: str, model: str = "hog", face_data_location: Path = DEFAULT_FACE_DATA_PATH) -> None:
    """
    Detects faces in an image and displays the result.

    Args:
        image_path (str): Path to the image with unknown faces.
        model (str): Model to use for face detection.
        face_data_location (Path): Path to the stored face data.

    Returns:
        None
    """
    fnt = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 16)

    with face_data_location.open(mode="rb") as f:
        stored_encodings = pickle.load(f)

    unknown_img = face_recognition.load_image_file(image_path)

    unknown_face_locations = face_recognition.face_locations(unknown_img, model=model)

    unknown_face_encodings = face_recognition.face_encodings(unknown_img, unknown_face_locations)

    converted_image = Image.fromarray(unknown_img)
    draw_tool = ImageDraw.Draw(converted_image)

    for face_coordinates, unknown_encoding in zip(unknown_face_locations, unknown_face_encodings):
        name = _identify_face(unknown_encoding, stored_encodings)
        if not name:
            name = "Unknown"
        _highlight_face(draw_tool, face_coordinates, name, fnt)

    del draw_tool
    converted_image.show()

def check_model_performance(model: str = "hog"):
    """
    Checks the performance of the face recognition model on validation images.

    Args:
        model (str): Model to use for face detection.

    Returns:
        None
    """
    for file_path in Path("validation_data").rglob("*"):
        if file_path.is_file():
            detect_faces(image_path=str(file_path.absolute()), model=model)

if __name__ == "__main__":
    if args.train:
        process_known_faces(model=args.m)
    if args.validate:
        check_model_performance(model=args.m)
    if args.test:
        detect_faces(image_path=args.f, model=args.m)
DEFAULT_FACE_DATA_PATH = Path("output/encodings.pkl")
