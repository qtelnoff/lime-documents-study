import hashlib
import json
from pathlib import Path
import os

import pandas as pd
import torch
import torchvision.transforms.v2 as v2
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR")) # type: ignore
RESULTS_DIR = Path(os.getenv("RESULTS_DIR")) # type: ignore

def hash_generation(liste_valeurs):
    # 1. Convertir tous les éléments en string (y compris None et 100)
    # 2. Les joindre avec un séparateur unique (ex: "|") pour éviter les collisions
    chaine_concatenee = "|".join(map(str, liste_valeurs))

    # 3. Encoder en bytes (utf-8)
    bytes_data = chaine_concatenee.encode("utf-8")

    # 4. Créer le hash (MD5 est souvent suffisant pour un nom court, SHA256 est plus sûr)
    # Utilisons MD5 pour avoir un nom de taille raisonnable (32 caractères)
    hash_obj = hashlib.md5(bytes_data)

    return hash_obj.hexdigest()


def load_label2id(dataset_name: str, model_name: str):
    path_to_id2label = (
        RESULTS_DIR / "inferences" / dataset_name / model_name / "models_to_be_explained" / "id2label.json"
    )
    with open(path_to_id2label, "r") as f:
        id2label = json.load(f)
    label2id = {v: int(k) for k, v in id2label.items()}
    return label2id


def load_model_and_preprocess(model_name: str, dataset_name: str):
    # load the best model checkpoint
    checkpoint_path = (
        RESULTS_DIR / "inferences" / dataset_name / model_name / "models_to_be_explained" /"parameters.pth"
    )

    checkpoint = torch.load(
        checkpoint_path, map_location=torch.device("cpu")
    )  # ordered dict

    if model_name == "resnet50":
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)

        # Replace the final fully connected layer
        num_classes = checkpoint["fc.weight"].shape[0]
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

        model.load_state_dict(checkpoint)

        transform = v2.Compose(
            [
                v2.Resize(
                    (
                        weights.transforms().crop_size[0],
                        weights.transforms().crop_size[-1],
                    )
                ),
                v2.Normalize(
                    mean=weights.transforms().mean, std=weights.transforms().std
                ),
            ]
        )
    else:
        raise NotImplementedError(f"Model {model_name} not implemented.")
    return model, transform


def load_image_for_inference(image_path: Path, preprocess):
    image = Image.open(image_path).convert("RGB")
    image = preprocess(image)
    return image


def load_data_info(dataset_name: str, split: str):
    path_to_data_info = (
        DATA_DIR / "data" / dataset_name / split / "dataset_info.jsonl"
    )
    return pd.read_json(path_to_data_info, lines=True)


if __name__ == "__main__":
    model, preprocess = load_model_and_preprocess("resnet50", "rvlcdip")
    print(model)
