import argparse
import os
from itertools import product
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from torchvision.io import read_image

from lime.lime_utils import explanation, get_norm_fn, load_data_for_training
from utils import load_model_and_preprocess

load_dotenv()
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))  # type: ignore
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "results"))  # type: ignore

PARAMETERS_GRID = {
    "segmentation_algorithm": [
        "quickshift",  #
        "slic",  #
        "grid_4x4_without_bboxes",  #
        "paddle_ocr",  #
        "paddle_ocr_background",  #
        "paddle_ocr_4x4_new",  #
        "paddle_ocr_10x10_new",  #
        "paddle_ocr_20x10_new",  #
        "paddle_ocr_30x10_new",  #
    ],
    "color": ["black", "white", "mean"],
    "output_type": ["probas", "logits"],
    "norm": ["euclidean_interpretable", "cosine_interpretable"],
    "kernel_width": [0.1, 0.3, 0.7, 1.0],
    "n_neighbors": [
        100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 
        2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000,
    ],
}

NB_CONFIGURATIONS = (
    len(PARAMETERS_GRID["segmentation_algorithm"]) *
    len(PARAMETERS_GRID["color"]) *
    len(PARAMETERS_GRID["output_type"]) *
    len(PARAMETERS_GRID["norm"]) *
    len(PARAMETERS_GRID["kernel_width"]) *
    len(PARAMETERS_GRID["n_neighbors"])
)
# --- Global Constants ---
DATASET_NAME = "rvlcdip"
MODEL_TO_BE_EXPLAINED = "resnet50"
SPLIT = "reduced_version"
RANDOM_WALK = "uniform"
INTERPRETABLE_MODEL = "linear"
ALPHA = None


def perturb_specific_segments(
    image: torch.Tensor,
    segmented_image: torch.Tensor,
    indices_to_perturb: list[int],
    color: str = "black",
    background_not_used: bool = False,
    transform: Optional[Callable] = None,
) -> torch.Tensor:
    """
    Perturbe des segments spécifiques d'une image.
    
    Args:
        image: Tensor de forme (C, H, W).
        segmented_image: Tensor de forme (H, W) contenant les IDs des superpixels.
        indices_to_perturb: Liste des IDs de superpixels à perturber (masquer).
        color: Couleur de perturbation ('median', 'mean', 'black', 'white').
        background_not_used: Si True, ajoute un 1 à la fin du vecteur de masque pour ignorer le fond.
        transform: Fonction de transformation optionnelle.
        
    Returns:
        L'image perturbée sous forme de Tensor.
    """
    # 1. S'assurer que l'image est float [0, 1]
    if image.dtype == torch.uint8:
        image = image.float() / 255.0
        
    segmented_image = segmented_image.long()
    num_segments: int = int(segmented_image.max().item() + 1)
    
    num_channels = image.shape[0]

    # 2. Définir la couleur de perturbation
    if color == "median":
        # Flatten H and W to properly calculate the spatial median per channel
        pixel_color = image.view(num_channels, -1).median(dim=1)[0].view(-1, 1, 1)
    elif color == "mean":
        pixel_color = image.mean(dim=(1, 2), keepdim=True)
    elif color == "black":
        pixel_color = torch.zeros((num_channels, 1, 1), device=image.device)
    elif color == "white":
        pixel_color = torch.ones((num_channels, 1, 1), device=image.device)
    else:
        raise ValueError(
            f"Perturbation color '{color}' not recognized. Choose from 'median', 'mean', 'black', or 'white'."
        )
        
    pixel_color = pixel_color.to(image.device)
    
    # 3. Créer le vecteur de masque (1 pour garder, 0 pour perturber)
    mask_vector = torch.ones((num_segments,), dtype=torch.float32, device=image.device)
    for idx in indices_to_perturb:
        if idx < num_segments:
            mask_vector[idx] = 0.0
            
    if background_not_used:
        # Ajouter 1 à la fin du mask_vector pour éviter l'indexation négative
        mask_vector = torch.cat(
            (mask_vector, torch.tensor([1.0], device=image.device))
        )
        
    # 4. Créer le masque spatial
    # Projeter mask_vector en 2D (H, W), puis ajouter la dimension (1, H, W) pour le broadcasting
    spatial_mask = mask_vector[segmented_image].unsqueeze(0)
    
    # 5. Appliquer la perturbation
    perturbed_image = image * spatial_mask + pixel_color * (1 - spatial_mask)

    # 6. Appliquer la transformation de prétraitement si nécessaire
    if transform:
        perturbed_image = transform(perturbed_image)
    
    return perturbed_image


def main(args):
    id = args.image_id - 1
    # load the dataset info to get the image path and label
    path_to_data = (
        DATA_DIR / "dataset_info.jsonl"
    )
    df = pd.read_json(path_to_data, lines=True)
    image_info = df.iloc[id]

    path_to_image = DATA_DIR / "imgs" / f"{image_info['hash']}.jpg"
    label_id = image_info["class_id"]
    image_hash = image_info["hash"]

    # Setup model and preprocess
    model, transform = load_model_and_preprocess(MODEL_TO_BE_EXPLAINED, DATASET_NAME)

    # Load the image
    image = read_image(path_to_image)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    image = image.to(device)

    # Container to save metrics so they aren't lost
    evaluation_results = []

    count = 0

    for algo in PARAMETERS_GRID["segmentation_algorithm"]:
        # Load segmented image
        path_to_segmented_image = (
            data_dir
            / "segmentations"
            / DATASET_NAME
            / algo
            / image_hash
            / "segmentation_mask.pt"
        )
        segmented_image = torch.load(path_to_segmented_image).to(torch.int16).to(device)
        segmented_image = segmented_image - 1
        background_not_used = (segmented_image < 0).any()

        for color, output_type in product(
            PARAMETERS_GRID["color"], PARAMETERS_GRID["output_type"]
        ):
            interpretable_inputs, outputs = load_data_for_training(
                dataset=DATASET_NAME,
                model_to_be_explained=MODEL_TO_BE_EXPLAINED,
                segmentation_algorithm=algo,
                color=color,
                hash=image_hash,
                which_walk=RANDOM_WALK,
                which_output=output_type,
            )
            
            # switch to numpy
            interpretable_inputs = interpretable_inputs.numpy()
            outputs = outputs.numpy()
            n_samples = interpretable_inputs.shape[0]

            for norm in PARAMETERS_GRID["norm"]:
                norm_fn = get_norm_fn(norm)
                for kernel_width in PARAMETERS_GRID["kernel_width"]:
                    norms = norm_fn(interpretable_inputs, segmented_image, path_to_image)
                    similarities = np.exp(-1 * (norms**2) / (kernel_width**2))
                    
                    for n_neighbors in PARAMETERS_GRID["n_neighbors"]:
                        # Prevent crash if n_neighbors > n_samples available
                        safe_n_neighbors = min(n_neighbors, n_samples)
                        indices = np.random.choice(n_samples, size=safe_n_neighbors, replace=False)

                        explanation_weights, r2 = explanation(
                            model=INTERPRETABLE_MODEL,
                            X=interpretable_inputs[indices],
                            y=outputs[indices, label_id],
                            weights=similarities[indices],
                        )
                        
                        # compute the insertion metric
                        # sort the segments less to most important according to the explanation weights
                        idx_sorted = np.argsort(explanation_weights)[::-1].tolist()
                        unique_vals, counts = torch.unique(segmented_image, return_counts=True)
                        segment_sizes = dict(zip(unique_vals.tolist(), counts.tolist()))
                        total_pixels = segmented_image.numel()

                        
                        x_list = [0.0]
                        current_inserted_pixels = 0
                        
                        for idx in idx_sorted:
                            current_inserted_pixels += segment_sizes.get(idx, 0)
                            x_list.append(current_inserted_pixels / total_pixels)
                        
                        # Generates the modified images
                        perturbed_images_list = []
                        for i in range(len(idx_sorted) + 1):
                            indices_to_perturb = idx_sorted[i:]
                            perturbed_image = perturb_specific_segments(
                                image=image,
                                segmented_image=segmented_image,
                                indices_to_perturb=indices_to_perturb,
                                color=color,
                                background_not_used=background_not_used,
                                transform=transform
                            )
                            perturbed_images_list.append(perturbed_image)
                        
                        # Batched inference
                        batch_size = 32 # Adjust based on your GPU memory
                        p_list = []
                        model.eval()
                        with torch.no_grad():
                            for i in range(0, len(perturbed_images_list), batch_size):
                                batch = torch.stack(perturbed_images_list[i : i + batch_size]).to(device)
                                probas = torch.nn.functional.softmax(model(batch), dim=1)
                                p_list.extend(probas[:, label_id].cpu().tolist())

                        # --- Save metrics instead of discarding them ---
                        evaluation_results.append({
                            "image_id": args.image_id,
                            "algo": algo,
                            "color": color,
                            "output_type": output_type,
                            "norm": norm,
                            "kernel_width": kernel_width,
                            "n_neighbors": safe_n_neighbors,
                            "x_list": x_list,
                            "p_list": p_list,
                            "r2_score": r2
                        })
                        count += 1
                        if count % 10 == 0:
                            print(f"Completed {count}/{NB_CONFIGURATIONS} configurations...")
                            
    # Optionally save results to disk at the end
    results_df = pd.DataFrame(evaluation_results)
    results_file = Path(data_dir) / "results" / "icdm" / f"insertion_image_{args.image_id}.parquet"
    results_df.to_parquet(results_file, index=False)
    print(f"\nEvaluation complete. Results saved to: {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the correctness of explanations on GPU."
    )
    parser.add_argument(
        "--image_id", type=int, default=0, help="ID of the image to evaluate"
    )
    args = parser.parse_args()
    main(args)