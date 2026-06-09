import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from tqdm import tqdm
from dotenv import load_dotenv

from lime.lime_utils import explanation, get_norm_fn, load_data_for_training
from utils import hash_generation, load_label2id

load_dotenv()
DATA_DIR = Path(os.getenv("DATA_DIR", "data")) # type: ignore
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "results")) # type: ignore

def spearman_consistency(explanations):
    spearman_corrs = []
    num_explanations = len(explanations)
    ij = [
        (i, j) for i in range(num_explanations) for j in range(i + 1, num_explanations)
    ]
    for i, j in ij:
        spearman_corr, _ = spearmanr(explanations[i], explanations[j])
        spearman_corrs.append(spearman_corr)
    return np.array(spearman_corrs)


def jaccard_index(explanations, top_k: int = 5):
    jaccard_indices = []
    num_explanations = len(explanations)
    ij = [
        (i, j) for i in range(num_explanations) for j in range(i + 1, num_explanations)
    ]
    for i, j in ij:
        top_k_i = set(np.argsort(explanations[i])[-top_k:])
        top_k_j = set(np.argsort(explanations[j])[-top_k:])
        intersection = len(top_k_i.intersection(top_k_j))
        union = len(top_k_i.union(top_k_j))
        jaccard_index = intersection / union if union != 0 else 0
        jaccard_indices.append(jaccard_index)
    return np.array(jaccard_indices)


def main(args):
    print("=== Consistency job started ===", flush=True)
    print("Args:", args, flush=True)

    run = vars(args)
    run_hash = hash_generation(list(run.values()))
    print(f"Run hash: {run_hash}", flush=True)

    print("Loading precomputed data...", flush=True)
    interpretable_inputs, outputs = load_data_for_training(
        dataset=args.dataset_name,
        model_to_be_explained=args.model_to_be_explained,
        segmentation_algorithm=args.segmentation_algorithm,
        color=args.color,
        hash=args.image_hash,
        which_walk=args.random_walk,
        which_output=args.output_type,
    )
    print(f"Loaded data: X={interpretable_inputs.shape}, y={outputs.shape}", flush=True)

    print("Loading segmentation mask...", flush=True)
    path_to_segmented_image = (
            RESULTS_DIR
            / "segmentations"
            / args.dataset_name
            / args.segmentation_algorithm
            / args.image_hash
            / "segmentation_mask.pt"
    )
    print(f"Segmentation path: {path_to_segmented_image}", flush=True)

    segmentation_mask = torch.load(path_to_segmented_image, map_location="cpu")
    print(f"Segmentation mask shape: {segmentation_mask.shape}", flush=True)

    print("Computing norms & similarities...", flush=True)

    # switch to numpy
    interpretable_inputs = interpretable_inputs.numpy()
    outputs = outputs.numpy()
    n_samples = interpretable_inputs.shape[0]
    print(f"interpretable_inputs dtype={interpretable_inputs.dtype}, size={interpretable_inputs.nbytes / 1e6:.1f} MB",
          flush=True)
    print(f"outputs dtype={outputs.dtype}, size={outputs.nbytes / 1e6:.1f} MB", flush=True)

    # similarity scores calculation
    # compute norms
    norm_fn = get_norm_fn(args.norm)
    norms = norm_fn(interpretable_inputs, segmentation_mask, args.path_to_image)
    similarities = np.exp(-1 * (norms**2) / (args.kernel_width**2))

    # choose the label
    label2id = load_label2id(
        model_name=args.model_to_be_explained, dataset_name=args.dataset_name
    )
    class_id = label2id[args.class_label]

    explanations = []
    r2_scores = []
    for i in tqdm(range(args.n_explanations), desc="Calculating n explanations"):
        indices = np.random.choice(n_samples, size=args.n_neighbors, replace=False)
        expl, r2 = explanation(
            model=args.interpretable_model,
            X=interpretable_inputs[indices],
            y=outputs[indices, class_id],
            weights=similarities[indices],
        )
        explanations.append(expl)
        r2_scores.append(r2)

        if i % 5 == 0:
            print(f"[{i}/{args.n_explanations}] r2 mean so far: {np.mean(r2_scores):.4f}", flush=True)

    # save the results in a dictionary

    results = {
        "mean_spearman_consistency": np.mean(spearman_consistency(explanations)),
        "std_spearman_consistency": np.std(spearman_consistency(explanations)),
        "mean_jaccard_index_top_5": np.mean(jaccard_index(explanations, top_k=5)),
        "std_jaccard_index_top_5": np.std(jaccard_index(explanations, top_k=5)),
        "mean_jaccard_index_top_10": np.mean(jaccard_index(explanations, top_k=10)),
        "std_jaccard_index_top_10": np.std(jaccard_index(explanations, top_k=10)),
        "mean_jaccard_index_top_20": np.mean(jaccard_index(explanations, top_k=20)),
        "std_jaccard_index_top_20": np.std(jaccard_index(explanations, top_k=20)),
        "mean_jaccard_index_top_50": np.mean(jaccard_index(explanations, top_k=50)),
        "std_jaccard_index_top_50": np.std(jaccard_index(explanations, top_k=50)),
        "mean_r2_score": np.mean(r2_scores),
        "std_r2_score": np.std(r2_scores),
    }

    run.update(results)

    output_path = RESULTS_DIR / "consistencies" / args.image_hash / args.segmentation_algorithm / f"{run_hash}.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving results to: {output_path}", flush=True)
    pd.DataFrame([run]).to_parquet(output_path)
    print("Save complete.", flush=True)


if __name__ == "__main__":
    # add argparse for command line arguments
    parser = argparse.ArgumentParser(
        description="Analyze variability of explanations with different numbers of neighbors."
    )
    parser.add_argument(
        "--dataset_name", type=str, default="rvlcdip", help="Name of the dataset."
    )
    parser.add_argument(
        "--model_to_be_explained",
        type=str,
        default="resnet50",
        help="Model name to be explained.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="reduced_version",
        help="Dataset split to use (reduced_version/full_version).",
    )
    parser.add_argument(
        "--path_to_image",
        type=str,
        default='/common/datasets/rvlcdip/Image/Test_Data/invoice/96054429.jpg',
        help="Path to the input image.",
    )
    parser.add_argument(
        "--class_label",
        type=str,
        default="invoice",
        help="Class label for which explanations are generated.",
    )
    parser.add_argument(
        "--image_hash",
        type=str,
        default="7878ff163caf34987c4d036b5bf3878e442112b432081d65599bd57cc3226d47",
    )
    # how many explanations to generate for each image (we will compute the consistency metrics on these explanations to analyze the variability)
    parser.add_argument(
        "--n_explanations",
        type=int,
        default=50,
        help="Number of explanations to generate for each image.",
    )

    # les paramètres à évaluer
    parser.add_argument(
        "--segmentation_algorithm",
        type=str,
        default="paddle_ocr",
        help="Segmentation method used for LIME explanations.",
    )
    parser.add_argument("--random_walk", type=str, default="uniform", help="uniforme")
    parser.add_argument(
        "--color", type=str, default="mean", help="Color used to perturb the segments."
    )
    parser.add_argument(
        "--output_type",
        type=str,
        default="logits",
        help="Type of model output to use (logits/probas).",
    )
    parser.add_argument(
        "--interpretable_model",
        type=str,
        default="linear",
        help="Type of interpretable model to use (linear/ridge).",
    )
    parser.add_argument(
        "--norm",
        type=str,
        default="euclidean_interpretable",
        help="Norm to use for weighting (none/l1/l2).",
    )
    parser.add_argument(
        "--kernel_width",
        type=float,
        default=0.1,
        help="Kernel width used for the segmentation.",
    )
    parser.add_argument(
        "--n_neighbors", type=int, default=1000, help="Number of neighbors to evaluate."
    )
    parser.add_argument(
        "--alpha",
        type=float,
        help="Regularization strength for Ridge and Lasso regression (if interpretable_model is ridge or lasso).",
    )
    args = parser.parse_args()

    try:
        main(args)
        print("=== Job finished successfully ===", flush=True)
    except Exception as e:
        import traceback

        print("=== JOB CRASHED ===", flush=True)
        traceback.print_exc()
        raise
