import argparse
import os
from pathlib import Path

import torch
from dotenv import load_dotenv
from torchvision.io import read_image

from lime.lime_utils import PerturbationDataset, inference_on_perturbation_dataset
from utils import load_model_and_preprocess

load_dotenv()

results_dir = Path(os.getenv("RESULTS_DIR"))  # type: ignore
data_dir = Path(os.getenv("DATA_DIR"))  # type: ignore

def main(
    dataset_name: str,
    model_to_be_explained: str,
    random_walk: str,
    color: str,
    path_to_image: str,
    image_hash: str,
    segmentation_algorithm: str,
    sample_number: int,
) -> None:

    # setup model and preprocess
    model, transform = load_model_and_preprocess(model_to_be_explained, dataset_name)

    # load the image
    image = read_image(path_to_image)  # normalize to [0, 1]

    # load segmented image
    path_to_segmented_image = (
        Path(results_dir)
        / "segmentations"
        / dataset_name
        / segmentation_algorithm
        / image_hash
        / "segmentation_mask.pt"
    )
    segmented_image = torch.load(path_to_segmented_image).to(torch.int16)
    segmented_image = segmented_image - 1
    background_not_used = (segmented_image < 0).any()

    # create perturbation dataset
    perturbation_dataset = PerturbationDataset(
        image=image,
        segmented_image=segmented_image,
        num_samples=sample_number,
        random_walk=random_walk,
        transform=transform,
        color=color,
        background_not_used=background_not_used,
    )

    # create dataloader
    perturbation_dataloader = torch.utils.data.DataLoader(
        perturbation_dataset, batch_size=32, shuffle=False, num_workers=4
    )

    # run inference on the perturbed images
    interpretable_inputs, logits, probas = inference_on_perturbation_dataset(
        model=model,
        perturbation_dataloader=perturbation_dataloader,
        background_not_used=background_not_used,
    )

    # save the outputs
    path_to_save_directory = (
        Path(results_dir)
        / "inferences"
        / dataset_name
        / model_to_be_explained
        / segmentation_algorithm
        / color
        / random_walk
        / image_hash
    )
    # ensure directory exists
    os.makedirs(path_to_save_directory, exist_ok=True)
    # then save
    torch.save(
        logits,
        path_to_save_directory / "logits.pt",
    )
    torch.save(
        interpretable_inputs,
        path_to_save_directory / "interpretable_inputs.pt",
    )
    torch.save(
        probas,
        path_to_save_directory / "probas.pt",
    )


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Save neighbor outputs for LIME explanations."
    )
    # dataset name
    parser.add_argument(
        "--dataset_name", type=str, default="rvlcdip", help="Name of the dataset."
    )
    # model to be explained
    parser.add_argument(
        "--model_to_be_explained",
        type=str,
        default="resnet50",
        help="Model to be explained. Example: 'resnet50'",
    )
    # random walk
    parser.add_argument(
        "--random_walk",
        type=str,
        default="uniform",
        help="Random walk strategy to use. Example: 'uniform', 'exponential', 'gaussian'",
    )
    # path to image
    parser.add_argument(
        "--path_to_image",
        type=str,
        default="/common/datasets/rvlcdip/Image/Test_Data/invoice/96054429.jpg",
        help="Path to the input image.",
    )
    # hash of the image
    parser.add_argument(
        "--image_hash",
        type=str,
        default="7878ff163caf34987c4d036b5bf3878e442112b432081d65599bd57cc3226d47",
        help="Hash of the input image.",
    )
    # segmentation algorithm
    parser.add_argument(
        "--segmentation_algorithm",
        type=str,
        default="paddle_ocr",
        help="Segmentation algorithm to use. Example: 'quickshift', rectangle_4x4, rectangle_8x8, rectangle_16x16",
    )
    # color
    parser.add_argument(
        "--color",
        type=str,
        default="white",
        help="Color to use for perturbation. Example: 'black', 'white', 'mean'",
    )

    # sample number
    parser.add_argument(
        "--sample_number", type=int, default=10000, help="Sample number for the image."
    )

    args = parser.parse_args()

    main(
        dataset_name=args.dataset_name,
        model_to_be_explained=args.model_to_be_explained,
        random_walk=args.random_walk,
        color=args.color,
        path_to_image=args.path_to_image,
        image_hash=args.image_hash,
        segmentation_algorithm=args.segmentation_algorithm,
        sample_number=args.sample_number,
    )
