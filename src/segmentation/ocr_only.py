import argparse
import json
import os
from pathlib import Path

import pandas as pd
import torch
from dotenv import load_dotenv
from paddleocr import PaddleOCR
from tqdm import tqdm

load_dotenv()

data_dir = os.getenv("DATA_DIR")
results_dir = os.getenv("RESULTS_DIR")


def get_segmentation_mask_from_ocr(bboxes, height, width):
    # create an empty mask
    mask = torch.zeros((height, width), dtype=torch.int16)
    for i, bbox in enumerate(bboxes, start=1):
        # bbox is a list like : [xtop, ytop, xbottom, ybottom]
        x_min, y_min, x_max, y_max = bbox
        mask[y_min:y_max, x_min:x_max] = i  # start labels from 1
    return mask


def main(path_to_data_info: Path):
    # init the results
    path_to_result = Path(results_dir) / "segmentation" / "ocr_only"
    path_to_result.mkdir(parents=True, exist_ok=True)
    # create a json file to save the config
    config = {
        "ocr_only": True,
        "model": "PaddleOCR",
    }
    with open(path_to_result / "config.json", "w") as f:
        json.dump(config, f, indent=4)

    # load data info
    df = pd.read_json(path_to_data_info, lines=True)

    # load the model
    ocr = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False)
    print("PaddleOCR model loaded successfully.")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
        image_hash = row["hash"]
        img_path = Path(data_dir) / "imgs" / f"{row['hash']}.jpg"
        results = ocr.predict(img_path)
        bboxes = results[0]["rec_boxes"]
        img = results[0]["doc_preprocessor_res"]["output_img"]
        h, w, c = img.shape

        mask = get_segmentation_mask_from_ocr(bboxes, h, w)

        # save the segmentation mask
        path_to_data_result = path_to_result / image_hash
        path_to_data_result.mkdir(parents=True, exist_ok=True)
        torch.save(mask, path_to_data_result / "segmentation_mask.pt")


if __name__ == "__main__":
    # argument parser to run the script from command line
    parser = argparse.ArgumentParser(
        description="Run OCR-only segmentation on the dataset."
    )
    parser.add_argument(
        "--data_info",
        type=str,
        default=str(Path(data_dir) / "dataset_info.jsonl"),
        help="Path to the dataset info jsonl file.",
    )
    args = parser.parse_args()
    main(path_to_data_info=Path(args.data_info))
