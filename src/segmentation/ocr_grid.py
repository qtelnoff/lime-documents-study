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

def get_ocr_grid_segmentation_mask(mask_ocr, num_rows: int, num_cols: int):
    height, width = mask_ocr.shape
    # create an empty mask
    mask = torch.zeros((height, width), dtype=torch.int16)
    # add rectangle segmentation for background
    grid_height_q = height // num_rows
    grid_height_r = height % num_rows
    grid_height = [
        grid_height_q + 1 if i < grid_height_r else grid_height_q
        for i in range(num_rows)
    ]
    grid_height_cumsum = [0] + [sum(grid_height[: i + 1]) for i in range(num_rows)]

    grid_width_q = width // num_cols
    grid_width_r = width % num_cols
    grid_width = [
        grid_width_q + 1 if i < grid_width_r else grid_width_q for i in range(num_cols)
    ]
    grid_width_cumsum = [0] + [sum(grid_width[: i + 1]) for i in range(num_cols)]
    segment_number = 0
    for i in range(num_rows):
        for j in range(num_cols):
            segment_number = i * num_cols + j + 1  # background segments start from 1
            mask[
                grid_height_cumsum[i] : grid_height_cumsum[i + 1],
                grid_width_cumsum[j] : grid_width_cumsum[j + 1],
            ] = segment_number
    
    max_segment_value = mask_ocr.max().item()
    for i in range(1, max_segment_value + 1):
        selection = mask_ocr == i
        mask[selection] = i + segment_number 

    return mask

def get_segmentation_mask_from_ocr(bboxes, height, width):
    # create an empty mask
    mask = torch.zeros((height, width), dtype=torch.int16)
    for i, bbox in enumerate(bboxes, start=1):
        # bbox is a list like : [xtop, ytop, xbottom, ybottom]
        x_min, y_min, x_max, y_max = bbox
        mask[y_min:y_max, x_min:x_max] = i  # start labels from 1
    return mask


def main(path_to_data_info: Path, num_rows: int = 4, num_cols: int = 4):
    # init the results
    path_to_result = Path(results_dir) / "segmentation" / f"ocr_grid_{num_rows}x{num_cols}" #ignore ty
    path_to_result.mkdir(parents=True, exist_ok=True)
    # create a json file to save the config
    config = {
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

        mask_ocr = get_segmentation_mask_from_ocr(bboxes, h, w)

        # save the segmentation mask
        mask = get_ocr_grid_segmentation_mask(mask_ocr, num_rows=num_rows, num_cols=num_cols)

        # save the segmentation mask
        path_to_data_result = path_to_result / image_hash
        path_to_data_result.mkdir(parents=True, exist_ok=True)
        torch.save(mask, path_to_data_result / "segmentation_mask.pt")


if __name__ == "__main__":
    # argument parser to run the script from command line
    parser = argparse.ArgumentParser(
        description="Run OCR + grid segmentation on the dataset."
    )
    parser.add_argument(
        "--data_info",
        type=str,
        default=str(Path(data_dir) / "dataset_info.jsonl"),
        help="Path to the dataset info jsonl file.",
    )
    parser.add_argument(
        "--num_rows",
        type=int,
        default=4,
        help="Number of rows for the grid segmentation.",
    )
    parser.add_argument(
        "--num_cols",
        type=int,
        default=4,
        help="Number of columns for the grid segmentation.",
    )
    args = parser.parse_args()
    main(path_to_data_info=Path(args.data_info), num_rows=args.num_rows, num_cols=args.num_cols)
