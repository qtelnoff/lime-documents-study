import json
from pathlib import Path
import os
import argparse

import torch
import pandas as pd
from skimage import segmentation
from skimage.io import imread

from dotenv import load_dotenv
load_dotenv()

data_dir = os.getenv("DATA_DIR")
results_dir = os.getenv("RESULTS_DIR")

def main(
    path_to_data_info: Path,
):
    # create a directory to save the results
    path_to_result = Path(results_dir) / "segmentation" / "slic"

    # init the results
    path_to_result.mkdir(parents=True, exist_ok=True)

    # create a json file to save the config
    config = {
        "n_segments": 100,  # number of superpixels
        "model": "slic",
    }
    with open(path_to_result / "config.json", "w") as f:
        json.dump(config, f, indent=4)

    # load data info
    df = pd.read_json(path_to_data_info, lines=True)

    for idx, row in df.iterrows():
        image_hash = row["hash"]
        path_to_data_result = path_to_result / image_hash
        path_to_data_result.mkdir(parents=True, exist_ok=True)
        img_path = Path(data_dir) / "imgs" / f"{row['hash']}.jpg"
        tmp_image = imread(img_path)
        # load segmentation method
        segments_slic = segmentation.slic(tmp_image, n_segments=config["n_segments"])
        torch.save(
            torch.tensor(segments_slic), path_to_data_result / "segmentation_mask.pt"
        )

    

if __name__ == "__main__":
    # argument parser to run the script from command line
    parser = argparse.ArgumentParser(description="Run SLIC segmentation on the dataset.")
    parser.add_argument(
        "--data_info",
        type=str,
        default=str(Path(data_dir) / "dataset_info.jsonl"),
        help="Path to the dataset info jsonl file.",
    )
    args = parser.parse_args()
    main(path_to_data_info=Path(args.data_info))
