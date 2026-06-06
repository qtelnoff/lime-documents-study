# lime-documents-study
This repository contains the research code for the paper: When Segmentation Shapes Explanation: A Data-Driven Study of LIME Reliability in Administrative Document Classification.

# How to run experiments ?
## 0 - Requierement
1 - install the requierements.txt
2 - create .env files with :
DATA_DIR path
RESULTS_DIR path

## 1 - Run segmentations scripts 
Here are the bash commands to run each of the segmentation scripts. You can run them sequentially or pick the ones you need:

```bash
# Run Quickshift segmentation
python src/segmentation/quickshift.py

# Run SLIC segmentation
python src/segmentation/slic.py

# Run OCR-only segmentation
python src/segmentation/ocr_only.py

# Run OCR with background grid segmentation (default 4x4)
python src/segmentation/ocr_grid.py --num_rows 4 --num_cols 4

# Run Grid without OCR bounding boxes (default 4x4)
python src/segmentation/rectangle_without_bboxes.py --num_rows 4 --num_cols 4
```

*Note: All scripts default to looking for the dataset info at the path specified by your `DATA_DIR` environment variable + `/dataset_info.jsonl`. If you need to specify a custom path, you can append `--data_info /path/to/your/custom_info.jsonl` to any of the commands.*

## 2 - Run inference script

