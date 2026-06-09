# lime-documents-study

This repository contains the research code for the paper: **When Segmentation Shapes Explanation: A Data-Driven Study of LIME Reliability in Administrative Document Classification**.

## Project Structure

```
lime-documents-study/
├── data/
│   └── dataset_info.jsonl       # Metadata for each image in the dataset
├── results/
│   ├── segmentations/           # Output of segmentation scripts (step 1)
│   ├── inferences/              # Output of inference scripts (step 2)
│   └── consistencies/           # Output of consistency evaluation (step 3a)
├── src/
│   ├── utils.py                 # Shared utilities (model loading, hashing, ...)
│   ├── lime/
│   │   ├── lime_utils.py        # Core LIME logic (perturbations, norms, ...)
│   │   └── save_neighbor_outputs.py  # Run LIME neighborhood inference
│   ├── metrics/
│   │   ├── consistency.py       # Explanation consistency (Spearman / Jaccard)
│   │   ├── deletion.py          # Deletion faithfulness metric
│   │   └── insertion.py         # Insertion faithfulness metric
│   └── segmentation/
│       ├── quickshift.py
│       ├── slic.py
│       ├── ocr_only.py
│       ├── ocr_grid.py
│       └── rectangle_without_bboxes.py
├── requirements.txt
└── README.md
```

---

## How to Run Experiments

### 0 - Requirements

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Create a `.env` file** at the root of the project with the following variables:
```env
DATA_DIR=/path/to/your/data
RESULTS_DIR=/path/to/your/results
```

---

### 1 - Run Segmentation Scripts

Segmentation scripts partition each image into superpixels or regions. Results are stored in `$RESULTS_DIR/segmentations/[segmentation_name]/[image_hash]/segmentation_mask.pt`.

Run the segmentation algorithm(s) you need:

```bash
# Quickshift segmentation
python src/segmentation/quickshift.py

# SLIC segmentation
python src/segmentation/slic.py

# OCR-only segmentation (segments based on detected text bounding boxes)
python src/segmentation/ocr_only.py

# OCR + background grid segmentation
python src/segmentation/ocr_grid.py --num_rows 4 --num_cols 4

# Uniform grid segmentation (no OCR bounding boxes)
python src/segmentation/rectangle_without_bboxes.py --num_rows 4 --num_cols 4
```

> All scripts default to loading dataset info from `$DATA_DIR/dataset_info.jsonl`. You can override this with `--data_info /path/to/custom_info.jsonl`.

---

### 2 - Run Inference Script (Save Neighbor Outputs)

This step runs the model on all LIME-perturbed versions of an image and saves the outputs (logits, probabilities, interpretable inputs) needed to fit the LIME explanation.

Results are stored in:
`$RESULTS_DIR/inferences/[dataset]/[model]/[segmentation]/[color]/[random_walk]/[image_hash]/`

```bash
python src/lime/save_neighbor_outputs.py \
    --dataset_name "rvlcdip" \
    --model_to_be_explained "resnet50" \
    --random_walk "uniform" \
    --path_to_image "/path/to/your/image.jpg" \
    --image_hash "hash_of_the_image" \
    --segmentation_algorithm "paddle_ocr" \
    --color "white" \
    --sample_number 10000
```

**Parameters:**

| Parameter | Description | Default |
| :--- | :--- | :--- |
| `--dataset_name` | Name of the dataset | `rvlcdip` |
| `--model_to_be_explained` | Model architecture to explain | `resnet50` |
| `--random_walk` | Perturbation sampling strategy | `uniform` |
| `--path_to_image` | Absolute path to the input image | — |
| `--image_hash` | Unique hash identifying the image | — |
| `--segmentation_algorithm` | Segmentation algorithm used in step 1 | `paddle_ocr` |
| `--color` | Replacement color for masked segments | `white` (`black`, `mean`) |
| `--sample_number` | Number of perturbed samples to generate | `10000` |

---

### 3a - Evaluate Explanation Consistency

This step measures the stability of LIME explanations by generating `n` independent explanations for the same image and computing Spearman correlation and Jaccard index between each pair.

Results are stored in:
`$RESULTS_DIR/consistencies/[image_hash]/[segmentation_algorithm]/[run_hash].parquet`

```bash
python src/metrics/consistency.py \
    --dataset_name "rvlcdip" \
    --model_to_be_explained "resnet50" \
    --path_to_image "/path/to/your/image.jpg" \
    --image_hash "hash_of_the_image" \
    --class_label "invoice" \
    --segmentation_algorithm "paddle_ocr" \
    --color "mean" \
    --output_type "logits" \
    --norm "euclidean_interpretable" \
    --kernel_width 0.1 \
    --n_neighbors 1000 \
    --n_explanations 50
```

**Parameters:**

| Parameter | Description | Default |
| :--- | :--- | :--- |
| `--dataset_name` | Name of the dataset | `rvlcdip` |
| `--model_to_be_explained` | Model architecture to explain | `resnet50` |
| `--path_to_image` | Absolute path to the input image | — |
| `--image_hash` | Unique hash identifying the image | — |
| `--class_label` | Class label to explain | `invoice` |
| `--segmentation_algorithm` | Segmentation algorithm used | `paddle_ocr` |
| `--color` | Replacement color for masked segments | `mean` |
| `--output_type` | Model outputs to use for fitting LIME | `logits` (`probas`) |
| `--norm` | Distance norm for computing sample weights | `euclidean_interpretable` |
| `--kernel_width` | Kernel width for the exponential similarity | `0.1` |
| `--n_neighbors` | Number of perturbed samples used per explanation | `1000` |
| `--n_explanations` | Number of independent explanations to compare | `50` |
| `--interpretable_model` | Local model type | `linear` (`ridge`) |
| `--random_walk` | Perturbation sampling strategy | `uniform` |
| `--alpha` | Regularization strength (ridge only) | — |

**Output columns include:** `mean_spearman_consistency`, `std_spearman_consistency`, `mean_jaccard_index_top_{5,10,20,50}`, `std_jaccard_index_top_{5,10,20,50}`, `mean_r2_score`, `std_r2_score`.

---

### 3b - Evaluate Faithfulness (Deletion & Insertion)

These scripts evaluate the faithfulness of LIME explanations by progressively removing (**deletion**) or revealing (**insertion**) image segments in order of their importance and measuring the model confidence at each step.

They sweep over a large grid of hyperparameters (segmentation algorithms, colors, norms, kernel widths, number of neighbors) for a single image at a time.

**Deletion** — starts from the full image and removes the most important segments first. A faithful explanation leads to a fast drop in model confidence.

```bash
python src/metrics/deletion.py --image_id 1
```

**Insertion** — starts from a fully masked image and inserts the most important segments first. A faithful explanation leads to a fast rise in model confidence.

```bash
python src/metrics/insertion.py --image_id 1
```

| Parameter | Description | Default |
| :--- | :--- | :--- |
| `--image_id` | 1-based index of the image in `dataset_info.jsonl` | `0` |

> Both scripts iterate over all combinations defined in `PARAMETERS_GRID` at the top of each file. Edit that dictionary to restrict or extend the search.

Results are saved as a `.parquet` (insertion) or `.csv` (deletion) file in `$RESULTS_DIR`.

