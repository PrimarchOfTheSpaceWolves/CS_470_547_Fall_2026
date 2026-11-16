###############################################################################
# IMPORTS
###############################################################################

import os
import sys
import cv2
import pandas as pd
import numpy as np
import shutil
from enum import Enum

base_dir = "assign03"

# Image directory (created by Prepare_Assign03.py)
# NOTE: Can be overridden by setting the environment variable ASSIGN03_DATA_DIR
DATA_DIR = os.environ.get("ASSIGN03_DATA_DIR", os.path.join(".", "data", "TXL_PBC"))

# Ground truth bounding boxes (included in the repository)
# NOTE: Can be overridden by setting the environment variable ASSIGN03_GROUND_TRUTH
DATA_CSV = os.environ.get("ASSIGN03_GROUND_TRUTH", os.path.join(base_dir, "Assign03_ground_truth.csv"))

class CELL_TYPES(Enum):
    RBC = "RBC"
    WBC = "WBC"
    PLATE = "Platelets"

# A predicted box must overlap a ground truth box by at least this much to count as a match
IOU_THRESHOLD = 0.5

###############################################################################
# EXTRACTS BOUNDING BOXES FOR PARTICULAR CELL TYPE
###############################################################################
def unpack_one_cell_type_only(objects, cell_type):
    # Get the bounding boxes and labels from the dictionary
    bboxes = objects['bbox']
    labels = objects['label']

    # Create an empty list to hold the bounding boxes we want to keep
    cell_boxes = []

    # For each entry...
    for i in range(len(bboxes)):
        # Get one bounding box and label
        bb = bboxes[i]
        label = labels[i]

        # If this is the cell we're looking for...
        if label == cell_type.value:
            # Add to our list
            cell_boxes.append(bb)

    return cell_boxes

###############################################################################
# LOADS UP DATA
# - "train" and "test" come from PBC
# - "cross" comes from BCCD and is ONLY used for cross-dataset testing
###############################################################################
def load_and_prepare_data(cell_type):
    # Assumes that Prepare_Assign03 has already been run

    # Load up CSV file
    data = pd.read_csv(DATA_CSV)

    def load_objects(df):
        object_data = {
            "bbox": [],
            "label": [],
            "filename": ""
        }
        # For each row
        for index, row in df.iterrows():
            # Get label
            label = row["cell_type"]

            # Get the bounding box
            # y1, x1, y2, x2
            box = np.array([row["ymin"], row["xmin"], row["ymax"], row["xmax"]], dtype="int")

            # Add to lists
            object_data["bbox"].append(box)
            object_data["label"].append(label)

        return object_data

    def prepare_dataset(split, cell_type):
        all_data = []
        split_data = data[data["split"] == split]
        # For each file
        for filename in split_data["filename"].unique():
            # Load image
            image = cv2.imread(os.path.join(DATA_DIR, "images", filename))
            # Load up bounding box and label info
            objects = load_objects(split_data[(split_data["filename"] == filename)
                                              & (split_data["cell_type"] == cell_type.value)])
            objects["filename"] = filename
            # Add to list
            all_data.append((image, objects))

        return all_data

    # Create "datasets" for training, testing, and cross-dataset testing
    train_data = prepare_dataset("train", cell_type)
    test_data = prepare_dataset("test", cell_type)
    cross_data = prepare_dataset("cross", cell_type)

    # Number of items
    print("Number of training images:", len(train_data))
    print("Number of testing images:", len(test_data))
    print("Number of cross-dataset testing images:", len(cross_data))

    return train_data, test_data, cross_data

###############################################################################
# COMPUTES INTERSECTION OVER UNION FOR BOUNDING BOXES
###############################################################################
def compute_one_IOU(predicted, ground):
    # Bounding box stored as (y1, x1, y2, x2)
    def compute_area(left, right, top, bottom):
        width = right - left
        height = bottom - top
        width = max(0, width)
        height = max(0, height)
        area = width * height
        return area

    # Get intersection
    left = max(predicted[1], ground[1])
    right = min(predicted[3], ground[3])
    top = max(predicted[0], ground[0])
    bottom = min(predicted[2], ground[2])
    intersection = compute_area(left, right, top, bottom)

    # Get union
    area_pred = compute_area(predicted[1], predicted[3], predicted[0], predicted[2])
    area_ground = compute_area(ground[1], ground[3], ground[0], ground[2])
    union = area_pred + area_ground - intersection

    # Get IOU
    if union <= 0:
        return 0.0
    iou = intersection / union

    return iou

###############################################################################
# MATCHES PREDICTED BOXES TO GROUND TRUTH BOXES (ONE IMAGE)
# - Greedy one-to-one matching: highest IOU pairs are matched first
# - Each predicted box and each ground truth box can be matched at most once
# - Pairs with IOU below IOU_THRESHOLD are never matched
# Returns:
# - true positives (matched pairs)
# - false positives (unmatched predicted boxes)
# - false negatives (unmatched ground truth boxes)
# - list of IOU values for the matched pairs
###############################################################################
def match_boxes(all_predicted, all_ground, iou_threshold=IOU_THRESHOLD):
    # Compute IOU for every (predicted, ground) pair
    all_pairs = []
    for i, predicted in enumerate(all_predicted):
        for j, ground in enumerate(all_ground):
            one_IOU = compute_one_IOU(predicted, ground)
            if one_IOU >= iou_threshold:
                all_pairs.append((one_IOU, i, j))

    # Sort from highest to lowest IOU
    all_pairs.sort(key=lambda pair: pair[0], reverse=True)

    # Greedily match
    used_predicted = set()
    used_ground = set()
    matched_IOU = []
    for one_IOU, i, j in all_pairs:
        if i not in used_predicted and j not in used_ground:
            used_predicted.add(i)
            used_ground.add(j)
            matched_IOU.append(one_IOU)

    true_positives = len(matched_IOU)
    false_positives = len(all_predicted) - true_positives
    false_negatives = len(all_ground) - true_positives

    return true_positives, false_positives, false_negatives, matched_IOU

###############################################################################
# DRAWS BOUNDING BOXES ON IMAGE
###############################################################################
def draw_bounding_boxes(image, bounding_boxes, color):
    # For each box...
    for bb in bounding_boxes:
        cv2.rectangle(image, (int(bb[1]), int(bb[0])), (int(bb[3]), int(bb[2])), color, thickness=2)

###############################################################################
# PREDICTS BOUNDING BOXES ON DATASET AND COMPUTES METRICS
###############################################################################
def predict_dataset(dataset, model_dir, prefix, out_dir, cell_type, cell_finder):

    # Running totals (over the WHOLE dataset)
    total_TP = 0
    total_FP = 0
    total_FN = 0
    all_matched_IOU = []
    all_count_error = []
    all_relative_count_error = []

    # Get total count
    total_cnt = len(dataset)

    # Print starting
    print("Starting on", prefix, "(" + str(total_cnt) + " samples total)")

    # For each datapoint...
    image_index = 0

    # Create cell finder
    cell_finder_obj = cell_finder(model_dir)
    if cell_type == CELL_TYPES.WBC:
        find_cell_func = cell_finder_obj.find_WBC
    elif cell_type == CELL_TYPES.RBC:
        find_cell_func = cell_finder_obj.find_RBC
    else:
        raise ValueError("Unsupported cell type for evaluation.")

    for data_pack in dataset:
        # Each item is a tuple, so separate data into image and objects
        image = np.copy(data_pack[0])
        objects = data_pack[1]
        orig_filename = objects["filename"]

        # Save output using original filename (with dataset prefix)
        output_filename = prefix + "_" + orig_filename

        # Objects is a dictionary, so we'll unpack the bounding boxes
        # for specific cells only
        true_bounding_boxes = unpack_one_cell_type_only(objects, cell_type)
        true_cell_count = len(true_bounding_boxes)

        # Calculate bounding boxes using your approach
        pred_bounding_boxes = find_cell_func(image)
        pred_cell_count = len(pred_bounding_boxes)

        # Draw bounding boxes on image
        draw_bounding_boxes(image, true_bounding_boxes, (0,0,0))
        draw_bounding_boxes(image, pred_bounding_boxes, (0,255,0))

        # Show images (DEBUG)
        #cv2.imshow("IMAGE", image)
        #cv2.waitKey(-1)

        # Save image
        cv2.imwrite(os.path.join(out_dir, output_filename), image)

        # Match predicted boxes to ground truth boxes
        TP, FP, FN, matched_IOU = match_boxes(pred_bounding_boxes, true_bounding_boxes)
        total_TP += TP
        total_FP += FP
        total_FN += FN
        all_matched_IOU += matched_IOU

        # How far off is the number of cells predicted?
        count_error = abs(pred_cell_count - true_cell_count)
        all_count_error.append(count_error)
        all_relative_count_error.append(count_error / max(true_cell_count, 1))

        # Increment index
        image_index += 1

        # Print progress
        percent = 100.0*image_index / total_cnt
        print("%.1f%% complete...       " % percent, end="\r", flush=True)

    # Print complete
    print(prefix, "complete!                    ")

    # Compute metrics
    def safe_divide(a, b):
        return a / b if b > 0 else 0.0

    metrics = {}
    metrics["Precision"] = safe_divide(total_TP, total_TP + total_FP)
    metrics["Recall"] = safe_divide(total_TP, total_TP + total_FN)
    metrics["F1"] = safe_divide(2*metrics["Precision"]*metrics["Recall"],
                                metrics["Precision"] + metrics["Recall"])
    metrics["Matched_IOU"] = float(np.mean(all_matched_IOU)) if len(all_matched_IOU) > 0 else 0.0
    metrics["Count_MAE"] = float(np.mean(all_count_error)) if total_cnt > 0 else 0.0
    metrics["Count_Relative_Error"] = float(np.mean(all_relative_count_error)) if total_cnt > 0 else 0.0
    metrics["TP"] = total_TP
    metrics["FP"] = total_FP
    metrics["FN"] = total_FN

    # Return metrics
    return metrics

###############################################################################
# PRINTS METRICS (to STDOUT or file)
###############################################################################
def print_metrics(all_metrics, stream=sys.stdout):
    for data_type in all_metrics:
        print(data_type + ":", file=stream)
        for key in all_metrics[data_type]:
            print("\t", key, "=", all_metrics[data_type][key], file=stream)

###############################################################################
# RECREATE DIRECTORY
###############################################################################
def recreate_dir(out_dir):
    # Does directory exist?
    if os.path.exists(out_dir):
        check_overwrite = input("Folder exists: "
                                + out_dir
                                + "\nDo you wish to overwrite it? (y/n) ")
        if check_overwrite == "y":
            shutil.rmtree(out_dir)
        else:
            print("Exiting...")
            exit(1)

    # Create output directory
    os.makedirs(out_dir)

###############################################################################
# RUN EVALUATION
###############################################################################
def evaluate(cell_finder, cell_type):
    # Load datasets
    train_data, test_data, cross_data = load_and_prepare_data(cell_type)

    # Prepare output directory
    out_dir = os.path.join(base_dir, "output_" + cell_type.value)
    recreate_dir(out_dir)

    # Also grab model directory
    model_dir = os.path.join(base_dir, "output_models")

    # Prepare metrics
    all_metrics = {}

    # Predict for training
    all_metrics["TRAINING"] = predict_dataset(train_data, model_dir, "TRAIN", out_dir,
                                                cell_type, cell_finder)

    # Predict for testing
    all_metrics["TESTING"] = predict_dataset(test_data, model_dir, "TEST", out_dir,
                                                cell_type, cell_finder)

    # Predict for cross-dataset testing
    all_metrics["CROSS_DATASET"] = predict_dataset(cross_data, model_dir, "CROSS", out_dir,
                                                    cell_type, cell_finder)

    # Save metrics
    print_metrics(all_metrics)
    with open(out_dir + "/ALL_RESULTS_" + cell_type.value + ".txt", "w") as f:
        print_metrics(all_metrics, f)

###############################################################################
# RUN TRAINING
###############################################################################
def train(cell_finder, cell_type):
    # Load datasets (ONLY the training data is used)
    train_data, _, _ = load_and_prepare_data(cell_type)

    # Prepare model directory (ONLY create if it doesn't exist)
    model_dir = os.path.join(base_dir, "output_models")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Run desired training
    cell_finder_obj = cell_finder(model_dir)
    if cell_type == CELL_TYPES.WBC:
        cell_finder_obj.train_WBC(train_data)
    elif cell_type == CELL_TYPES.RBC:
        cell_finder_obj.train_RBC(train_data)
    else:
        raise ValueError("Unsupported cell type for training.")

