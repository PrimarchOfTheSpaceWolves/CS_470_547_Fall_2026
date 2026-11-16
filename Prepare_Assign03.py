###############################################################################
# IMPORTS
###############################################################################

import requests
import os
import pandas as pd
import shutil
from General_Assign03 import *

###############################################################################
# CONSTANTS
###############################################################################

# Pinned version of TXL-PBC (so the data cannot change mid-semester)
TXL_PBC_COMMIT = "fd9aadd9636d590ce5b25a1936e61bff4e581a48"

###############################################################################
# DOWNLOAD FILE
###############################################################################

def download_file(file_url, output_dir, output_filename):

    r = requests.get(file_url, verify=True)
    r.raise_for_status()
    save_path = os.path.join(output_dir, output_filename)

    with open(save_path,'wb') as f:
        f.write(r.content)

###############################################################################
# MAIN
###############################################################################

def main():
    # Using an external data directory? Make sure we really want to overwrite it.
    if "ASSIGN03_DATA_DIR" in os.environ and os.path.exists(DATA_DIR):
        check_overwrite = input("ASSIGN03_DATA_DIR is set; this will DELETE and re-download: "
                                + DATA_DIR + "\nContinue? (y/n) ")
        if check_overwrite != "y":
            print("Exiting...")
            exit(1)

    # Re-create output directory if it doesn't exist
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)
    os.makedirs(DATA_DIR)

    # Download the list of files...
    print("***********************************************************")
    print("Images and annotations provided by TXL-PBC:")
    print("     https://github.com/lugan113/TXL-PBC_Dataset")
    print("Original images from:")
    print("     PBC: https://www.kaggle.com/datasets/orvile/microscopic-peripheral-blood-cell-images")
    print("     BCCD: https://github.com/Shenggan/BCCD_Dataset")
    print("***********************************************************")

    base_image_url = ("https://raw.githubusercontent.com/lugan113/TXL-PBC_Dataset/"
                      + TXL_PBC_COMMIT + "/TXL-PBC/images/")

    # Open CSV file with ground truth (already in the repository)
    data = pd.read_csv(DATA_CSV)

    # Make image directory
    image_dir = os.path.join(DATA_DIR, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    # Loop through each image file and download it
    all_images = data[["filename", "txl_split", "txl_filename"]].drop_duplicates()
    for _, row in all_images.iterrows():
        print("*", row["filename"])
        full_image_url = base_image_url + row["txl_split"] + "/" + row["txl_filename"]
        download_file(full_image_url, image_dir, row["filename"])

if __name__ == "__main__":
    main()

