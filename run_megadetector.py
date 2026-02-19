from megadetector.detection.run_detector_batch import load_and_run_detector_batch, write_results_to_file
from megadetector.utils import path_utils

image_folder = "./data/train_features"
output_file = "./data/megadetector_train.json"

# find all images recursively
image_file_names = path_utils.find_images(image_folder, recursive=True)

# run — 'MDV5A' downloads automatically, no need to specify a .pt file path
results = load_and_run_detector_batch('MDV5A', image_file_names)

# save separately
write_results_to_file(
    results,
    output_file,
    relative_path_base=image_folder,
    detector_file='MDV5A'
)

# for test_dataset
write_results_to_file(
    load_and_run_detector_batch('MDV5A', path_utils.find_images("./data/test_features", recursive=True)),
    "./data/megadetector_test.json",
    relative_path_base="./data/test_features",
    detector_file='MDV5A'
)
