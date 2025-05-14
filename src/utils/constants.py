ABSOLUTE_INPUT_TRAIN_FOLDER_PATH = r"C:\Users\rkoza\VRP_Algorithms\datasets\train"
ABSOLUTE_INPUT_VALIDATE_FOLDER_PATH = r"C:\Users\rkoza\VRP_Algorithms\datasets\validate"
ABSOLUTE_INPUT_TEST_FOLDER_PATH = r"C:\Users\rkoza\VRP_Algorithms\datasets\test"

ABSOLUTE_OUTPUT_TRAIN_FOLDER_PATH = (
    r"C:\Users\rkoza\VRP_Algorithms\results\{algorithm}\train"
)
ABSOLUTE_OUTPUT_VALIDATE_FOLDER_PATH = (
    r"C:\Users\rkoza\VRP_Algorithms\results\{algorithm}\validate"
)
ABSOLUTE_OUTPUT_TEST_FOLDER_PATH = (
    r"C:\Users\rkoza\VRP_Algorithms\results\{algorithm}\test"
)

CSV = ".csv"
JSON = ".json"

RELATIVE_OUTPUT_TRAIN_PATH = "../../datasets/train/{path}"
RELATIVE_OUTPUT_VALIDATE_PATH = "../../datasets/validate/{path}"
RELATIVE_OUTPUT_TEST_PATH = "../../datasets/test/{path}"
RELATIVE_INPUT_CSV_PATH = "../../datasets/raw/cities_in_csv.csv"

NN_ALGORITHM_NAME = "nearest_neighbor_algorithm"
GENETIC_ALGORITHM_NAME = "genetic_algorithm"
CLARKE_WRIGHT_SAVINGS_NAME = "clarke_wright_savings"
GRASP_ALGORITHM_NAME = "grasp"
