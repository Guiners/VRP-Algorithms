import csv
import logging

import tsplib95

# logging.basicConfig(
#     level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
# )
logger = logging.getLogger()

file_path = "../../datasets/raw/kagglesanta2018.197769.tsp"
problem = tsplib95.load(file_path)


output_path = "../../datasets/raw/cities_in_csv.csv"
with open(output_path, mode="w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["CityID", "X", "Y"])

    for city_id, (x, y) in problem.node_coords.items():
        writer.writerow([city_id, x, y])

with open(output_path, mode="r") as csv_file:
    reader = csv.reader(csv_file)
    rows = list(reader)
    num_rows = len(rows) - 1
    num_columns = len(rows[0]) if rows else 0

    logger.info(f"File: {output_path}")
    logger.info(f"Number of columns: {num_columns}")
    logger.info(f"Number of rows: {num_rows}")
