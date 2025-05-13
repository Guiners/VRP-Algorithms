import csv
import json
import math
from typing import Dict

from src.utils.logger_config import logger


class VRPInstance:
    def __init__(self, city_id, x, y):
        self.city_id = city_id
        self.x = x
        self.y = y

    def convert_to_dict(self):
        return {"city_id": self.city_id, "x": self.x, "y": self.y}

    def distance_to(self, other_city):
        return math.hypot(self.x - other_city.x, self.y - other_city.y)


class VRPData:
    def __init__(self, cities, depot, vehicles):
        self.cities = cities  # List[VRPInstance]
        self.depot = depot  # VRPInstance
        self.vehicles = vehicles  # int


class VehicleInfo:
    def __init__(self, avg_fuel_consumption, fuel_price):
        self.mpg = avg_fuel_consumption
        self.fuel_price = fuel_price

    def count_cost(self, distance: int):
        return round((distance / 100) * self.fuel_price * self.mpg, 2)


class VRPInstanceLoader:
    @classmethod
    def load_dataset(cls, csv_path, config_path):
        with open(config_path, "r") as config_file:
            config = json.load(config_file)

        depot_id = config["depot_id"]
        vehicles = config["vehicles"]

        cities = []
        depot = None

        with open(csv_path, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                city = VRPInstance(
                    city_id=int(row["CityID"]), x=float(row["X"]), y=float(row["Y"])
                )
                if city.city_id == depot_id:
                    depot = city
                else:
                    cities.append(city)

        if depot is None:
            raise ValueError(f"Depot ID {depot_id} not found in city list!")

        return VRPData(cities=cities, depot=depot, vehicles=vehicles)

    @staticmethod
    def decode_routes(routes) -> Dict:

        def routes_generator():
            _car_count = 1
            for _car_route in routes:
                yield _car_route, _car_count
                _car_count += 1

        decoded_routes: dict = {}
        for car_route, car_count in routes_generator():
            decoded_routes[f"car{car_count}"] = [
                cities.convert_to_dict() for cities in car_route
            ]

        return decoded_routes

    @classmethod
    def save_results_to_file(
        cls,
        total_distance_in_km,
        processing_time,
        routes,
        vehicle_info,
        output_file_path,
    ):
        data_to_save = {
            "routes": cls.decode_routes(routes),
            "distance": round(total_distance_in_km, 2),
            "processing_time": processing_time,
            "cost": vehicle_info.count_cost(total_distance_in_km),
        }
        logger.debug("Routes: %s", data_to_save["routes"])
        logger.debug("Total distance: %s", data_to_save["distance"])
        logger.debug("Total processing time: %s", data_to_save["processing_time"])
        logger.debug("Cost of whole operation in $: %s", data_to_save["cost"])
        with open(output_file_path, "w") as json_file:
            json.dump(data_to_save, json_file, indent=4)
            logger.info("File was created")


# data = VRPInstanceLoader.load_dataset("../../../datasets/test/vpr201_7_3.csv", "../../../datasets/test/vpr201_7_3.json")
#
# logger.info("Liczba klientów: %s", len(data.cities))
# logger.info("Depot ID: %s", data.depot.city_id)
# logger.info("Liczba pojazdów: %s", data.vehicles)
