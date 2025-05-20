import csv
import json
import math
from typing import Dict, List, Optional, Any

from src.utils.logger_config import logger


class VRPInstance:
    """
    Represents a single city (node) in the VRP problem.
    """

    def __init__(self, city_id: int, x: float, y: float) -> None:
        self.city_id: int = city_id
        self.x: float = x
        self.y: float = y

    def convert_to_dict(self) -> Dict[str, Any]:
        """
        Converts the instance to a dictionary.
        """
        return {"city_id": self.city_id, "x": self.x, "y": self.y}

    def distance_to(self, other_city: "VRPInstance") -> float:
        """
        Calculates the Euclidean distance to another city.
        """
        return math.hypot(self.x - other_city.x, self.y - other_city.y)


class VRPData:
    """
    Holds the data for a VRP problem instance.
    """

    def __init__(self, cities: List[VRPInstance], depot: VRPInstance, vehicles: int) -> None:
        self.cities: List[VRPInstance] = cities
        self.depot: VRPInstance = depot
        self.vehicles: int = vehicles


class VehicleInfo:
    """
    Stores information about the vehicle's fuel consumption and price.
    """

    def __init__(self, avg_fuel_consumption: float, fuel_price: float) -> None:
        self.mpg: float = avg_fuel_consumption
        self.fuel_price: float = fuel_price

    def count_cost(self, distance: float) -> float:
        """
        Calculates the cost of the trip based on distance, fuel consumption, and fuel price.
        """
        return round((distance / 100) * self.fuel_price * self.mpg, 2)


class VRPInstanceLoader:
    """
    Loads VRP instances from files and saves results.
    """

    @classmethod
    def load_dataset(cls, csv_path: str, config_path: str) -> VRPData:
        """
        Loads a VRP dataset from a CSV file and a JSON config file.
        """
        with open(config_path, "r") as config_file:
            config = json.load(config_file)

        depot_id: int = config["depot_id"]
        vehicles: int = config["vehicles"]

        cities: List[VRPInstance] = []
        depot: Optional[VRPInstance] = None

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
    def decode_routes(routes: List[List[VRPInstance]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Converts a list of routes (each route is a list of VRPInstance) to a dictionary format.
        """
        def routes_generator():
            _car_count = 1
            for _car_route in routes:
                yield _car_route, _car_count
                _car_count += 1

        decoded_routes: Dict[str, List[Dict[str, Any]]] = {}
        for car_route, car_count in routes_generator():
            decoded_routes[f"car{car_count}"] = [
                city.convert_to_dict() for city in car_route
            ]

        return decoded_routes

    @classmethod
    def save_results_to_file(
        cls,
        total_distance_in_km: float,
        processing_time: float,
        routes: List[List[VRPInstance]],
        vehicle_info: VehicleInfo,
        output_file_path: str,
    ) -> None:
        """
        Saves the results of the VRP solution to a JSON file.
        """
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
            logger.debug("File was created")