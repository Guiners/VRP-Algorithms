import csv
import json
import logging
logger = logging.getLogger()
class VRPInstance:
    def __init__(self, city_id, x, y):
        self.city_id = city_id
        self.x = x
        self.y = y

class VRPData:
    def __init__(self, cities, depot, vehicles):
        self.cities = cities              # List[VRPInstance]
        self.depot = depot                # VRPInstance
        self.vehicles = vehicles          # int

class VRPInstanceLoader:
    @classmethod
    def load(cls, csv_path, config_path):
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)

        depot_id = config["depot_id"]
        vehicles = config["vehicles"]

        cities = []
        depot = None

        with open(csv_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                city = VRPInstance(
                    city_id=int(row["CityID"]),
                    x=float(row["X"]),
                    y=float(row["Y"])
                )
                if city.city_id == depot_id:
                    depot = city
                else:
                    cities.append(city)

        if depot is None:
            raise ValueError(f"Depot ID {depot_id} not found in city list!")

        return VRPData(cities=cities, depot=depot, vehicles=vehicles)

data = VRPInstanceLoader.load("../../../datasets/test/vpr201_7_3.csv", "../../../datasets/test/vpr201_7_3.json")

logger.info("Liczba klientów:", len(data.cities))
logger.info("Depot ID:", data.depot.city_id)
logger.info("Liczba pojazdów:", data.vehicles)