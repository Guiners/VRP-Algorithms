import random
import time
import math
from copy import deepcopy
from src.algorithms.tools.vrp_tools import VRPInstanceLoader


class GeneticAlgorithmVRP(VRPInstanceLoader):
    def __init__(self, vehicle_info, population_size=50, generations=200, mutation_rate=0.1, tournament_size=5):
        super().__init__()
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.vehicle_info = vehicle_info

    def auto_tune_params(self, cities):
        n = len(cities)
        if n <= 30:
            self.population_size = 30
            self.generations = 100
            self.mutation_rate = 0.05
            self.tournament_size = 2
        elif n <= 100:
            self.population_size = 50
            self.generations = 200
            self.mutation_rate = 0.1
            self.tournament_size = 5
        else:
            self.population_size = 100
            self.generations = 500
            self.mutation_rate = 0.15
            self.tournament_size = 10

    def _distance(self, route, depot):
        distance = 0
        current = depot
        for city in route:
            distance += current.distance_to(city)
            current = city
        distance += current.distance_to(depot)
        return distance

    def _split_into_routes(self, chromosome, vehicles):
        chunk_size = len(chromosome) // vehicles
        return [chromosome[i * chunk_size:(i + 1) * chunk_size] for i in range(vehicles)]

    def _fitness(self, chromosome, vehicles, depot):
        total_distance = 0
        routes = self._split_into_routes(chromosome, vehicles)
        for route in routes:
            total_distance += self._distance(route, depot)
        return total_distance

    def _create_individual(self, cities):
        cities = cities[:]
        random.shuffle(cities)
        return cities

    def _tournament_selection(self, population, vehicles, depot):
        tournament = random.sample(population, self.tournament_size)
        tournament.sort(key=lambda chromo: self._fitness(chromo, vehicles, depot))
        return tournament[0]

    def _crossover(self, parent1, parent2):
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        child = [None] * size
        child[start:end] = parent1[start:end]

        pointer = 0
        for city in parent2:
            if city not in child:
                while child[pointer] is not None:
                    pointer += 1
                child[pointer] = city
        return child

    def _mutate(self, chromosome):
        for i in range(len(chromosome)):
            if random.random() < self.mutation_rate:
                j = random.randint(0, len(chromosome) - 1)
                chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
        return chromosome

    def solve(self, csv_path, config_path, output_file_path):
        start_time = time.time()
        data = self.load_dataset(csv_path, config_path)
        self.auto_tune_params(data.cities)
        population = [self._create_individual(data.cities) for _ in range(self.population_size)]
        best_individual = min(population, key=lambda chromo: self._fitness(chromo, data.vehicles, data.depot))

        for _ in range(self.generations):
            new_population = []
            for _ in range(self.population_size):
                parent1 = self._tournament_selection(population, data.vehicles, data.depot)
                parent2 = self._tournament_selection(population, data.vehicles, data.depot)
                child = self._crossover(parent1, parent2)
                child = self._mutate(child)
                new_population.append(child)

            population = new_population
            current_best = min(population, key=lambda chromo: self._fitness(chromo, data.vehicles, data.depot))
            if self._fitness(current_best, data.vehicles, data.depot) < self._fitness(best_individual, data.vehicles, data.depot):
                best_individual = deepcopy(current_best)

        end_time = time.time()
        total_distance = self._fitness(best_individual, data.vehicles, data.depot)
        processing_time = end_time - start_time

        total_distance_in_km = total_distance / 1000


        routes = self._split_into_routes(best_individual, data.vehicles)
        routes = [[data.depot] + r + [data.depot] for r in routes]
        self.save_results_to_file(total_distance_in_km, processing_time, routes, self.vehicle_info, output_file_path)

        return routes
