import random
import time
from copy import deepcopy
from src.algorithms.tools.vrp_tools import VRPInstanceLoader
from src.utils.logger_config import logger

class GeneticAlgorithmVRP(VRPInstanceLoader):
    def __init__(self, vehicle_info, population_size=100, generations=300, mutation_rate=0.1, tournament_size=5):
        super().__init__()
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.vehicle_info = vehicle_info
        logger.info("Initialized GeneticAlgorithmVRP with population_size=%d, generations=%d, mutation_rate=%.2f, tournament_size=%d",
                    population_size, generations, mutation_rate, tournament_size)

    def _distance(self, route, depot):
        distance = 0
        current = depot
        for city in route:
            distance += current.distance_to(city)
            current = city
        distance += current.distance_to(depot)
        return distance

    def _split_into_routes(self, chromosome, vehicles):
        n = len(chromosome)
        base_size = n // vehicles
        remainder = n % vehicles
        routes = []
        start = 0
        for i in range(vehicles):
            end = start + base_size + (1 if i < remainder else 0)
            routes.append(chromosome[start:end])
            start = end
        return routes

    def _fitness(self, chromosome, vehicles, depot):
        total_distance = 0
        routes = self._split_into_routes(chromosome, vehicles)
        for route in routes:
            total_distance += self._distance(route, depot)
        return total_distance

    def _create_individual(self, cities):
        individual = cities[:]
        random.shuffle(individual)
        return individual

    def _tournament_selection(self, population, vehicles, depot):
        tournament = random.sample(population, self.tournament_size)
        return min(tournament, key=lambda chromo: self._fitness(chromo, vehicles, depot))

    def _crossover(self, parent1, parent2):
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        child = [None] * size
        child[start:end] = parent1[start:end]

        pointer = 0
        for city in parent2:
            if city not in child:
                while pointer < size and child[pointer] is not None:
                    pointer += 1
                if pointer < size:
                    child[pointer] = city
                else:
                    break
        return child

    def _mutate(self, chromosome):
        for i in range(len(chromosome)):
            if random.random() < self.mutation_rate:
                j = random.randint(0, len(chromosome) - 1)
                chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
        return chromosome

    def auto_tune_params(self, cities):
        n = len(cities)
        logger.info("Auto-tuning parameters for %d cities", n)
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
        logger.info("Tuned parameters: population_size=%d, generations=%d, mutation_rate=%.2f, tournament_size=%d",
                    self.population_size, self.generations, self.mutation_rate, self.tournament_size)


    def solve(self, csv_path, config_path, output_file_path):
        logger.info("Started solving VRP using Genetic Algorithm")
        start_time = time.time()
        data = self.load_dataset(csv_path, config_path)

        logger.info("Loaded dataset with %d cities and %d vehicles", len(data.cities), data.vehicles)
        self.auto_tune_params(data.cities)
        population = [self._create_individual(data.cities) for _ in range(self.population_size)]
        best_individual = min(population, key=lambda chromo: self._fitness(chromo, data.vehicles, data.depot))
        best_fitness = self._fitness(best_individual, data.vehicles, data.depot)
        logger.info("Initial best fitness: %.2f", best_fitness)

        for generation in range(self.generations):
            new_population = [deepcopy(best_individual)]  # Elitism
            while len(new_population) < self.population_size:
                parent1 = self._tournament_selection(population, data.vehicles, data.depot)
                parent2 = self._tournament_selection(population, data.vehicles, data.depot)
                child = self._crossover(parent1, parent2)
                self._mutate(child)
                new_population.append(child)

            population = new_population
            current_best = min(population, key=lambda chromo: self._fitness(chromo, data.vehicles, data.depot))
            current_fitness = self._fitness(current_best, data.vehicles, data.depot)

            if current_fitness < best_fitness:
                best_individual = deepcopy(current_best)
                best_fitness = current_fitness
                logger.info("Generation %d - New best fitness: %.2f", generation + 1, best_fitness)

            if generation % 10 == 0 or generation == self.generations - 1:
                logger.debug("Generation %d - Current best: %.2f", generation + 1, current_fitness)

        end_time = time.time()
        processing_time = end_time - start_time
        total_distance_km = best_fitness / 1000

        logger.info("Optimization completed in %.2f seconds", processing_time)
        logger.info("Total distance: %.2f km", total_distance_km)

        routes = self._split_into_routes(best_individual, data.vehicles)
        routes = [[data.depot] + r + [data.depot] for r in routes]
        self.save_results_to_file(total_distance_km, processing_time, routes, self.vehicle_info, output_file_path)

        logger.info("Results saved to %s", output_file_path)
        return routes
