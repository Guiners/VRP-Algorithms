import random
import time
from copy import deepcopy
from typing import List, Any

from src.algorithms.tools.vrp_tools import VRPInstance, VehicleInfo, VRPInstanceLoader
from src.utils.logger_config import logger


class GeneticAlgorithmVRP(VRPInstanceLoader):
    """
    Genetic Algorithm for solving the Vehicle Routing Problem (VRP).
    """

    def __init__(
        self,
        vehicle_info: VehicleInfo,
        population_size: int = 200,
        generations: int = 500,
        mutation_rate: float = 0.1,
        tournament_size: int = 5,
        use_biased_selection: bool = False,
        seed_count: int = 10,
        elite_fraction: float = 0.1,
        relocate_samples_per_vehicle: int = 5,
    ) -> None:
        """
        Initialize the genetic algorithm parameters.

        :param vehicle_info: Vehicle information (fuel consumption, price, etc.)
        :param population_size: Number of individuals in the population
        :param generations: Number of generations to run
        :param mutation_rate: Probability of mutation per gene
        :param tournament_size: Number of individuals in tournament selection
        :param use_biased_selection: Whether to use biased selection
        :param seed_count: Number of initial seed individuals
        :param elite_fraction: Fraction of elites preserved each generation
        :param relocate_samples_per_vehicle: Number of relocate/exchange samples per vehicle
        """
        super().__init__()
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.vehicle_info = vehicle_info
        self.use_biased = use_biased_selection
        self.seed_count = seed_count
        self.elite_fraction = elite_fraction
        self.relocate_samples = relocate_samples_per_vehicle
        logger.info(
            "GA init: pop=%d, gen=%d, mut=%.2f, tourn=%d, biased=%s, seeds=%d",
            population_size,
            generations,
            mutation_rate,
            tournament_size,
            use_biased_selection,
            seed_count,
        )

    def _distance(self, route: List[VRPInstance], depot: VRPInstance) -> float:
        """
        Calculate the total distance of a route (including return to depot).

        :param route: List of cities in the route
        :param depot: Depot instance
        :return: Total distance
        """
        d = 0.0
        cur = depot
        for city in route:
            d += cur.distance_to(city)
            cur = city
        d += cur.distance_to(depot)
        return d

    def _split_into_routes(self, chromosome: List[VRPInstance], vehicles: int) -> List[List[VRPInstance]]:
        """
        Split a chromosome into routes for each vehicle.

        :param chromosome: List of cities (individual)
        :param vehicles: Number of vehicles
        :return: List of routes (one per vehicle)
        """
        n = len(chromosome)
        base = n // vehicles
        rem = n % vehicles
        routes, start = [], 0
        for i in range(vehicles):
            end = start + base + (1 if i < rem else 0)
            routes.append(chromosome[start:end])
            start = end
        return routes

    def _fitness(self, chromosome: List[VRPInstance], vehicles: int, depot: VRPInstance) -> float:
        """
        Calculate the fitness (total distance) of a chromosome.

        :param chromosome: List of cities (individual)
        :param vehicles: Number of vehicles
        :param depot: Depot instance
        :return: Total distance (fitness)
        """
        total = 0.0
        for route in self._split_into_routes(chromosome, vehicles):
            total += self._distance(route, depot)
        return total

    def _create_individual(self, cities: List[VRPInstance]) -> List[VRPInstance]:
        """
        Create a random individual (random permutation of cities).

        :param cities: List of city instances
        :return: Shuffled list of cities
        """
        ind = cities[:]
        random.shuffle(ind)
        return ind

    def _nearest_neighbor_chromosome(self, cities: List[VRPInstance], depot: VRPInstance) -> List[VRPInstance]:
        """
        Create an individual using the nearest neighbor heuristic.

        :param cities: List of city instances
        :param depot: Depot instance
        :return: List of cities ordered by nearest neighbor
        """
        unvisited = set(cities)
        tour = []
        cur = depot
        while unvisited:
            nxt = min(unvisited, key=lambda c: cur.distance_to(c))
            tour.append(nxt)
            unvisited.remove(nxt)
            cur = nxt
        return tour

    def _seed_population(self, cities: List[VRPInstance], depot: VRPInstance) -> List[List[VRPInstance]]:
        """
        Generate initial population using nearest neighbor heuristic.

        :param cities: List of city instances
        :param depot: Depot instance
        :return: List of seed individuals
        """
        seeds = []
        for _ in range(min(self.seed_count, self.population_size)):
            chrom = self._nearest_neighbor_chromosome(cities, depot)
            seeds.append(chrom[:])
        return seeds

    def _tournament_selection(
        self, population: List[List[VRPInstance]], vehicles: int, depot: VRPInstance
    ) -> List[VRPInstance]:
        """
        Select an individual using tournament selection.

        :param population: List of individuals
        :param vehicles: Number of vehicles
        :param depot: Depot instance
        :return: Selected individual
        """
        tour = random.sample(population, self.tournament_size)
        return min(tour, key=lambda c: self._fitness(c, vehicles, depot))

    def _biased_selection(
        self, population: List[List[VRPInstance]], vehicles: int, depot: VRPInstance
    ) -> List[VRPInstance]:
        """
        Select an individual using fitness-proportional (biased) selection.

        :param population: List of individuals
        :param vehicles: Number of vehicles
        :param depot: Depot instance
        :return: Selected individual
        """
        inv_fits = [
            1.0 / (self._fitness(c, vehicles, depot) + 1e-9) for c in population
        ]
        total = sum(inv_fits)
        pick = random.random() * total
        cum = 0.0
        for c, w in zip(population, inv_fits):
            cum += w
            if cum >= pick:
                return c
        return population[-1]

    def _crossover(self, p1: List[VRPInstance], p2: List[VRPInstance]) -> List[VRPInstance]:
        """
        Perform ordered crossover (OX) between two parents.

        :param p1: Parent 1
        :param p2: Parent 2
        :return: Child individual
        """
        size = len(p1)
        a, b = sorted(random.sample(range(size), 2))
        child = [None] * size
        child[a:b] = p1[a:b]
        pos = b
        for city in p2[b:] + p2[:b]:
            if city not in child:
                if pos >= size:
                    pos = 0
                child[pos] = city
                pos += 1
        return child

    def _rbx_crossover(
        self, p1: List[VRPInstance], p2: List[VRPInstance], vehicles: int
    ) -> List[VRPInstance]:
        """
        Perform route-based crossover (RBX) between two parents.

        :param p1: Parent 1
        :param p2: Parent 2
        :param vehicles: Number of vehicles
        :return: Child individual
        """
        r1 = self._split_into_routes(p1, vehicles)
        r2 = self._split_into_routes(p2, vehicles)
        take1 = random.sample(r1, vehicles // 2)
        take2 = random.sample(r2, vehicles - len(take1))
        child_routes = take1 + take2
        used = set()
        final = []
        for route in child_routes:
            filt = [c for c in route if c not in used]
            used |= set(filt)
            final.append(filt)
        leftovers = [c for c in p1 if c not in used]
        idx = 0
        for c in leftovers:
            final[idx % vehicles].append(c)
            idx += 1
        return [city for route in final for city in route]

    def _mutate(self, chrom: List[VRPInstance]) -> List[VRPInstance]:
        """
        Mutate an individual by swapping cities with a given probability.

        :param chrom: Individual to mutate
        :return: Mutated individual
        """
        for i in range(len(chrom)):
            if random.random() < self.mutation_rate:
                j = random.randrange(len(chrom))
                chrom[i], chrom[j] = chrom[j], chrom[i]
        return chrom

    def _two_opt_route(
        self, route: List[VRPInstance], depot: VRPInstance, max_swaps: int = 1
    ) -> List[VRPInstance]:
        """
        Apply 2-opt local search to a single route.

        :param route: Route to optimize
        :param depot: Depot instance
        :param max_swaps: Maximum number of swaps
        :return: Optimized route
        """
        best = route
        swaps = 0
        for i in range(1, len(best) - 2):
            if swaps >= max_swaps:
                break
            for j in range(i + 1, len(best) - 1):
                new = best[:i] + best[i : j + 1][::-1] + best[j + 1 :]
                if self._distance(new, depot) < self._distance(best, depot):
                    best = new
                    swaps += 1
                    break
        return best

    def _relocate(self, routes: List[List[VRPInstance]], depot: VRPInstance) -> List[List[VRPInstance]]:
        """
        Try to relocate cities between routes to improve the solution.

        :param routes: List of routes
        :param depot: Depot instance
        :return: Modified routes
        """
        n_veh = len(routes)
        for _ in range(self.relocate_samples * n_veh):
            i = random.randrange(n_veh)
            if len(routes[i]) <= 2:
                continue
            j = random.randrange(1, len(routes[i]) - 1)
            city = routes[i][j]
            k = random.randrange(n_veh)
            if k == i:
                continue
            pos = random.randrange(1, len(routes[k]))
            r1 = routes[i][:j] + routes[i][j + 1 :]
            r2 = routes[k][:pos] + [city] + routes[k][pos:]
            gain = (self._distance(r1, depot) + self._distance(r2, depot)) - (
                self._distance(routes[i], depot) + self._distance(routes[k], depot)
            )
            if gain < 0:
                routes[i].pop(j)
                routes[k].insert(pos, city)
        return routes

    def _exchange(self, routes: List[List[VRPInstance]], depot: VRPInstance) -> List[List[VRPInstance]]:
        """
        Try to exchange cities between routes to improve the solution.

        :param routes: List of routes
        :param depot: Depot instance
        :return: Modified routes
        """
        n_veh = len(routes)
        for _ in range(self.relocate_samples * n_veh):
            i = random.randrange(n_veh)
            j = random.randrange(1, len(routes[i]) - 1) if len(routes[i]) > 2 else None
            k = random.randrange(n_veh)
            if k <= i or j is None or len(routes[k]) <= 2:
                continue
            l = random.randrange(1, len(routes[k]) - 1)
            c1, c2 = routes[i][j], routes[k][l]
            r1 = routes[i][:j] + [c2] + routes[i][j + 1 :]
            r2 = routes[k][:l] + [c1] + routes[k][l + 1 :]
            gain = (self._distance(r1, depot) + self._distance(r2, depot)) - (
                self._distance(routes[i], depot) + self._distance(routes[k], depot)
            )
            if gain < 0:
                routes[i][j], routes[k][l] = c2, c1
        return routes

    def solve(
        self, csv_path: str, config_path: str, output_file_path: str
    ) -> List[List[VRPInstance]]:
        """
        Run the genetic algorithm to solve the VRP and save the results.

        :param csv_path: Path to the CSV file with city data
        :param config_path: Path to the JSON config file
        :param output_file_path: Path to save the results
        :return: List of final routes (each route is a list of VRPInstance)
        """
        logger.info("GA solve start")
        t0 = time.time()
        data = self.load_dataset(csv_path, config_path)
        cities, vehicles, depot = data.cities, data.vehicles, data.depot
        logger.info("Data: %d cities, %d vehicles", len(cities), vehicles)

        # initial population
        seeds = self._seed_population(cities, depot)
        rand = [
            self._create_individual(cities)
            for _ in range(self.population_size - len(seeds))
        ]
        population = seeds + rand

        best = min(population, key=lambda c: self._fitness(c, vehicles, depot))
        best_fit = self._fitness(best, vehicles, depot)

        for gen in range(1, self.generations + 1):
            logger.info("Generation %s", gen)

            # generate all offspring
            offspring = []
            while len(offspring) < self.population_size:
                if self.use_biased:
                    p1 = self._biased_selection(population, vehicles, depot)
                    p2 = self._biased_selection(population, vehicles, depot)
                else:
                    p1 = self._tournament_selection(population, vehicles, depot)
                    p2 = self._tournament_selection(population, vehicles, depot)

                if random.random() < 0.5:
                    child = self._rbx_crossover(p1, p2, vehicles)
                else:
                    child = self._crossover(p1, p2)
                self._mutate(child)
                offspring.append(child)

            # evaluate
            scored = [(self._fitness(c, vehicles, depot), c) for c in offspring]
            scored.sort(key=lambda x: x[0])
            elite_count = max(1, int(self.elite_fraction * self.population_size))
            elites = [c for _, c in scored[:elite_count]]
            others = [c for _, c in scored[elite_count:]]

            # memetic local search on elites
            new_pop = []
            for c in elites:
                routes = self._split_into_routes(c, vehicles)
                routes = [self._two_opt_route(r, depot) for r in routes]
                routes = self._relocate(routes, depot)
                routes = self._exchange(routes, depot)
                new_pop.append([city for r in routes for city in r])

            # fill up with rest without LS
            new_pop += others[: self.population_size - len(new_pop)]
            population = new_pop

            # update best
            cur, fit = min(
                [(c, self._fitness(c, vehicles, depot)) for c in population],
                key=lambda x: x[1],
            )
            if fit < best_fit:
                best, best_fit = deepcopy(cur), fit

        elapsed = time.time() - t0
        km = best_fit / 1000
        logger.info("GA done in %.2f s, distance=%.2f km", elapsed, km)

        final_routes = self._split_into_routes(best, vehicles)
        final_routes = [[depot] + r + [depot] for r in final_routes]
        self.save_results_to_file(
            km, elapsed, final_routes, self.vehicle_info, output_file_path
        )
        return final_routes