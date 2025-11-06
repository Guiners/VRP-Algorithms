# VRP Algorithms

A collection of algorithms and tools to solve Vehicle Routing Problems (VRP). This repository contains implementations, datasets and result exports for multiple heuristics and metaheuristics.

## Key features

- Implementations of classic VRP algorithms and heuristics (e.g. nearest neighbor, Clarke\-Wright savings, genetic algorithms, GRASP)
- Dataset collection with different sizes (`datasets/small`, `datasets/medium`, `datasets/large`, `datasets/xlarge`, `datasets/raw`)
- Scripts and utilities under `src/` for running algorithms and analysing results
- Precomputed results and charts in `results/`

## Requirements

- Python 3.11
- Recommended: [Poetry](https://python-poetry.org/) to manage dependencies

Main dependencies (from `pyproject.toml`): `tsplib95`, `logger`. Development tools listed: `black`, `isort`, `pylint`.

## Installation (Windows)

1. Clone the repository:
   - `git clone https://github.com/Guiners/VRP-Algorithms`
   - `cd VRP-Algorithms`

2. Install dependencies with Poetry:
   - `poetry install`
   - Optionally enter the virtual environment: `poetry shell`

If you do not use Poetry, create and activate a venv and install required packages manually.

## Usage

- Entry point: `src/main.py`

Example (using Poetry):
- `poetry run python src\main.py --input datasets\small\vpr101_3_1.csv --algorithm nearest_neighbor`

Notes:
- Available algorithms correspond to folders in `src\algorithms\` and to result folders in `results\` (e.g. `nearest_neighbor_algorithm`, `clarke_wright_savings`, `genetic_algorithm`, `grasp`).
- Many scripts accept dataset file paths from the `datasets/` directory. Adjust CLI arguments according to the implemented options in `src/main.py`.

## Project structure (high level)

- `src/` \- source code
  - `src\main.py` \- main entry point
  - `src\algorithms\` \- algorithm implementations
  - `src\data_analyst\` \- result analysis and plotting
  - `src\utils\` \- utility helpers
- `datasets/` \- problem instances (small / medium / large / xlarge / raw)
- `results/` \- saved outputs, charts and tables
- `pyproject.toml` \- project metadata and dependencies
- `README.md` \- this file

## Datasets

Datasets are provided in the `datasets/` folder and grouped by size:
- `datasets/small`, `datasets/medium`, `datasets/large`, `datasets/xlarge`, `datasets/raw`

Each dataset typically has a CSV and a JSON description (for example `datasets/small/vpr101_3_1.csv` and `datasets/small/vpr101_3_1.json`).

## Results and charts

Precomputed results and visualizations are available under `results/`. Subfolders group outputs per algorithm and per dataset size.

## Contributing

- Open issues for bugs or feature requests.
- Pull requests are welcome. Follow repository code style and run linters/formatters if configured.

## Tests

- No automated tests included in the repository root. Add a tests folder and CI configuration to enable test runs.

## License

- No license file found. Add a `LICENSE` file to define terms for reuse.
