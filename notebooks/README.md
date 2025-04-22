# Notebooks Folder

This folder contains Jupyter notebooks for interactive data analysis and experimentation in the Docker Data Science project.

## Contents

- `01_data_exploration.ipynb`: A notebook for exploratory data analysis techniques. This notebook shows how to load data, create visualizations, and perform initial statistical analysis within the Docker environment.
- `02_model_training.ipynb`: A notebook for training and evaluating machine learning models on the prepared data.

## Usage

Notebooks in this folder can be run through:

1. VS Code with the Jupyter extension (when connected to the Docker container)
2. Jupyter Lab/Notebook interface by accessing http://localhost:8888 in your browser
3. Direct execution through the Python kernel in the container

## Benefits of Containerized Notebooks

- Consistent execution environment
- Reproducible analysis across different systems
- Pre-configured dependencies
- Isolation from system-level conflicts

## Best Practices

- Keep notebooks organized and focused on specific tasks
- Include markdown cells to document your analysis
- Remember to save outputs before shutting down containers
- Consider converting finished notebooks to Python scripts for production use
- Use relative paths for accessing data (e.g., `../data/sample.csv`) 