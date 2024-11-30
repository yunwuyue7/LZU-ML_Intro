# README
This repository demonstrates how to classify the Iris dataset using Least Squares Classifier (LSC) and Fisher Linear Discriminant (FLD).

## Prerequisites

- `Python 3.x`
- `pip`

## Installation
1. Clone the repository:
   ```powershell
   git clone https://github.com/yourusername/iris-classification.git
   cd LSC

2. Install the required dependencies：
   ```powershell
   pip install -r requirements.txt

## Usage
1. Run the filtering script to generate the filtered data:
   ```powershell
   python filter.py

2. Run the main script to read the filtered data, perform classification, and plot the results:

   ```powershell
   python main.py
   ```

## Files
- `filter.py`: Filters the Iris dataset to include only the two most common species.
- `main.py`: Reads the filtered data, performs classification using LSC and FLD, and plots the results.
- `iris`: A folder containing the Iris dataset files.
- `requirements.txt`: List of required Python packages.
- `README.md`: This file.

## Dependencies

- `pandas`: For data manipulation.
- `matplotlib`: For plotting.
- `numpy`: For numerical operations.