# High-Frequency Limit Order Book Dynamics

This project simulates and analyzes Limit Order Book (LOB) dynamics using high-frequency trading strategies and statistical models. It includes a real-time dashboard, backtesting engine, and implementations of the Avellaneda-Stoikov market making strategy and Hawkes processes.

## Features

- **Real-time LOB Dashboard**: Visualize order book snapshots, spread evolution, and market depth.
- **Avellaneda-Stoikov Strategy**: Implementation of the classic market-making strategy.
- **Hawkes Process Modeling**: Simulation and analysis of order arrival intensities.
- **Backtesting Engine**: robust backtesting framework for strategy evaluation.
- **Risk Metrics**: Real-time calculation of spread, volatility, and PnL.

## Prerequisites

- Python 3.10 or higher
- pip (Python package installer)

## Installation

1. Clone the repository (if you haven't already):
   ```bash
   git clone <repository-url>
   cd High-Frequency-Limit-Order-Book-Dynamics-PERSONAL
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Project

### Dashboard

To launch the interactive Streamlit dashboard:

**Using the provided script (Linux/Mac/Git Bash):**
```bash
./run_dashboard.sh
```

**Using Python directly (Windows/Universal):**
```bash
python -m streamlit run dashboard/app.py
```
The dashboard will open in your default web browser.

### Backtesting

To run a batch backtest on multiple simulated stocks:

```bash
python src/backtesting/batch_runner.py
```
This will generate a `batch_backtest_results.csv` file with the performance metrics.

## Testing and Verification

To verify that the core components (Strategy, Engine, Visualization) are working correctly, run the verification script:

```bash
python src/verify_day3.py
```

## Project Structure

- `src/`: Core source code.
  - `strategy/`: Strategy implementations (e.g., Avellaneda-Stoikov).
  - `backtesting/`: Backtesting engine and optimizer.
  - `models/`: Statistical models (e.g., Hawkes processes).
  - `visualization/`: Plotting functions.
  - `data_pipeline/`: LOB data structure and simulation.
- `dashboard/`: Streamlit dashboard application.
- `tests/`: Unit tests.