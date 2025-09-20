# Amazon Reviews Anomaly Detection Project

A project for detecting anomalous reviews in Amazon product datasets using DBSCAN clustering, advanced feature engineering and a sentiment model. 

## Project Overview

This project includes:
- Loading and preprocessing Amazon Reviews 2023 dataset
- Feature engineering for anomaly detection
- DBSCAN-based anomaly detection with parameter optimization
- Comprehensive evaluation and visualization of results
- Analysis of coordinated attacks and user behavior patterns

## Project Structure

### Root Files

#### `requirements.txt`
Python dependencies file containing all required packages.

### Source Code (`src/`)

#### Core Anomaly Detection Scripts

##### `db_scan_anomaly_detection.py`
Main DBSCAN anomaly detection implementation featuring:
- **DBScanAnomalyDetector class**: Memory-optimized DBSCAN clustering
- **Batch processing**: Handles large datasets with configurable batch sizes
- **Memory monitoring**: Tracks memory usage during processing
- **Noise point sorting**: Ranks anomalies by distance to core points
- **Feature engineering**: Uses 6 key features for detection

##### `dbscan_parameter_selection.py`
Parameter optimization tool for DBSCAN.

##### `scan_utils.py`
Utility functions for data processing.

##### `evaluate_db_scan_anomaly_detection.py`
Comprehensive evaluation framework:
- **AnomalyDetectionEvaluator class**: Complete evaluation pipeline
- **Multi-dimensional analysis**: Basic metrics, user patterns, coordinated attacks
- **Report generation**: Automated text and JSON reports

#### Data Creation Pipeline (`src/data_creation/`)

##### `amazon_reviews_loader.py`
Amazon Reviews 2023 dataset loader:
- **AmazonReviews2023Loader class**: Handles Hugging Face dataset loading
- **Multiple categories**: Supports all 34 Amazon categories

##### `create_subsets.py`
Dataset creation and preprocessing pipeline:
- **Multi-category processing**: Handles multiple product categories
- **Text cleaning**: Removes HTML, normalizes whitespace
- **Token filtering**: Limits reviews by token count (max 512)
- **Data validation**: Ensures no null values in critical fields
- **Duplicate removal**: Removes duplicate reviews
- **Train/val/test splits**: Creates balanced datasets

##### `enhance_data_with_more_features.py`
Feature enhancement script:
- **Product metadata integration**: Adds product average ratings
- **Rating deviation calculation**: Computes rating vs. product average
- **User behavior features**: Adds reviewer review counts

#### Analysis and Visualization (`src/analysis/`)

##### `basic_analysis.py`
Core analysis and visualization functions.

##### `user_analysis_visualization.py`
User behavior analysis.


##### `coordinated_attacks_visualization.py`
Advanced coordinated attack detection.

#### Machine Learning Model (`src/model/`)
- Contains trained model files (`.model` and `.safetensors` formats)
- Used for rating prediction in feature engineering

#### `regression_model.ipynb`
Google Colab notebook for model training.

#### `db_scan.ipynb`
Jupyter notebook for DBSCAN anomaly detection experimentation and analysis.

### Results and Evaluation (`evaluation_results/`)

Organized by category and parameter combinations.

Each result directory contains:
- **`evaluation_metrics.json`** - Complete numerical results and statistics
- **`evaluation_report.txt`** - Summary report
- **`plots/`** - Visualization subdirectories:
  - `basic_analysis/` - Feature distributions and comparisons
  - `user_analysis/` - User behavior visualizations
  - `coordinated_attacks/` - Spam detection plots

### Output Files (`output/`)

DBSCAN anomaly detection results.

### Parameter Selection Plots (`k-distance_plots/`)

K-distance analysis plots for parameter optimization:
- `books.png` - Books category k-distance plot
- `clothing_shoes_and_jewerly.png` - Clothing category k-distance plot
- `electronics.png` - Electronics category k-distance plot

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Data Preparation

Create datasets from Amazon Reviews 2023:
```bash
cd src/data_creation
python create_subsets.py --categories Books Electronics Clothing_Shoes_and_Jewelry --train-samples 10000 --test-samples 30000
```

Enhance with additional features:
```bash
python enhance_data_with_more_features.py data/test/books_test.csv Books
```

### 3. Parameter Selection

Generate k-distance plots for parameter optimization:
```bash
cd src
python dbscan_parameter_selection.py --csv-path data/test/books_test.csv --plot k-distance_plots/books.png
```

### 4. Anomaly Detection

Run DBSCAN anomaly detection:
```bash
python db_scan_anomaly_detection.py --csv-path data/test/books_test.csv --eps 0.6 --min-samples 12 --batch-size 100000 --out output/books_test_scan_anomalies.csv
```

Or use the Jupyter notebook:
```bash
jupyter notebook db_scan.ipynb
```

### 5. Evaluation and Analysis

Generate comprehensive evaluation:
```bash
python evaluate_db_scan_anomaly_detection.py --anomaly-data-file output/books_test_scan_anomalies_eps_0.6_min_samples_12_batch_size_100000.csv --original-data data/test/books_test.csv --category books --output-dir evaluation_results
```