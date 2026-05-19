# Customer Segmentation using RFM Analysis & K-Means Clustering

A portfolio-grade customer segmentation project that uses RFM analysis and K-Means clustering to identify high-value, at-risk, and loyal customers from retail transaction data.

## Project Overview

This repository contains a complete end-to-end analysis pipeline for customer segmentation:

- Data loading and cleaning
- Exploratory data analysis (EDA)
- Recency, Frequency, and Monetary (RFM) scoring
- K-Means clustering for customer segmentation
- Cluster profiling and business recommendations

## Files

- `Customer_Segmentation_RFM_KMeans.ipynb` - Full notebook with visual storytelling and analysis.
- `customer_segmentation_analysis.py` - Reusable script to run the segmentation pipeline and export results.
- `requirements.txt` - Python packages required for the project.
- `.gitignore` - Files to exclude from Git.
- `data/README.md` - Dataset placement instructions.

## Dataset

Download the retail dataset from Kaggle:

- https://www.kaggle.com/datasets/sidharthxmalik/customer-segmentation-retail-data

Place the dataset in the `data/` folder as `online_retail_ii.xlsx`.

## How to use

1. Create a Python environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the dataset and place it at `data/online_retail_ii.xlsx`.
4. Run the Python script:
   ```bash
   python customer_segmentation_analysis.py
   ```
5. Open `Customer_Segmentation_RFM_KMeans.ipynb` to review the notebook analysis.

## Business Impact

This project helps retail teams identify:

- high-value customers for VIP programs
- loyal customers for reward campaigns
- at-risk customers for reactivation messaging
- low-value segments for efficient marketing spend

