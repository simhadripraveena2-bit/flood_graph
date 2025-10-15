# **Flood Graph Project**
##### **Spatio-Temporal Graph Modeling and Prediction of Rainfall and Inflow Patterns**

## **Overview**
This project explores spatio-temporal analysis and graph-based modeling of rainfall and inflow patterns using real-world hydrological data.

It demonstrates a full machine learning and graph neural network pipeline from preprocessing and clustering to baseline prediction and GNN-based temporal forecasting.

You can use this project to:

* Detect spatially and temporally coherent rainfall clusters
* Construct region-wise rainfall graphs
* Predict inflow based on rainfall distribution and time evolution
* Experiment with Graph Neural Networks (GNNs) for scientific modeling

## **Project Structure**
flood_prediction/
* data/ 
  * newflood.xlsx    # Input dataset (2 sheets: rainfall + inflow)
* src/ 
  * preprocessing.py 
    * Read & transform both rainfall and inflow sheets 
    * Extracts rainfall grid, reshapes into tidy DataFrame (year, month, day, lat, lon, intensity)
  * clustering.py 
    * Performs spatio-temporal DBSCAN to identify coherent rainfall clusters
  * graph_utils.py 
    * Builds a spatial adjacency graph using nearest-neighbor relationships
  * models.py 
    * Uses RandomForest for inflow regression based on rainfall clusters
  * gnn_model.py 
    * Implements Graph Neural Network (GCN) to learn spatio-temporal dependencies
  * evaluation.py 
    * Metrics for baseline
  * plotting.py 
    * Plots spatial clusters, rainfall intensity maps, correlation trends
* main.py
  * runs full pipeline
* requirements.txt
* README.md

## **Dataset Description**
data/newflood.xlsx contains two sheets:

#### **1. Sheet 1 – Rainfall Grid**
* First row -> Latitude values
* Second row -> Longitude values
* Subsequent rows -> year, month, day, intensity grid
* Each cell (lat, lon) represents rainfall intensity on that date.

#### **2. Sheet 2 – Inflow Data**
* Columns: Date, Inflow (cumecs)
* Used as target variable for prediction and correlation analysis.

## **Example Workflow**
Run the entire pipeline:

_python main.py_

This will:
* Load and preprocess both sheets from data/newflood.xlsx
* Perform spatio-temporal clustering
* Build a rainfall graph
* Train a RandomForest model for inflow prediction
* Train a GNN for spatio-temporal forecasting (optional)
* Give evaluation metrics and plots

## **Graph Neural Network (GNN)**
The GNN uses PyTorch Geometric to model spatio-temporal dependencies between rainfall locations.
It learns how rainfall at connected locations propagates to influence inflow intensity over time.

### **Architecture**
* GraphConv -> ReLU -> Dropout -> Linear
* Edge weights are based on spatial proximity or correlation.
* Optimized with MSELoss for inflow prediction.

### **Evaluation Metrics**
* Mean Squared Error (MSE)
* Mean Absolute Error (MAE)
* R² Score

Run GNN training:

_python main.py --gnn_

## **Visualization Samples**
* Spatio-Temporal Clusters
* Graph Connectivity
* Prediction Comparison

## **Requirements**
All dependencies are listed in requirements.txt

Install them using:

_pip install -r requirements.txt_

## Research Relevance
This project is aligned with Climate Informatics, Data-Driven Hydrology, and Spatio-Temporal Machine Learning research.

Potential research directions:
* Graph Neural Networks for rainfall–runoff modeling
* Cluster-based hydrological forecasting
* Transfer learning for regional flood prediction
* Uncertainty quantification in graph-based predictions

## **Author**
#### **Simhadri Praveena**
#### **Department of Computer Science and Engineering**
#### **IIT Kharagpur**
#### **Research Interests: Machine Learning, Data Science**