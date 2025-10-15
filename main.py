import argparse

from src.preprocessing import load_and_process
from src.clustering import spatio_temporal_clustering
from src.plotting import plot_clusters, plot_inflow_pred
from src.graph_utils import build_spatial_graph
from src.models import train_baseline
from src.evaluation import evaluate_model
from src.gnn_model import train_gnn
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gnn", action="store_true", help="Run GNN model instead of baseline")
    args = parser.parse_args()

    df_rain, df_inflow = load_and_process("data/newflood.xlsx")

    print("Data loaded:", df_rain.shape, df_inflow.shape)
    print("Data loaded:", df_rain.head(), df_inflow.head())
    df_clustered, _ = spatio_temporal_clustering(df_rain)
    plot_clusters(df_clustered)

    G = build_spatial_graph(df_clustered)
    print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # model, X_test, y_test, preds = train_baseline(df_rain, df_inflow)
    # metrics = evaluate_model(y_test, preds)
    # print("Model performance:", metrics)
    #
    # plot_inflow_pred(y_test, preds)

    if args.gnn:
        print("Training GNN model...")
        model, metrics, data = train_gnn(df_rain, df_inflow, G)
        print("GNN Performance Metrics:", metrics)
    else:
        print("Running Random Forest baseline...")
        model, X_test, y_test, preds = train_baseline(df_rain, df_inflow)
        metrics = evaluate_model(y_test, preds)
        print("Baseline Performance:", metrics)
        plot_inflow_pred(y_test, preds)

if __name__ == "__main__":
    main()
