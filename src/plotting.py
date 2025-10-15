import matplotlib.pyplot as plt
import seaborn as sns

def plot_clusters(df):
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=df, x="longitude", y="latitude", hue="cluster_id", palette="tab20", s=30)
    plt.title("Spatio-Temporal Clusters of Rainfall")
    plt.show()

def plot_inflow_pred(y_true, y_pred):
    plt.figure(figsize=(8,5))
    plt.plot(y_true.values, label="True", linewidth=2)
    plt.plot(y_pred, label="Predicted", linewidth=2)
    plt.legend()
    plt.title("Inflow Prediction (Random Forest)")
    plt.show()
