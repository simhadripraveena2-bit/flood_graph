import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def train_baseline(df_rain, df_inflow):
    """
    Aggregates rainfall by date and predicts inflow.
    """
    agg = df_rain.groupby("date")["intensity"].mean().reset_index()
    merged = pd.merge(df_inflow, agg, on="date", how="inner")

    X = merged[["intensity"]]
    y = merged["inflow"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    return model, X_test, y_test, preds
