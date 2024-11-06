import sys
import os

# Get the project root directory and add it to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../"))  # Go up two levels
print(current_dir,root_dir)
sys.path.append(root_dir)


#%%
import numpy as np
import pandas as pd
from src.features.build_features import apply_feature_engineering
from src.utils.guardrails import validate_prediction_results
from src.utils.store import AssignmentStore


# Define path to your serialized model and preprocessing pipeline
model_path = os.path.join(root_dir,"models", "saved_model.pkl")
pipeline_path = os.path.join(root_dir,"models", "preprocessing_pipeline.pkl")

# Load the model and preprocessing pipeline
def main():
    store = AssignmentStore()

    df_test = store.get_raw("test_data.csv")
    df_test = apply_feature_engineering(df_test)

    model = store.get_model("saved_model.pkl")
    df_test["score"] = model.predict(df_test)

    selected_drivers = choose_best_driver(df_test)
    store.put_predictions("results.csv", selected_drivers)


def choose_best_driver(df: pd.DataFrame) -> pd.DataFrame:
    df = df.groupby("order_id").agg({"driver_id": list, "score": list}).reset_index()
    df["best_driver"] = df.apply(
        lambda r: r["driver_id"][np.argmax(r["score"])], axis=1
    )
    df = df.drop(["driver_id", "score"], axis=1)
    df = df.rename(columns={"best_driver": "driver_id"})
    print("Columns in dataset during inference:", df.columns)
    return df


if __name__ == "__main__":
    main()