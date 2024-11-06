from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV,StratifiedKFold

from src.models.classifier import SklearnClassifier
from src.utils.config import load_config
from src.utils.guardrails import validate_evaluation_metrics
from src.utils.store import AssignmentStore
from lightgbm import LGBMClassifier
import numpy as np

@validate_evaluation_metrics
def main():
    store = AssignmentStore()
    config = load_config()

    df = store.get_processed("transformed_dataset.csv")

    df_train, df_test = train_test_split(df, test_size=config["test_size"])


    param_dist = {
        'n_estimators': [50, 100, 200],  # Number of boosting rounds (trees)
        'max_depth': [None, 10, 20],  # Maximum depth of the trees
        'learning_rate': [0.01, 0.05, 0.1],  # Learning rate
        'num_leaves': [31, 50, 100],  # Number of leaves in the tree
        'min_child_samples': [20, 50],  # Minimum number of data points in a leaf
        'subsample': [0.7, 0.8, 1.0],  # Fraction of samples used for training
        'colsample_bytree': [0.7, 0.8, 1.0],  # Fraction of features used
    }
        
    # Initialize LGBMClassifier
    lgbm_estimator = LGBMClassifier(class_weight='balanced')

    # Perform RandomizedSearchCV for hyperparameter tuning
    random_search = RandomizedSearchCV(
        lgbm_estimator, 
        param_distributions=param_dist, 
        n_iter=10,  # Number of parameter settings sampled
        cv=StratifiedKFold(5),  # 5-fold cross-validation
        random_state=42,
        n_jobs=1  # Limit parallel processing to 1 job
    )


    # Fit RandomizedSearchCV to the training data
    random_search.fit(df_train[config["features"]], df_train[config["target"]])
    
    # Get the best estimator from the RandomizedSearchCV
    best_lgbm_estimator = random_search.best_estimator_

    # Initialize the SklearnClassifier with the best LGBM model
    model = SklearnClassifier(best_lgbm_estimator, config["features"], config["target"])
    model.train(df_train)

    # Evaluate the model on the test set
    metrics = model.evaluate(df_test)

    # Save the best model and the evaluation metrics
    store.put_model("saved_model.pkl", model)
    store.put_metrics("metrics.json", metrics)

    # Output the best hyperparameters
    print(f"Best hyperparameters found: {random_search.best_params_}")
    print(f"Best cross-validation score: {random_search.best_score_}")


if __name__ == "__main__":
    main()

