import warnings
import pandas as pd
import joblib
from sklearn.exceptions import ConvergenceWarning


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)


from helpers import *
from config import *


def main():
    df = pd.read_csv("house_price_prediction.csv")
    X, y = data_preb(df, num_method="median", cat_length=17)
    final_model = hyperparameter_optimization(X, y)
    joblib.dump(final_model, "gbm_final.pkl")
    return final_model


if __name__ == "__main__":
    print("Process is beginning.")
    main()

