import joblib
from helpers import *


df = pd.read_csv("house_price_prediction.csv")
new_model = joblib.load("gbm_final.pkl")


X, y = data_preb(df)

random_user = X.sample(1)

new_model.predict(random_user)

y[1379]