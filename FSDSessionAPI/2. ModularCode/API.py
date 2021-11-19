import numpy as np
import pandas as pd
import pickle
from config import *

df = pd.read_csv(testdata)

model = pickle.load(open(modeloutput, "rb"))

prediction = model.predict(df[features])
print(prediction)
