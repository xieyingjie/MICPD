from statsmodels.tsa.statespace.varmax import VARMAX

import detectCP
import simulate
import utils
import numpy as np
import argparse



parser = argparse.ArgumentParser(description='interface of running experiments for TSCP2 baselines')
parser.add_argument('--datapath', type=str, required=True, help='[ ./data ] prefix path to data directory')
window_size=3
DATA_PATH = args.datapath
test_ds = load_dataset(DATA_PATH, mode = "test")
timeseries, windows, parameters = simulate.generate_jumpingmean(window_size)
model = VARMAX(endog=windows)
model_fit = model.fit()

prediction = model_fit.forecast(model_fit.y, steps=len(test_ds))

dissimilarities = detectCP.smoothened_dissimilarity_measures(prediction, windows, window_size)
change_point_scores = detectCP.change_point_score(dissimilarities, window_size)

np.savetxt("change_point_scores.txt", change_point_scores)

print(prediction)