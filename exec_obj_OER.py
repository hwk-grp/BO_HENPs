from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
import numpy as np
import pandas as pd
from sklearn.gaussian_process.kernels import Matern, RBF
from sklearn.gaussian_process import GaussianProcessRegressor


# init_points = 0
n_iter = 200
random_state_init = 1
n_repeat = 10

# read data
# 1st round
df0 = pd.read_excel('data/metal8_OER_50.xlsx')
# df0 = pd.read_excel('data/metal8_HER_60.xlsx')
raw_data = np.array(df0)
x = raw_data[:, :8] # # of features = 8,
y = raw_data[:, 8]
n_data = len(x[:, 0])

bo_cats = list()

for repeat in range(0, n_repeat):

    random_state = random_state_init + repeat

    # GPR model (consistent with internal gp in bayesian_optimization.py)
    model = GaussianProcessRegressor(
        kernel=Matern(length_scale=0.1, length_scale_bounds=(1e-1, 1e2), nu=0.5),
        alpha=1e-6,
        normalize_y=True,
        n_restarts_optimizer=100,
        random_state=random_state,
    )

    model.fit(x, y)

    # Create a BayesianOptimization Object
    def model_pred(x1, x2, x3, x4, x5, x6, x7, x8):
        z = np.hstack([x1, x2, x3, x4, x5, x6, x7, x8]).reshape(1, -1)
        # return 1.0 / (model.predict(z).item() + 1e-5)  # minimization
        return 1.0 / (model.predict(z).item() + 1e-5)  # minimization

    optimizer = BayesianOptimization(f=model_pred, pbounds={'x1': (0.005, 0.05), 'x2': (0.005, 0.05), 'x3': (0.005, 0.05),
                                                            'x4': (0.005, 0.05), 'x5': (0.005, 0.05), 'x6': (0.005, 0.05),
                                                            'x7': (0.005, 0.05), 'x8': (0.005, 0.05),
                                                          },
                                     allow_duplicate_points=True, random_state=random_state)

    # Acquisition function: Upper Confidence Bound
    # The parameter 'kappa' controls the balance between exploration and exploitation
    acq_function = UtilityFunction(kind="ucb", kappa=5)
    # acq_function = UtilityFunction(kind="ei", xi=1e+1)
    for n in range(0, n_data):
        optimizer.probe(params={'x1': x[n, 0], 'x2': x[n, 1], 'x3': x[n, 2], 'x4': x[n, 3],
                                'x5': x[n, 4], 'x6': x[n, 5], 'x7': x[n, 6], 'x8': x[n, 7],
                                }, lazy=True,)

    optimizer.maximize(init_points=0, n_iter=n_iter, acquisition_function=acq_function)

    list_target = list()
    for iter in range(0, n_iter):
        list_target.append(optimizer.res[n_data + iter]['target'])

    max_ndx = np.argmax(list_target)

    # print(optimizer)
    x_max = list()
    for i in range(0, 8):
        x_max.append(optimizer.res[n_data + max_ndx]['params']['x{}'.format(i+1)])
    bo_cats.extend(x_max)
    x_max = np.array(x_max).reshape(1, -1)

    _, std = model.predict(x_max, return_std=True)
    bo_cats.append( 1.0 / optimizer.res[n_data + max_ndx]['target'] - 1e-5)
    bo_cats.append(std.item())

bo_cats = np.array(bo_cats).reshape(-1, 10)
df1 = pd.DataFrame(bo_cats,
                   columns=list(df0.columns) + ['uncertainty'])
df1.to_excel("suggested_cats_OER_1st.xlsx", index=False)
# df1.to_excel("suggested_cats_HER_2nd.v2.xlsx", index=False)