import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from gryffin import Gryffin
import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# TODO
# - setup github repo
# - investigate negative values
# - update configuration
# - unblock graphing


def em(xA, T):
    mA = 0.567  # Sc
    mB = 0.52  # Ti
    xB = 1-xA
    em = ((xA*mA)-(xB*mB))-(((xA*xB)*(xB*mA) /
                             (xA*mB))-((xB*xA)*(xA*mB)/(xB*mA)))*T
    return em


def compute_objective(param):
    xA = param['xA']
    T = param['T']
    param['obj'] = em(xA, T)
    return param


# TODO configure gryffin
config = {
    "general": {
        "random_seed": 42,
        "verbosity": 0,
        "boosted":  False,
    },
    "parameters": [
        {"name": "xA", "type": "continuous", "low": 0.01, "high": 0.99},
        {"name": "T", "type": "continuous", "low": 273, "high": 1273},
    ],
    "objectives": [
        {"name": "obj", "goal": "max"},
    ]
}

gryffin = Gryffin(config_dict=config)

observations = []
MAX_ITER = 20

for num_iter in range(MAX_ITER):
    print('-'*20, 'Iteration:', num_iter+1, '-'*20)

    # Query for new parameters
    params = gryffin.recommend(
        observations=observations,
    )

    param = params[0]
    print('  Proposed Parameters:', param, end=' ')

    # Evaluate the proposed parameters.
    observation = compute_objective(param)
    print('==> eM:', observation['obj'])

    # Append this observation to the previous experiments
    observations.append(param)

x_domain = np.linspace(0.01, 1.)
y_domain = np.linspace(273., 1273.)
X, Y = np.meshgrid(x_domain, y_domain)
Z = np.zeros((x_domain.shape[0], y_domain.shape[0]))

# for x_index, x_element in enumerate(x_domain):
#     for y_index, y_element in enumerate(y_domain):
#         print('==> xA: ', x_element)
#         print('==> T: ', y_element)
#         em = em(x_element, y_element)
#         Z[y_index, x_index] = em

# fig, ax = plt.subplots()
# contours = plt.contour(X, Y, Z, 3, colors='black')
# ax.clabel(contours, inline=True, fontsize=8)
# ax.imshow(Z, extent=[0, 1, 0, 1], origin='lower', cmap='RdGy', alpha=0.5)
# _ = ax.set_title('Heat Capacity')
# _ = ax.set_xlabel('Composition')
# _ = ax.set_ylabel('Temp')

# # observed parameters and objectives
# samples_x = [obs['x_0'] for obs in observations]
# samples_y = [obs['x_1'] for obs in observations]
# samples_z = [obs['obj'] for obs in observations]

# _ = plt.scatter(samples_x, samples_y, zorder=10, s=150,
#                 color='r', edgecolor='#444444', label='samples')

# plt.show()
