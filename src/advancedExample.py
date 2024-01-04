import olympus
from olympus import Surface
from gryffin import Gryffin
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# %matplotlib inline

objective = Surface(kind='Dejong', param_dim=2)


def compute_objective(param):
    param['obj'] = objective.run([val for key, val in param.items()])[0][0]
    return param


config = {
    "general": {
        "random_seed": 42,
        "verbosity": 0,
        "boosted":  False,
    },
    "parameters": [
        {"name": "x_0", "type": "continuous", "low": 0.0, "high": 1.0},
        {"name": "x_1", "type": "continuous", "low": 0.0, "high": 1.0},
    ],
    "objectives": [
        {"name": "obj", "goal": "min"},
    ]
}


def known_constraints(param):
    return param['x_0'] + param['x_1'] < 1.2


gryffin = Gryffin(config_dict=config, known_constraints=known_constraints)

sampling_strategies = [1, -1]

observations = []
MAX_ITER = 5

for num_iter in range(MAX_ITER):
    print('-'*20, 'Iteration:', num_iter+1, '-'*20)

    # Select alternating sampling strategy (i.e. lambda value presented in the Phoenics paper)
    select_ix = num_iter % len(sampling_strategies)
    sampling_strategy = sampling_strategies[select_ix]

    # Query for new parameters
    params = gryffin.recommend(
        observations=observations,
        sampling_strategies=[sampling_strategy]
    )

    param = params[0]
    print('  Proposed Parameters:', param, end=' ')

    # Evaluate the proposed parameters.
    observation = compute_objective(param)
    print('==> Merit:', observation['obj'])

    # Append this observation to the previous experiments
    observations.append(param)

x_domain = np.linspace(0., 1., 60)
y_domain = np.linspace(0., 1., 60)
X, Y = np.meshgrid(x_domain, y_domain)
Z = np.zeros((x_domain.shape[0], y_domain.shape[0]))

for x_index, x_element in enumerate(x_domain):
    for y_index, y_element in enumerate(y_domain):
        loss_value = objective.run([x_element, y_element])[0][0]
        print('==> loss value: ', loss_value)
# ==> loss value:  4.10008950934604
        Z[y_index, x_index] = loss_value

fig, ax = plt.subplots()
contours = plt.contour(X, Y, Z, 3, colors='black')
ax.clabel(contours, inline=True, fontsize=8)
ax.imshow(Z, extent=[0, 1, 0, 1], origin='lower', cmap='RdGy', alpha=0.5)
_ = ax.set_title('Dejong Surface')
_ = ax.set_xlabel('$x_0$')
_ = ax.set_ylabel('$x_1$')

fig, ax = plt.subplots()
contours = plt.contour(X, Y, Z, 3, colors='black')
ax.clabel(contours, inline=True, fontsize=8)
ax.imshow(Z, extent=[0, 1, 0, 1], origin='lower', cmap='RdGy', alpha=0.5)

ax.plot(x_domain, 1.2-y_domain, c='k', ls='--', lw=1)
ax.fill_between(x_domain, 1.2-y_domain, 1.2 -
                y_domain+0.8, color='k', alpha=0.4, )
ax.set_ylim(0., 1.)

_ = ax.set_title('Constrained Dejong Surface')
_ = ax.set_xlabel('$x_0$')
_ = ax.set_ylabel('$x_1$')

# observed parameters and objectives
samples_x = [obs['x_0'] for obs in observations]
samples_y = [obs['x_1'] for obs in observations]
samples_z = [obs['obj'] for obs in observations]
print(samples_y)

_ = plt.scatter(samples_x, samples_y, zorder=10, s=150,
                color='r', edgecolor='#444444', label='samples')

plt.show()
