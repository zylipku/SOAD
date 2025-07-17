import argparse
import operator
import os

import jax
import powerpax as ppx
import pyqg_jax
from jax import numpy as jnp
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--seedno", default=0, type=int)
args = parser.parse_args()

seedno = args.seedno
print(f"generating with seedno: {seedno}")

data_save_root = "/public5/share/lizhuoyuan/datasets/qg3/"

minute = 60.0
hour = 60 * minute
day = 24 * hour
month = 3 * day
year = 365 * day

base_model = pyqg_jax.qg_model.QGModel(nx=512, precision=pyqg_jax.state.Precision.DOUBLE)
# print(base_model)
# param_model = pyqg_jax.parameterizations.smagorinsky.apply_parameterization(
#     base_model, constant=0.08,
# )
dt = 15 * minute
# dt=14400


stepper = pyqg_jax.steppers.AB3Stepper(dt=dt)
stepped_model = pyqg_jax.steppers.SteppedModel(base_model, stepper)
init_state = stepped_model.create_initial_state(jax.random.key(seedno))


@jax.jit
def warmup(state):
    def loop_fn(carry, _x):
        current_state = carry
        next_state = stepped_model.step_model(current_state)
        # Note: we output the current state for ys
        # This includes the starting step in the trajectory
        return next_state, current_state

    _final_carry, traj_steps = ppx.sliced_scan(
        loop_fn, state, None, start=int(5 * year / dt), length=int(5 * year / dt) + 1
    )

    return traj_steps


@jax.jit
def forward_one_traj(state):
    def loop_fn(carry, _x):
        current_state = carry
        next_state = stepped_model.step_model(current_state)
        # Note: we output the current state for ys
        # This includes the starting step in the trajectory
        return next_state, current_state

    _final_carry, traj_steps = ppx.sliced_scan(loop_fn, state, None, length=int(100 * day / dt), step=int(day / dt))

    return traj_steps


warmup_state = warmup(init_state)
prev_state = jax.tree_util.tree_map(operator.itemgetter(-1), warmup_state)

ntrajs = 1000

for idx in range(ntrajs):
    print(f"{idx=}")
    traj = forward_one_traj(prev_state)
    window = jax.tree_util.tree_map(lambda leaf: leaf[:32], traj)
    jnp.save(os.path.join(data_save_root, f"seedno={seedno}_traj={idx:05d}.npy"), window.state.q)

    prev_state = jax.tree_util.tree_map(operator.itemgetter(-1), traj)

    final_q = prev_state.state.q

    fig, axs = plt.subplots(1, 2, layout="constrained")
    for layer, ax in enumerate(axs):
        # final_q is now a plain JAX array, we can slice it directly
        data = final_q[layer]
        vmax = jnp.abs(data).max()
        ax.set_title(f"Layer {layer}")
        ax.imshow(data, cmap="twilight", vmin=-vmax, vmax=vmax)

    # plt.savefig(f'seedno={seedno}_traj={idx:05d}.png')
    plt.clf()
    plt.close()
