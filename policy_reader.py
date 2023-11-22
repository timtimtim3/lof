import pickle as pkl
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import gym
import envs
from plotting.plotting import create_grid_plot, plot_policy, get_plot_arrow_params

action_dict = ["LEFT", "UP", "RIGHT", "DOWN"]
PLOTS = True
# LEFT, UP, RIGHT, DOWN = 0, 1, 2, 3

if __name__ == "__main__":

    # This is to check how the SF representation of the discovered policies look like

    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument("--env", type=str, required=True)

    args = parser.parse_args()
    dir = args.results
    env_name = args.env
    env = gym.make(env_name)
    map = env.MAP

    STATES = env.states

    optionsdir = os.path.abspath(f"results/{dir}/options")


    for i, optiondir in enumerate(os.listdir(optionsdir)):
        
        with open(os.path.join(optionsdir, optiondir, "Q.pkl"), "rb") as fp:
            Q = pkl.load(fp)
        
        with open(os.path.join(f"results/{dir}", "metapolicy.pkl"), "rb") as fp:
            metapolicy = pkl.load(fp)

        print(f"\nOption {i}")
        # print(policy["reward"])
        # q = policy["q_table"]
        # int_keys = [key for key in q.keys() if isinstance(key, int)]
        # for key in int_keys:
        #     del q[key]
        # ss = sorted(q.keys())

        q_table = {}

        for idx, state in enumerate(STATES):

            qvalues = Q[idx]
           
            for j in range(len(qvalues)):

                # print(state, action_dict[j], qvalues[j])
                q_table[state] = qvalues

            # print(15 * '--')
        
        
        if PLOTS:
            plt.figure(i)
            ax = plt.subplot(111)
            create_grid_plot(ax=ax, grid=map != 'X')
            start_map = np.zeros_like(map, dtype=bool)
            for char in env.PHI_OBJ_TYPES:
                start_map = np.logical_or(start_map, map == char)

            create_grid_plot(ax=ax, grid=start_map, color_map="YlGn")
            quiv = plot_policy(
                ax=ax, arrow_data=get_plot_arrow_params(q_table, grid_env=env), grid=map,
                values=False, max_index=False
            )
            # plt.show()
            plot_dir = f"results/{dir}/plots"
            if not os.path.isdir(plot_dir):
                os.makedirs(plot_dir)
            plt.savefig(f"{plot_dir}/option{i}.pdf", format='pdf', dpi=1000, pad_inches=0, bbox_inches='tight')