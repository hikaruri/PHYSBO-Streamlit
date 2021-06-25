import streamlit as st
import numpy as np
from matplotlib import pyplot as plt
import physbo
import time


def function(x: float) -> float:
    return 0.1 * (2 * x - 1) * (x - 3) * (x - 5)


def simulator(actions: int) -> float:
    x = alpha_val[actions][0]
    fx = function(x)
    alpha_action.append(x)
    fx_action.append(fx)
    return fx


if __name__ == '__main__':
    # Make a set of candidates, test_X
    window_num = 10001
    alpha_max = 5.0
    alpha_min = 0.0
    alpha_action = []
    fx_action = []
    alpha_val = np.linspace(alpha_min, alpha_max,
                            window_num).reshape(window_num, 1)

    policy = physbo.search.discrete.policy(test_X=alpha_val)
    policy.set_seed(10)
    policy.random_search(max_num_probes=1, simulator=simulator)
    # policy.bayes_search(max_num_probes=10, simulator=simulator,
    #                     score="EI", interval=1, num_rand_basis=500)
    plot_area = st.empty()
    fig = plt.figure()
    ax = fig.add_subplot()

    for i in range(5):
        policy.bayes_search(max_num_probes=1, simulator=simulator, score="EI",
                            interval=1, num_rand_basis=i)
        ax.clear()
        mean = policy.get_post_fmean(alpha_val)
        var = policy.get_post_fcov(alpha_val)
        std = np.sqrt(var)
        x = alpha_val[:, 0]
        ax.fill_between(x, (mean-std), (mean+std), color='b', alpha=.1)
        ax.scatter(alpha_action, fx_action)
        x1 = np.arange(0, 5, 0.01)
        y1 = function(x1)
        plt.plot(x1, y1, color='#ff4500')
        # グラフを描画し直す
        ax.plot(x, mean)
        # プレースホルダに書き出す
        plot_area.pyplot(fig)
        time.sleep(3)

    # score = policy.get_score(mode="EI", xs=test_X)
    best_fx, best_actions = policy.history.export_sequence_best_fx()
    st.write(f"best_fx: {best_fx[-1]} at {alpha_val[best_actions[-1], :]}")
