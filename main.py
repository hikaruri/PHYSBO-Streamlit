import streamlit as st
import numpy as np
from matplotlib import pyplot as plt
import physbo
import time


def function(One_Param_Func: str, x: float) -> float:
    # fx = eval(One_Func_Param)
    fx = eval(One_Param_Func)
    return fx


def simulator(actions: int) -> float:
    x = alpha_val[actions][0]
    fx = function(One_Param_Func, x)
    alpha_action.append(x)
    fx_action.append(fx)
    return fx


if __name__ == '__main__':
    # Make a set of candidates, test_X

    alpha_action = []
    fx_action = []

    st.title('PHYSBO Simulation')
    st.sidebar.title('Setting')
    One_Param_Func = st.sidebar.text_input(
                    label='Function',
                    value='0.1 * (2 * x - 1) * (x - 3) * (x - 5)')
    Rand_Num = st.sidebar.number_input('Random Search Num：', 1, 100, 1)
    Bayz_Num = st.sidebar.number_input('Bayesian Opt. Num：', 1, 100, 5)

    window_num = st.sidebar.number_input('X_Window Num：', 10001)
    alpha_min = st.sidebar.number_input('Xmin：', -100.0, 100.0, 0.0)
    alpha_max = st.sidebar.number_input('Xmax：', -100.0, 100.0, 5.0)
    graph_mergin = st.sidebar.number_input('Mergin val.：', 0.0, 100.0, 0.5)

    alpha_val = np.linspace(alpha_min, alpha_max,
                            window_num).reshape(window_num, 1)

    plot_area = st.empty()

    fig = plt.figure()
    ax = fig.add_subplot()
    x1 = np.arange(0, 5, 0.01)
    y1 = function(One_Param_Func, x1)
    plt.plot(x1, y1, color='#ff4500')
    plt.xlim([alpha_min, alpha_max])
    plt.ylim([min(y1)-graph_mergin, max(y1)+graph_mergin])
    plot_area.pyplot(fig)

    if st.sidebar.button('start'):
        placeholder = st.empty()
        policy = physbo.search.discrete.policy(test_X=alpha_val)
        policy.random_search(max_num_probes=Rand_Num, simulator=simulator)
        best_fx, best_actions = policy.history.export_sequence_best_fx()
        with placeholder:
            st.write(
                f"best_fx: {best_fx[-1]} at {alpha_val[best_actions[-1], :]}")

        for i in range(Bayz_Num):
            policy.bayes_search(max_num_probes=1, simulator=simulator,
                                score="EI", interval=1, num_rand_basis=i)
            ax.clear()
            mean = policy.get_post_fmean(alpha_val)
            var = policy.get_post_fcov(alpha_val)
            std = np.sqrt(var)
            x = alpha_val[:, 0]
            ax.fill_between(x, (mean-std), (mean+std), color='b', alpha=.1)
            ax.scatter(alpha_action, fx_action)
            plt.plot(x1, y1, color='#ff4500')
            plt.xlim([alpha_min, alpha_max])
            plt.ylim([min(y1)-graph_mergin, max(y1)+graph_mergin])
            ax.plot(x, mean)

            plot_area.pyplot(fig)
            time.sleep(2)
            best_fx, best_actions = policy.history.export_sequence_best_fx()
            with placeholder:
                st.write(
                    f"best_fx: {best_fx[-1]} at {alpha_val[best_actions[-1], :]}")
