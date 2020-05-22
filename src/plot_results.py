import argparse

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('-t', '--n_tasks', type=int, help='number of tasks', default=100)
    args = parser.parse_args()

    results_df = pd.read_csv("results_{}.csv".format(args.exp_name))
    results_df = results_df[results_df["task_trained_on"] <= args.n_tasks]

    # Average task performance
    plt.figure(figsize=(12,6))
    for model_name in ["EWC", "naive", "foolish"]:
        avg_perf = results_df[results_df["model"] == model_name].groupby(["task_trained_on"])["accuracy"].mean()
        plt.plot(np.arange(args.n_tasks) + 1, avg_perf, label=model_name)
    plt.xlabel('Tasks Trained On', fontsize=14)
    plt.ylabel('Average Accuracy', fontsize=14)
    plt.title('Ability to remember previous tasks on CIFAR100', fontsize=14)
    plt.xticks([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    plt.legend(prop={'size': 16})
    plt.savefig("plots/avg_perf_{}.png".format(args.exp_name))

    # Performance on the last task
    plt.figure(figsize=(12,6))
    for model_name in ["EWC", "naive", "foolish"]:
        model_df = results_df[results_df["model"] == model_name]
        last_task_perf = model_df[model_df["task_trained_on"] == model_df["task_number"]][["task_trained_on", "accuracy"]]
        plt.plot(last_task_perf["task_trained_on"] + 1, last_task_perf["accuracy"], label=model_name)
    plt.xlabel('Tasks Trained On', fontsize=14)
    plt.ylabel('Accuracy on Last Task', fontsize=14)
    plt.title('Ability to learn a new task on CIFAR100', fontsize=14)
    plt.xticks([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    plt.legend(prop={'size': 16})
    plt.savefig("plots/last_task_perf_{}.png".format(args.exp_name))

    # Performance on recent tasks
    plt.figure(figsize=(12,6))
    color_dict = {"EWC": "b", "naive": "orange", "foolish": "g"}
    for model_name in ["EWC", "naive", "foolish"]:
        cur_color = color_dict[model_name]
        model_df = results_df[results_df["model"] == model_name]
        for task_trained_on in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            previous_tasks_perf = model_df[model_df["task_trained_on"] == task_trained_on-1][["task_number", "accuracy"]]
            previous_10_tasks_perf = previous_tasks_perf[previous_tasks_perf["task_number"] >= (task_trained_on - 10)]
            if task_trained_on == 100:
                plt.plot(previous_10_tasks_perf["task_number"] - task_trained_on + 1, previous_10_tasks_perf["accuracy"], 
                        label=model_name, alpha=task_trained_on/100, color=cur_color)
            else:
                plt.plot(previous_10_tasks_perf["task_number"] - task_trained_on + 1, previous_10_tasks_perf["accuracy"], 
                        alpha=task_trained_on/100, color=cur_color)
    plt.xlabel('Lag Between T1 and T2', fontsize=14)
    plt.ylabel('Accuracy on Recent Tasks', fontsize=14)
    plt.title('Ability to remember recent tasks on CIFAR100', fontsize=14)
    plt.xticks([-9, -8, -7, -6, -5, -4, -3, -2, -1, 0])
    plt.legend(prop={'size': 16})
    plt.savefig("plots/recent_task_perf_{}.png".format(args.exp_name))