import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"
import time
import argparse
import pickle
import flwr as fl
from utils import none_or_str
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import torch
from utils import *

#torch.cuda.empty_cache()
parser = argparse.ArgumentParser(description="Federated Learning Parameters")
parser.add_argument("--num_rounds", default=200, type=int, help="number of federated rounds")
parser.add_argument("--num_clients", default=20, type=int, help="number of clients")
parser.add_argument("--participation", default=1.0, type=float, help="participation percentage of clients in each round")
parser.add_argument("--data_split", default='iid', type=str, help="data split procedure")
parser.add_argument("--max_parallel_executions", default=5, type=int, help="number of clients instances to run in parallel")
parser.add_argument("--timeout", default=1500, type=int, help="maximum seconds of federated round until timeout")
parser.add_argument("--init_model", default=None, type=none_or_str, nargs='?', help="initialization weights path")
args = parser.parse_args()

def evaluate():
    # Load the global model weights from the results.pkl file
    with open('results.pkl', 'rb') as file:
        results_data = pickle.load(file)
        global_model_weights = results_data['history'][-1]['parameters']

    # Create server
    server = create_server(init_model=global_model_weights)
    # Start simulation
    history = fl.simulation.start_simulation(client_fn=create_client, server=server, num_clients=args.num_clients, \
                                             ray_init_args= {"ignore_reinit_error": True, "num_cpus": int(min(args.max_parallel_executions,args.num_clients)),}, \
                                             config=fl.server.ServerConfig(num_rounds=args.num_rounds, round_timeout=args.timeout),)
    print(history)
    return history

def plot_results(history, marker_interval=10, start_round=20):
    sns.set(style="darkgrid")

    avg_clean_score = [item[1] for item in history.metrics_distributed_fit['avg_clean_score']]
    avg_noisy_score = [item[1] for item in history.metrics_distributed_fit['avg_noisy_score']]
    rounds = [item[0] for item in history.metrics_distributed_fit['avg_clean_score']]

    # Filter the data to start from the specified round
    avg_clean_score = [score for i, score in enumerate(avg_clean_score) if rounds[i] >= start_round]
    avg_noisy_score = [score for i, score in enumerate(avg_noisy_score) if rounds[i] >= start_round]
    rounds = [round_num for round_num in rounds if round_num >= start_round]

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot lines
    ax.plot(rounds, avg_clean_score, linestyle='-', color='navy', linewidth=1, label='Average Clean Score')
    ax.plot(rounds, avg_noisy_score, linestyle='-', color='crimson', linewidth=1, label='Average Noisy Score')

    # Plot markers for every nth point
    ax.plot(rounds[::marker_interval], avg_clean_score[::marker_interval], 'b^', markersize=5)
    ax.plot(rounds[::marker_interval], avg_noisy_score[::marker_interval], 'ro', markersize=5)

    ax.set_xlabel('Round', fontsize=14, fontweight='bold')
    ax.set_ylabel('Rank Score', fontsize=14, fontweight='bold')
    #ax.set_title('Result of Average Rank Score on Clients (5 Clean, 15 Noisy)', fontsize=16, fontweight='bold')
    ax.set_title('Result of Average Rank Score (20 clean clients)', fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', shadow=True, fontsize='large')
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.savefig('rank_score_5_15_cifar10_e35_3epoch_clean.png', format='png', dpi=300)

def plot_accuracy(history, marker_interval=10):
    # Set Seaborn style
    sns.set(style="darkgrid")

    # Extracting accuracy data
    accuracy_data = history.metrics_centralized['accuracy']
    rounds = [item[0] for item in accuracy_data]
    accuracy = [item[1].item() for item in accuracy_data]  # Convert tensors to floats

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(rounds, accuracy, linestyle='-', color='darkgreen', linewidth=2, label='Accuracy')

    # Plotting markers for every nth point
    ax.plot(rounds[::marker_interval], accuracy[::marker_interval], 'go', markersize=5)

    ax.set_xlabel('Rounds', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
    ax.set_title('Accuracy Over Rounds (20 clean clients)', fontsize=16, fontweight='bold')

    ax.legend(loc='upper left', shadow=True, fontsize='large')
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.savefig('accuracy_5_15_cifar10_e35_3epoch_clean.png', format='png', dpi=300)

def run_experiment(num_rounds=120, num_clients=20, participation=1.0, data_split='iid', max_parallel_executions=5, timeout=1500, init_model=None, noisy_clients=None):

    def create_client(cid):
        import os, sys; sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from utils import (grab_gpu, get_split_fn)
        time.sleep(int(cid)*0.75)
        os.environ['CUDA_VISIBLE_DEVICES'] = grab_gpu()
        import warnings
        warnings.simplefilter("ignore")
        import network
        import data
        from client import Client
        #is_noisy_client = cid in noisy_clients
        return Client(int(cid), num_clients=num_clients, model_loader=network.get_network, data_loader=data.get_data, split_fn=get_split_fn(data_split), noisy_clients=noisy_clients)

    def create_server(init_model=None):
        from utils import grab_gpu
        os.environ['CUDA_VISIBLE_DEVICES'] = grab_gpu()
        import warnings
        warnings.simplefilter("ignore")
        import network
        import data
        from server import Server
        return Server(num_rounds=num_rounds, num_clients=num_clients, participation=participation,
            model_loader=network.get_network, data_loader=data.get_data, init_model=init_model)

    server = create_server()
    history = fl.simulation.start_simulation(client_fn=create_client, server=server, num_clients=num_clients,
                                             ray_init_args={"ignore_reinit_error": True, "num_cpus": int(min(max_parallel_executions, num_clients))},
                                             config=fl.server.ServerConfig(num_rounds=num_rounds, round_timeout=timeout))
    return history


def run_with_different_configs():
    random.seed(42)
    np.random.seed(42)
    configs = [
#        {"num_rounds": 150, "num_clients": 20, "data_split": "iid", "noisy_clients": [], "participation": 1.0, "max_parallel_executions": 6},
#        {"num_rounds": 150, "num_clients": 20, "data_split": "noniid", "noisy_clients": [], "participation": 1.0, "max_parallel_executions": 6},
        {"num_rounds": 150, "num_clients": 20, "data_split": "iid", "noisy_clients": random.sample(range(20), 15), "participation": 1.0, "max_parallel_executions": 6},
        {"num_rounds": 150, "num_clients": 20, "data_split": "noniid", "noisy_clients": random.sample(range(20), 15), "participation": 1.0, "max_parallel_executions": 6},
    ]

    for config in configs:
        history = run_experiment(num_rounds=config["num_rounds"], num_clients=config["num_clients"], data_split=config["data_split"], noisy_clients=config["noisy_clients"], participation=config["participation"], max_parallel_executions=config["max_parallel_executions"])
        log_results(history, config)

def log_results(history, config):
    with open("exp_log.txt", "a") as file:
        file.write(f"Config: {config}\n")
        file.write(f"History: {history}\n")
        file.write("---------------------------\n")

if __name__ == "__main__":

    run_with_different_configs()
