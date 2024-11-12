import time
import logging
import flwr as fl
import torch
import collections
import torchmetrics
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
#import opinionated
import colormaps as cmaps
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.cluster import KMeans
from typing import List, Tuple, Union, Optional, Callable, Dict
from flwr.common import (
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate
from flwr.server.client_manager import ClientManager

from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union

from flwr.common import (
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log

class CustomFedTrimmedAvg(fl.server.strategy.FedTrimmedAvg):
    threshold = None
    #threshold = 0

    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        beta: float = 0.2,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        self.beta = beta

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]],
    ) -> Tuple[Optional[fl.common.Parameters], fl.common.Scalar]:
        if not results:
            return None, {}

        if CustomFedTrimmedAvg.threshold is None:
            clean_list, noisy_list = self.cluster_clients([fit_res.metrics for _, fit_res in results])

            metrics_aggregated = self.aggregate_client_gnorms([fit_res.metrics for _, fit_res in results], clean_list, noisy_list)
            CustomFedTrimmedAvg.threshold = metrics_aggregated['threshold']
        else:
            metrics_list = [fit_res.metrics for _, fit_res in results]
            clean_list= [res['client_id'] for res in metrics_list if not res['noisy_flag']]
            noisy_list = [res['client_id'] for res in metrics_list if res['noisy_flag']]

        # Apply custom weighted aggregation
        weighted_parameters = self.weighted_aggregation(results, clean_list, noisy_list)
        weighted_ndarrays = parameters_to_ndarrays(weighted_parameters)
        weights_results = [
            (weighted_ndarrays, fit_res.num_examples)
            for _, fit_res in results
        ]

        # Continue with trimmed mean aggregation
        aggregated_weights = ndarrays_to_parameters(
            self.aggregate_trimmed_avg(weights_results, self.beta)
        )

        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        return aggregated_weights, metrics_aggregated

#    def aggregate_fit(self, rnd, results, failures):
#        aggregated_weights, num_samples = super().aggregate_fit(rnd, results, failures)
#
#        return aggregated_weights, num_samples

    def _trim_mean(self, array, proportiontocut):
        """Compute trimmed mean along axis=0.
        https://docs.scipy.org/doc/scipy/reference/generated/
        scipy.stats.trim_mean.html.
        """
        axis = 0
        nobs = array.shape[axis]
        lowercut = int(proportiontocut * nobs)
        uppercut = nobs - lowercut
        if lowercut > uppercut:
            raise ValueError("Proportion too big.")

        atmp = np.partition(array, (lowercut, uppercut - 1), axis)

        slice_list = [slice(None)] * atmp.ndim
        slice_list[axis] = slice(lowercut, uppercut)
        result: NDArray = np.mean(atmp[tuple(slice_list)], axis=axis)
        return result

    def aggregate_trimmed_avg(self, results, proportiontocut):
        """Compute trimmed average."""
        # Create a list of weights and ignore the number of examples
        weights = [weights for weights, _ in results]

        trimmed_w: NDArrays = [
            self._trim_mean(np.asarray(layer), proportiontocut=proportiontocut)
            for layer in zip(*weights)
        ]

        return trimmed_w

    def weighted_aggregation(self, results, clean_clients, noisy_clients):
        total_weights = 0
        aggregated_ndarrays = None

        for client, fit_res in results:
            weight = self.calculate_weight(client, clean_clients, noisy_clients)
            total_weights += weight
            client_weights = parameters_to_ndarrays(fit_res.parameters)

            if aggregated_ndarrays is None:
                aggregated_ndarrays = [
                    w * weight for w in client_weights
                ]
            else:
                for i in range(len(client_weights)):
                    aggregated_ndarrays[i] += client_weights[i] * weight

        aggregated_ndarrays = [w / total_weights for w in aggregated_ndarrays]
        return ndarrays_to_parameters(aggregated_ndarrays)

    def calculate_weight(self, client, clean_clients, noisy_clients):
        if client in noisy_clients:
            return 0.1
        else:
            return 1.2

    def cluster_clients(self, metrics_list):
        # 1. Using Mean of Gradient Norm
        grad_norms = [np.mean(res['grad_norms']) for res in metrics_list]

        # 2. Using Variance of Gradient Norm
        #grad_norms = [np.var(res['grad_norms']) for res in metrics_list]

        client_ids = [res['client_id'] for res in metrics_list]
        agg_gradnorms = np.array(grad_norms).reshape(-1, 1)
        # Clustering
        kmeans = KMeans(n_clusters=2, random_state=42).fit(agg_gradnorms)
        labels = kmeans.labels_

        # Separating client IDs based on clusters
        cluster_0 = [metrics_list[i]['client_id'] for i in range(len(metrics_list)) if labels[i] == 0]
        cluster_1 = [metrics_list[i]['client_id'] for i in range(len(metrics_list)) if labels[i] == 1]

        # Determining which cluster represents clean clients
        mean_0 = np.mean([agg_gradnorms[i][0] for i in range(len(agg_gradnorms)) if labels[i] == 0])
        mean_1 = np.mean([agg_gradnorms[i][0] for i in range(len(agg_gradnorms)) if labels[i] == 1])

        clean_list = cluster_0 if mean_0 > mean_1 else cluster_1
        noisy_list = cluster_1 if mean_0 > mean_1 else cluster_0

        return clean_list, noisy_list

    def compare_clusters_with_flags(self, metrics_list, clean_list, noisy_list):
        noisy_flags = {res['client_id']: res['noisy_flag'] for res in metrics_list}
        mismatches = 0
        total = len(clean_list) + len(noisy_list)

        for client_id in clean_list:
            if noisy_flags[client_id]   :
                mismatches += 1

        for client_id in noisy_list:
            if not noisy_flags[client_id]:
                mismatches += 1

        accuracy = (total - mismatches) / total
        return accuracy, mismatches

    def aggregate_client_gnorms(self, metrics_list, clean_list, noisy_list):
        plt.style.use("opinionated_m")
        clean_list= [res['client_id'] for res in metrics_list if not res['noisy_flag']]
        noisy_list = [res['client_id'] for res in metrics_list if res['noisy_flag']]

        clean_gnorms, noisy_gnorms = [], []
        for res in metrics_list:
            if res['client_id'] in noisy_list:
                noisy_gnorms.extend(res['grad_norms'])
            elif res['client_id'] in clean_list:
                clean_gnorms.extend(res['grad_norms'])

        if len(noisy_gnorms) != 0:
            avg_noisy_gnorms = np.mean(noisy_gnorms)
        else:
            avg_noisy_gnorms = 0

        if len(clean_gnorms) != 0:
            avg_clean_gnorms = np.mean(clean_gnorms)
        else:
            avg_clean_gnorms = 0

        noisy_gnorms = np.array(noisy_gnorms)
        clean_gnorms = np.array(clean_gnorms)
        low_percentile, high_percentile = 5, 95
        clean_gnorms = np.percentile(clean_gnorms, np.arange(low_percentile, high_percentile+1))
        noisy_gnorms = np.percentile(noisy_gnorms, np.arange(low_percentile, high_percentile+1))

        kde_noisy = gaussian_kde(noisy_gnorms, bw_method='scott')
        kde_clean = gaussian_kde(clean_gnorms, bw_method='scott')
        x_values = np.linspace(min(np.min(noisy_gnorms), np.min(clean_gnorms)),
                               max(np.max(noisy_gnorms), np.max(clean_gnorms)), 1000)
        kde_noisy_values = kde_noisy(x_values)
        kde_clean_values = kde_clean(x_values)

        intersections = []
        for i in range(1, len(x_values)):
            if (kde_noisy_values[i-1] - kde_clean_values[i-1]) * (kde_noisy_values[i] - kde_clean_values[i]) < 0:
                intersection_x_value = x_values[i] - (x_values[i] - x_values[i-1]) / \
                                        (kde_noisy_values[i] - kde_clean_values[i] - (kde_noisy_values[i-1] - kde_clean_values[i-1])) * \
                                        (kde_noisy_values[i] - kde_noisy_values[i-1])
                intersections.append(intersection_x_value)

        plt.style.use("seaborn-v0_8-whitegrid")
        fig = plt.figure(figsize=(10, 6))
        mpl.rcParams['font.size'] = 18

        sns.kdeplot(noisy_gnorms, bw_adjust=0.5, fill=True, common_norm=False, label='Noisy', color='red')
        sns.kdeplot(clean_gnorms, bw_adjust=0.5, fill=True, common_norm=False, label='Clean', color='green')
        for intersection_x_value in intersections:
            plt.axvline(intersection_x_value, color='blue', linestyle='--', label='Intersection' if intersection_x_value == intersections[0] else "")

        plt.title('Density Plot of Aggregated Clients Gradient Norm on CIFAR10', loc='center', fontsize=19, pad=10)
        plt.xlabel(r'$L_1$ Norm of Last Layer Gradients', loc='center', fontsize=18, labelpad=10)
        plt.ylabel('Density of Gradient Norm', loc='center', fontsize=18, labelpad=10)
        plt.legend(loc='upper right', fontsize='medium', frameon=True)
        plt.tight_layout()
        plt.show()
        fig.savefig(f'Density_GradNorm_CIFAR100.pdf', format='pdf', dpi=600, bbox_inches='tight')
        plt.close()

        return {
            'avg_noisy_gnorms': avg_noisy_gnorms,
            'avg_clean_gnorms': avg_clean_gnorms,
            'threshold': intersections
        }

class CustomFedNova(fl.server.strategy.FedAvg):
    threshold = None
    #threshold = 0

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]],
    ) -> Tuple[Optional[fl.common.Parameters], fl.common.Scalar]:

        if not results:
            return None, {}

        if CustomFedNova.threshold is None:
            clean_list, noisy_list = self.cluster_clients([fit_res.metrics for _, fit_res in results])

            metrics_aggregated = self.aggregate_client_gnorms([fit_res.metrics for _, fit_res in results], clean_list, noisy_list)
            CustomFedNova.threshold = metrics_aggregated['threshold']
        else:
            metrics_list = [fit_res.metrics for _, fit_res in results]
            clean_list = [res['client_id'] for res in metrics_list if not res['noisy_flag']]
            noisy_list = [res['client_id'] for res in metrics_list if res['noisy_flag']]

        # Compute tau_effective from summation of local client tau: Eqn-6: Section 4.1
        local_tau = [res.metrics["tau"] for _, res in results]
        tau_eff = np.sum(local_tau)

        aggregate_parameters = []

        for _, res in results:
            cid = res.metrics['client_id']
            params_weight = self.calculate_weight(cid, clean_list, noisy_list)
            params = parameters_to_ndarrays(res.parameters)
            #weighted_params = self.weighted_client_params(cid, params, clean_list, noisy_list)
            #weighted_params = parameters_to_ndarrays(res.parameters)

            scale = tau_eff / float(res.metrics["local_norm"])
            scale *= float(res.metrics["ratio"])
            scale *= params_weight
            #print(f"print out scale for the client:{scale}")

            aggregate_parameters.append((params, scale))

        # Aggregate all client parameters with a weighted average using the scale
        agg_cum_gradient = aggregate(aggregate_parameters)

        # Aggregate custom metrics if aggregation function was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        return ndarrays_to_parameters(agg_cum_gradient), metrics_aggregated

#    def aggregate_fit(
#        self,
#        server_round: int,
#        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
#        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]],
#    ) -> Tuple[Optional[fl.common.Parameters], fl.common.Scalar]:
#
#        if not results:
#            return None, {}
#
#        # Compute tau_effective from summation of local client tau: Eqn-6: Section 4.1
#        local_tau = [res.metrics["tau"] for _, res in results]
#        tau_eff = np.sum(local_tau)
#
#        aggregate_parameters = []
#
#        for _client, res in results:
#            params = parameters_to_ndarrays(res.parameters)
#            # compute the scale by which to weight each client's gradient
#            # res.metrics["local_norm"] contains total number of local update steps
#            # for each client
#            # res.metrics["ratio"] contains the ratio of client dataset size
#            # Below corresponds to Eqn-6: Section 4.1
#            scale = tau_eff / float(res.metrics["local_norm"])
#            scale *= float(res.metrics["ratio"])
#
#            aggregate_parameters.append((params, scale))
#
#        # Aggregate all client parameters with a weighted average using the scale
#        agg_cum_gradient = aggregate(aggregate_parameters)
#
#        # Aggregate custom metrics if aggregation function was provided
#        metrics_aggregated = {}
#        if self.fit_metrics_aggregation_fn:
#            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
#            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
#
#        return ndarrays_to_parameters(agg_cum_gradient), metrics_aggregated

    def weighted_client_params(self, cid, client_params, clean_clients, noisy_clients):

        weight = self.calculate_weight(cid, clean_clients, noisy_clients)
        weighted_client_params = [w * weight for w in client_params]

        return ndarrays_to_parameters(weighted_client_params)

    def calculate_weight(self, client, clean_clients, noisy_clients):
        # Your logic to determine weights based on client's noisy or clean status
        if client in noisy_clients:
            return 0.3
        else:
            return 2.0

    def cluster_clients(self, metrics_list):
        # 1. Using Mean of Gradient Norm
        #grad_norms = [np.mean(res['grad_norms']) for res in metrics_list]

        # 2. Using Variance of Gradient Norm
        grad_norms = [np.var(res['grad_norms']) for res in metrics_list]

        client_ids = [res['client_id'] for res in metrics_list]
        mean_gradnorms = np.array(grad_norms).reshape(-1, 1)

        # Clustering
        kmeans = KMeans(n_clusters=2, random_state=42).fit(mean_gradnorms)
        labels = kmeans.labels_

        # Separating client IDs based on clusters
        cluster_0 = [metrics_list[i]['client_id'] for i in range(len(metrics_list)) if labels[i] == 0]
        cluster_1 = [metrics_list[i]['client_id'] for i in range(len(metrics_list)) if labels[i] == 1]

        # Determining which cluster represents clean clients
        mean_0 = np.mean([mean_gradnorms[i][0] for i in range(len(mean_gradnorms)) if labels[i] == 0])
        mean_1 = np.mean([mean_gradnorms[i][0] for i in range(len(mean_gradnorms)) if labels[i] == 1])

        clean_list = cluster_0 if mean_0 > mean_1 else cluster_1
        noisy_list = cluster_1 if mean_0 > mean_1 else cluster_0

        return clean_list, noisy_list

    def compare_clusters_with_flags(self, metrics_list, clean_list, noisy_list):
        noisy_flags = {res['client_id']: res['noisy_flag'] for res in metrics_list}
        mismatches = 0
        total = len(clean_list) + len(noisy_list)

        for client_id in clean_list:
            if noisy_flags[client_id]   :
                mismatches += 1

        for client_id in noisy_list:
            if not noisy_flags[client_id]:
                mismatches += 1

        accuracy = (total - mismatches) / total
        return accuracy, mismatches

    def aggregate_client_gnorms(self, metrics_list, clean_list, noisy_list):
        plt.style.use("opinionated_m")
        clean_list= [res['client_id'] for res in metrics_list if not res['noisy_flag']]
        noisy_list = [res['client_id'] for res in metrics_list if res['noisy_flag']]

        clean_gnorms, noisy_gnorms = [], []
        for res in metrics_list:
            if res['client_id'] in noisy_list:
                noisy_gnorms.extend(res['grad_norms'])
            elif res['client_id'] in clean_list:
                clean_gnorms.extend(res['grad_norms'])

        if len(noisy_gnorms) != 0:
            avg_noisy_gnorms = np.mean(noisy_gnorms)
        else:
            avg_noisy_gnorms = 0

        if len(clean_gnorms) != 0:
            avg_clean_gnorms = np.mean(clean_gnorms)
        else:
            avg_clean_gnorms = 0

        noisy_gnorms = np.array(noisy_gnorms)
        clean_gnorms = np.array(clean_gnorms)
        low_percentile, high_percentile = 5, 95
        clean_gnorms = np.percentile(clean_gnorms, np.arange(low_percentile, high_percentile+1))
        noisy_gnorms = np.percentile(noisy_gnorms, np.arange(low_percentile, high_percentile+1))

        kde_noisy = gaussian_kde(noisy_gnorms, bw_method='scott')
        kde_clean = gaussian_kde(clean_gnorms, bw_method='scott')
        x_values = np.linspace(min(np.min(noisy_gnorms), np.min(clean_gnorms)),
                               max(np.max(noisy_gnorms), np.max(clean_gnorms)), 1000)
        kde_noisy_values = kde_noisy(x_values)
        kde_clean_values = kde_clean(x_values)

        intersections = []
        for i in range(1, len(x_values)):
            if (kde_noisy_values[i-1] - kde_clean_values[i-1]) * (kde_noisy_values[i] - kde_clean_values[i]) < 0:
                intersection_x_value = x_values[i] - (x_values[i] - x_values[i-1]) / \
                                        (kde_noisy_values[i] - kde_clean_values[i] - (kde_noisy_values[i-1] - kde_clean_values[i-1])) * \
                                        (kde_noisy_values[i] - kde_noisy_values[i-1])
                intersections.append(intersection_x_value)

        plt.style.use("seaborn-v0_8-whitegrid")
        fig = plt.figure(figsize=(10, 6))
        mpl.rcParams['font.size'] = 18

        sns.kdeplot(noisy_gnorms, bw_adjust=0.5, fill=True, common_norm=False, label='Noisy', color='red')
        sns.kdeplot(clean_gnorms, bw_adjust=0.5, fill=True, common_norm=False, label='Clean', color='green')
#        for intersection_x_value in intersections:
#            plt.axvline(intersection_x_value, color='blue', linestyle='--', label='Intersection' if intersection_x_value == intersections[0] else "")

        plt.title('Density Plot of Aggregated Clients Gradient Norm on CIFAR100', loc='center', fontsize=19, pad=10)
        plt.xlabel(r'$L_1$ Norm of Last Layer Gradients', loc='center', fontsize=18, labelpad=10)
        plt.ylabel('Density of Gradient Norm', loc='center', fontsize=18, labelpad=10)
        plt.legend(loc='upper right', fontsize='medium', frameon=True)
        plt.tight_layout()
        plt.show()
        fig.savefig(f'Density_GradNorm_CIFAR10.pdf', format='pdf', dpi=600, bbox_inches='tight')
        plt.close()

        return {
            'avg_noisy_gnorms': avg_noisy_gnorms,
            'avg_clean_gnorms': avg_clean_gnorms,
            'threshold': intersections
        }


class CustomFedProx(fl.server.strategy.FedProx):
    threshold = None

    def aggregate_fit(self, rnd, results, failures):
        aggregated_weights, num_samples = super().aggregate_fit(rnd, results, failures)

        return aggregated_weights, num_samples

#    def aggregate_fit(
#        self,
#        server_round: int,
#        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
#        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]],
#    ) -> Tuple[Optional[fl.common.Parameters], fl.common.Scalar]:
#        if not results:
#            return None, {}
#
#        if CustomFedAvg.threshold is None:
#            clean_list, noisy_list = self.cluster_clients([fit_res.metrics for _, fit_res in results])
#
#            metrics_aggregated = self.aggregate_client_gnorms([fit_res.metrics for _, fit_res in results], clean_list, noisy_list)
#            CustomFedAvg.threshold = metrics_aggregated['threshold']
#        else:
#            metrics_list = [fit_res.metrics for _, fit_res in results]
#            clean_list= [res['client_id'] for res in metrics_list if not res['noisy_flag']]
#            noisy_list = [res['client_id'] for res in metrics_list if res['noisy_flag']]
#
#        # Aggregate model updates
#        aggregated_weights = self.weighted_aggregation(results, clean_list, noisy_list)
#
#        # Aggregate custom metrics if aggregation function was provided
#        metrics_aggregated = {}
#        if self.fit_metrics_aggregation_fn:
#            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
#            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
#
#        return aggregated_weights, metrics_aggregated

    def weighted_aggregation(self, results, clean_clients, noisy_clients):
        total_samples = sum([fit_res.num_examples for _, fit_res in results])
        total_weights = 0
        aggregated_ndarrays = None

        for client, fit_res in results:
            weight = self.calculate_weight(client, clean_clients, noisy_clients)
            total_weights += weight
            client_weights = fl.common.parameters_to_ndarrays(fit_res.parameters)

            if aggregated_ndarrays is None:
                aggregated_ndarrays = [
                    w * weight for w in client_weights
                ]
            else:
                for i in range(len(client_weights)):
                    aggregated_ndarrays[i] += client_weights[i] * weight

        # Normalize aggregated weights
        aggregated_ndarrays = [w / total_weights for w in aggregated_ndarrays]
        return fl.common.ndarrays_to_parameters(aggregated_ndarrays)

    def calculate_weight(self, client, clean_clients, noisy_clients):
        # Implement your logic to calculate the weight for each client
        # For example, reducing the weight for noisy clients
        if client in noisy_clients:
            return 0.1
        else:
            return 1.5

    def cluster_clients(self, metrics_list):
        grad_norms = [np.mean(res['grad_norms']) for res in metrics_list]
        client_ids = [res['client_id'] for res in metrics_list]
        mean_gradnorms = np.array(grad_norms).reshape(-1, 1)

        # Clustering
        kmeans = KMeans(n_clusters=2, random_state=42).fit(mean_gradnorms)
        labels = kmeans.labels_

        # Separating client IDs based on clusters
        cluster_0 = [metrics_list[i]['client_id'] for i in range(len(metrics_list)) if labels[i] == 0]
        cluster_1 = [metrics_list[i]['client_id'] for i in range(len(metrics_list)) if labels[i] == 1]

        # Determining which cluster represents clean clients
        mean_0 = np.mean([mean_gradnorms[i][0] for i in range(len(mean_gradnorms)) if labels[i] == 0])
        mean_1 = np.mean([mean_gradnorms[i][0] for i in range(len(mean_gradnorms)) if labels[i] == 1])

        clean_list = cluster_0 if mean_0 > mean_1 else cluster_1
        noisy_list = cluster_1 if mean_0 > mean_1 else cluster_0

        return clean_list, noisy_list

    def compare_clusters_with_flags(self, metrics_list, clean_list, noisy_list):
        noisy_flags = {res['client_id']: res['noisy_flag'] for res in metrics_list}
        mismatches = 0
        total = len(clean_list) + len(noisy_list)

        for client_id in clean_list:
            if noisy_flags[client_id]   :
                mismatches += 1

        for client_id in noisy_list:
            if not noisy_flags[client_id]:
                mismatches += 1

        accuracy = (total - mismatches) / total
        return accuracy, mismatches

    def aggregate_client_gnorms(self, metrics_list, clean_list, noisy_list):
        clean_gnorms, noisy_gnorms = [], []
        for res in metrics_list:
            if res['client_id'] in noisy_list:
                noisy_gnorms.extend(res['grad_norms'])
            elif res['client_id'] in clean_list:
                clean_gnorms.extend(res['grad_norms'])

        if len(noisy_gnorms) != 0:
            avg_noisy_gnorms = np.mean(noisy_gnorms)
        else:
            avg_noisy_gnorms = 0

        if len(clean_gnorms) != 0:
            avg_clean_gnorms = np.mean(clean_gnorms)
        else:
            avg_clean_gnorms = 0

        noisy_gnorms = np.array(noisy_gnorms)
        clean_gnorms = np.array(clean_gnorms)
        low_percentile, high_percentile = 5, 95
        clean_gnorms = np.percentile(clean_gnorms, np.arange(low_percentile, high_percentile+1))
        noisy_gnorms = np.percentile(noisy_gnorms, np.arange(low_percentile, high_percentile+1))

        kde_noisy = gaussian_kde(noisy_gnorms, bw_method='scott')
        kde_clean = gaussian_kde(clean_gnorms, bw_method='scott')
        x_values = np.linspace(min(np.min(noisy_gnorms), np.min(clean_gnorms)),
                               max(np.max(noisy_gnorms), np.max(clean_gnorms)), 1000)
        kde_noisy_values = kde_noisy(x_values)
        kde_clean_values = kde_clean(x_values)

        intersections = []
        for i in range(1, len(x_values)):
            if (kde_noisy_values[i-1] - kde_clean_values[i-1]) * (kde_noisy_values[i] - kde_clean_values[i]) < 0:
                intersection_x_value = x_values[i] - (x_values[i] - x_values[i-1]) / \
                                        (kde_noisy_values[i] - kde_clean_values[i] - (kde_noisy_values[i-1] - kde_clean_values[i-1])) * \
                                        (kde_noisy_values[i] - kde_noisy_values[i-1])
                intersections.append(intersection_x_value)

        sns.set(style="whitegrid")
        plt.figure(figsize=(10, 6))
        sns.kdeplot(noisy_gnorms, bw_adjust=0.5, fill=True, common_norm=False, label='Noisy', color='red')
        sns.kdeplot(clean_gnorms, bw_adjust=0.5, fill=True, common_norm=False, label='Clean', color='green')
        for intersection_x_value in intersections:
            plt.axvline(intersection_x_value, color='blue', linestyle='--', label='Intersection' if intersection_x_value == intersections[0] else "")
        plt.title('Density Plot of Gradient Norm on CIFAR100')
        plt.xlabel('L1 Norm of Last Layer Gradients')
        plt.ylabel('Density')
        plt.legend()
        plt.savefig('./Gradient_Norm_Fed_L1_10.png')
        plt.close()

        return {
            'avg_noisy_gnorms': avg_noisy_gnorms,
            'avg_clean_gnorms': avg_clean_gnorms,
            'threshold': intersections
        }

class CustomFedAvg(fl.server.strategy.FedAvg):
    threshold = None

    def aggregate_fit(self, rnd, results, failures):
        aggregated_weights, num_samples = super().aggregate_fit(rnd, results, failures)

        return aggregated_weights, num_samples

#    def aggregate_fit(
#        self,
#        server_round: int,
#        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
#        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]],
#    ) -> Tuple[Optional[fl.common.Parameters], fl.common.Scalar]:
#        if not results:
#            return None, {}
#
#        if CustomFedAvg.threshold is None:
#            clean_list, noisy_list = self.cluster_clients([fit_res.metrics for _, fit_res in results])
#
#            metrics_aggregated = self.aggregate_client_gnorms([fit_res.metrics for _, fit_res in results], clean_list, noisy_list)
#            CustomFedAvg.threshold = metrics_aggregated['threshold']
#        else:
#            metrics_list = [fit_res.metrics for _, fit_res in results]
#            clean_list= [res['client_id'] for res in metrics_list if not res['noisy_flag']]
#            noisy_list = [res['client_id'] for res in metrics_list if res['noisy_flag']]
#
#        # Aggregate model updates
#        aggregated_weights = self.weighted_aggregation(results, clean_list, noisy_list)
#
#        # Aggregate custom metrics if aggregation function was provided
#        metrics_aggregated = {}
#        if self.fit_metrics_aggregation_fn:
#            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
#            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
#
#        return aggregated_weights, metrics_aggregated

    def weighted_aggregation(self, results, clean_clients, noisy_clients):
        total_samples = sum([fit_res.num_examples for _, fit_res in results])
        total_weights = 0
        aggregated_ndarrays = None

        for client, fit_res in results:
            weight = self.calculate_weight(client, clean_clients, noisy_clients)
            total_weights += weight
            client_weights = fl.common.parameters_to_ndarrays(fit_res.parameters)

            if aggregated_ndarrays is None:
                aggregated_ndarrays = [
                    w * weight for w in client_weights
                ]
            else:
                for i in range(len(client_weights)):
                    aggregated_ndarrays[i] += client_weights[i] * weight

        # Normalize aggregated weights
        aggregated_ndarrays = [w / total_weights for w in aggregated_ndarrays]
        return fl.common.ndarrays_to_parameters(aggregated_ndarrays)

    def calculate_weight(self, client, clean_clients, noisy_clients):
        if client in noisy_clients:
            return 0.1
        else:
            return 1.5

    def cluster_clients(self, metrics_list):
        grad_norms = [np.mean(res['grad_norms']) for res in metrics_list]
        client_ids = [res['client_id'] for res in metrics_list]
        mean_gradnorms = np.array(grad_norms).reshape(-1, 1)

        # Clustering
        kmeans = KMeans(n_clusters=2, random_state=42).fit(mean_gradnorms)
        labels = kmeans.labels_

        # Separating client IDs based on clusters
        cluster_0 = [metrics_list[i]['client_id'] for i in range(len(metrics_list)) if labels[i] == 0]
        cluster_1 = [metrics_list[i]['client_id'] for i in range(len(metrics_list)) if labels[i] == 1]

        # Determining which cluster represents clean clients
        mean_0 = np.mean([mean_gradnorms[i][0] for i in range(len(mean_gradnorms)) if labels[i] == 0])
        mean_1 = np.mean([mean_gradnorms[i][0] for i in range(len(mean_gradnorms)) if labels[i] == 1])

        clean_list = cluster_0 if mean_0 > mean_1 else cluster_1
        noisy_list = cluster_1 if mean_0 > mean_1 else cluster_0

        return clean_list, noisy_list

    def compare_clusters_with_flags(self, metrics_list, clean_list, noisy_list):
        noisy_flags = {res['client_id']: res['noisy_flag'] for res in metrics_list}
        mismatches = 0
        total = len(clean_list) + len(noisy_list)

        for client_id in clean_list:
            if noisy_flags[client_id]   :
                mismatches += 1

        for client_id in noisy_list:
            if not noisy_flags[client_id]:
                mismatches += 1

        accuracy = (total - mismatches) / total
        return accuracy, mismatches

    def aggregate_client_gnorms(self, metrics_list, clean_list, noisy_list):
        clean_gnorms, noisy_gnorms = [], []
        for res in metrics_list:
            if res['client_id'] in noisy_list:
                noisy_gnorms.extend(res['grad_norms'])
            elif res['client_id'] in clean_list:
                clean_gnorms.extend(res['grad_norms'])

        if len(noisy_gnorms) != 0:
            avg_noisy_gnorms = np.mean(noisy_gnorms)
        else:
            avg_noisy_gnorms = 0

        if len(clean_gnorms) != 0:
            avg_clean_gnorms = np.mean(clean_gnorms)
        else:
            avg_clean_gnorms = 0

        noisy_gnorms = np.array(noisy_gnorms)
        clean_gnorms = np.array(clean_gnorms)
        low_percentile, high_percentile = 5, 95
        clean_gnorms = np.percentile(clean_gnorms, np.arange(low_percentile, high_percentile+1))
        noisy_gnorms = np.percentile(noisy_gnorms, np.arange(low_percentile, high_percentile+1))

        kde_noisy = gaussian_kde(noisy_gnorms, bw_method='scott')
        kde_clean = gaussian_kde(clean_gnorms, bw_method='scott')
        x_values = np.linspace(min(np.min(noisy_gnorms), np.min(clean_gnorms)),
                               max(np.max(noisy_gnorms), np.max(clean_gnorms)), 1000)
        kde_noisy_values = kde_noisy(x_values)
        kde_clean_values = kde_clean(x_values)

        intersections = []
        for i in range(1, len(x_values)):
            if (kde_noisy_values[i-1] - kde_clean_values[i-1]) * (kde_noisy_values[i] - kde_clean_values[i]) < 0:
                intersection_x_value = x_values[i] - (x_values[i] - x_values[i-1]) / \
                                        (kde_noisy_values[i] - kde_clean_values[i] - (kde_noisy_values[i-1] - kde_clean_values[i-1])) * \
                                        (kde_noisy_values[i] - kde_noisy_values[i-1])
                intersections.append(intersection_x_value)

        sns.set(style="whitegrid")
        plt.figure(figsize=(10, 6))
        sns.kdeplot(noisy_gnorms, bw_adjust=0.5, fill=True, common_norm=False, label='Noisy', color='red')
        sns.kdeplot(clean_gnorms, bw_adjust=0.5, fill=True, common_norm=False, label='Clean', color='green')
        for intersection_x_value in intersections:
            plt.axvline(intersection_x_value, color='blue', linestyle='--', label='Intersection' if intersection_x_value == intersections[0] else "")
        plt.title('Density Plot of Gradient Norm on CIFAR100')
        plt.xlabel('L1 Norm of Last Layer Gradients')
        plt.ylabel('Density')
        plt.legend()
        plt.savefig('./Gradient_Norm_Fed_L1_10.png')
        plt.close()

        return {
            'avg_noisy_gnorms': avg_noisy_gnorms,
            'avg_clean_gnorms': avg_clean_gnorms,
            'threshold': intersections
        }

    def aggregate_client_metrics(self, metrics_list):
        clean_client_score, noisy_client_score = [], []
        for res in metrics_list:
            if res['noisy_flag'] == True:
                noisy_client_score.extend(res['client_score'])
            else:
                clean_client_score.extend(res['client_score'])

        # plot the density of each list of client dps
        sns.set(style="whitegrid")
        plt.figure(figsize=(10, 6))
        noisy_dps = np.array(noisy_client_score)
        clean_dps = np.array(clean_client_score)
        # Plotting the density
        sns.kdeplot(noisy_dps, bw_adjust=0.5, fill=True, common_norm=False, palette="crest", label='Noisy', color='salmon')
        sns.kdeplot(clean_dps, bw_adjust=0.5, fill=True, common_norm=False, palette="crest", label='Clean', color='green')

        #Cosine Similarity
        plt.title('Density of Similarity Score on Pre-trained Model')
        plt.xlabel('Sample-wise Similarity Score')

        # DPS
        #plt.title('Density of Data Purity Score on Pre-trained Model')
        #plt.xlabel('Sample-wise Data Purity Score')
        plt.ylabel('Density')
        plt.legend()
        #plt.savefig('./DPS_Fed_10epoch_e25.png')
        plt.savefig('./Sim_Fed_Pretrained_median.png')
        plt.close()

        if len(noisy_client_score) != 0:
            avg_noisy_score = sum(noisy_client_score) / len(noisy_client_score)
        else:
            avg_noisy_score = 0
        if len(clean_client_score) != 0:
            avg_clean_score = sum(clean_client_score) / len(clean_client_score)
        else:
            avg_clean_score = 0
        return {'avg_noisy_score': avg_noisy_score, 'avg_clean_score': avg_clean_score}



