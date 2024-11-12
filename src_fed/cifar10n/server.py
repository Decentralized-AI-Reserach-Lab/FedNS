import logging
import flwr as fl
import torch
import collections
import torchmetrics
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.cluster import KMeans
from typing import List, Tuple, Union, Optional
from strategy import *

class Server(fl.server.Server):

    def __init__(self, model_loader, data_loader, num_rounds, num_clients=20,
        participation=1.0, init_model=None, log_level=logging.INFO):

        self.num_rounds = num_rounds
        self.data, self.num_classes, self.num_samples = data_loader(return_eval_ds=True)
        self.model_loader = model_loader
        self.init_model = init_model
        self.clients_config = {"epochs":5, "lr":1e-3}
        self.num_clients = num_clients
        self.participation = participation
        self.set_strategy(self)
        self._client_manager = fl.server.client_manager.SimpleClientManager()
        self.max_workers = None
        self.device = 'cuda'
        logging.getLogger("flower").setLevel(log_level)

    def set_max_workers(self, *args, **kwargs):
        return super(Server, self).set_max_workers(*args, **kwargs)

    def set_strategy(self, *_):

#        self.strategy = CustomFedTrimmedAvg(
#            min_available_clients=self.num_clients, fraction_fit=self.participation,
#            min_fit_clients=int(self.participation*self.num_clients), fraction_evaluate=0.0,
#            min_evaluate_clients=0, evaluate_fn=self.get_evaluation_fn(),
#            on_fit_config_fn=self.get_client_config_fn(), initial_parameters=self.get_initial_parameters(),
#        )

        self.strategy = CustomFedNova(
            min_available_clients=self.num_clients, fraction_fit=self.participation,
            min_fit_clients=int(self.participation*self.num_clients), fraction_evaluate=0.0,
            min_evaluate_clients=0, evaluate_fn=self.get_evaluation_fn(),
            on_fit_config_fn=self.get_client_config_fn(), initial_parameters=self.get_initial_parameters(),
        )

#        self.strategy = CustomFedProx(
#            min_available_clients=self.num_clients, fraction_fit=self.participation,
#            min_fit_clients=int(self.participation*self.num_clients), fraction_evaluate=0.0,
#            min_evaluate_clients=0, evaluate_fn=self.get_evaluation_fn(),
#            on_fit_config_fn=self.get_client_config_fn(), initial_parameters=self.get_initial_parameters(),
#            proximal_mu=0.1,
#        )

#        self.strategy = CustomFedAvg(
#            min_available_clients=self.num_clients, fraction_fit=self.participation,
#            min_fit_clients=int(self.participation*self.num_clients), fraction_evaluate=0.0,
#            min_evaluate_clients=0, evaluate_fn=self.get_evaluation_fn(),
#            on_fit_config_fn=self.get_client_config_fn(), initial_parameters=self.get_initial_parameters(),
#        )

    def client_manager(self, *args, **kwargs):
        return super(Server, self).client_manager(*args, **kwargs)

    def get_parameters(self, config={}):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters, config):
        if not hasattr(self, 'model'):
            self.model = self.model_loader(num_classes=self.num_classes).to(self.device)
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = collections.OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def get_initial_parameters(self, *_):
        """ Get initial random model weights """
        if self.init_model is not None:
            self.init_weights = torch.load(self.init_model, map_location=self.device).state_dict()
        else:
            self.init_weights = [val.cpu().numpy() for _, val in self.model_loader(num_classes=self.num_classes).state_dict().items()]
        return fl.common.ndarrays_to_parameters(self.init_weights)

    def get_evaluation_fn(self):
        def evaluation_fn(rnd, parameters, config):
            self.set_parameters(parameters, config)
            metrics = __class__.evaluate(model=self.model, ds=self.data, num_classes=self.num_classes)
            return metrics[0], {"accuracy":metrics[1]}
        return evaluation_fn

    def get_client_config_fn(self):
        " Define fit config function with constant self objects."
        def get_on_fit_config_fn(rnd):
            if rnd >= 2:
                self.clients_config["threshold"] = CustomFedAvg.threshold
            self.clients_config["round"] = rnd
            return self.clients_config
        return get_on_fit_config_fn

    def process_client_score(self, results):
        for client_id, client_result in results:
            client_score = client_result[2]['client_score']
            noisy_flag = client_result[2]['noisy_flag']
            print(f"Client {client_id}: Score - {client_score}, Noisy - {noisy_flag}")

    def after_round(self, rnd, results):
        self.process_client_score(results)

    @staticmethod
    def evaluate(ds, model, num_classes, metrics=None, loss=torch.nn.CrossEntropyLoss(), verbose=False):
        device = next(model.parameters()).device
        if metrics is None:
            metrics = torchmetrics.classification.MulticlassAccuracy(num_classes=num_classes, average='micro').to(device)
        model.eval()
        _loss = 0.0
        with torch.no_grad():
            for _, (x, y) in enumerate(ds):
                x, y = x.to(device), y.to(device).long().squeeze()
                preds, feature_outputs = model(x)
                _loss += loss(preds, y).item()
                metrics(preds.max(1)[-1], y)
        _loss /= len(ds)
        acc = metrics.compute()
        if verbose:
            print(f"Loss: {_loss:.4f} - Accuracy: {100. * acc:.2f}%")
        return (_loss, acc)
