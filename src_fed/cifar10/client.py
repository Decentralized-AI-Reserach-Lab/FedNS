import os
import flwr as fl
import clip
import collections
import copy
import torch
import torch.nn.functional as F
import torchmetrics
import torch.optim as optim
from model_utils import *
import numpy as np
import time
from torch.utils.data import DataLoader
from typing import Callable, Dict, List, Tuple
from flwr.common.typing import NDArrays, Scalar

class Client(fl.client.NumPyClient):

    def __init__(self, cid, num_clients, model_loader, data_loader, ratio=None, eva_data=None, pre_eva_data=None, valid_data=None, split_fn=None, noisy_clients=None, device='cuda', noisy_flag=True):
        self.cid = cid
        self.data, self.eva_data, self.pre_eva_data, self.valid_data, self.num_classes, self.num_samples, self.ratio, self.noisy_flag = data_loader(id=cid, num_clients=num_clients, split_fn=split_fn, noisy_clients=noisy_clients)
        self.model_loader = model_loader
        self.device = device

    def set_parameters(self, parameters, config):
        if not hasattr(self, 'model'):
            self.model = self.model_loader(num_classes=self.num_classes).to(self.device)
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = collections.OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config={}):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        self.set_parameters(parameters, config)

        # SGD
        optimizer = optim.SGD(self.model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
        # AdamW
        #optimizer = optim.AdamW(self.model.parameters(),lr=config['lr'],betas=(0.9, 0.999),eps=1e-8,weight_decay=1e-2, amsgrad=False)
        #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])

        # Apply parameters for FedProx and FedNova, commented if not using
        #global_params = [p.clone().detach() for p in self.model.parameters()]
        #proximal_mu=config["proximal_mu"]

        # 1. DPS on-the-fly evaluation on train data
        #score = __class__.evaluate(ds=self.eva_data, model=self.model, num_classes=self.num_classes, noisy_flag=self.noisy_flag)

        # 2. DPS evaluation with pre-trained model
        #score = __class__.pre_evaluate(ds=self.pre_eva_data, noisy_flag=self.noisy_flag)

        # 3. DPS on-the-fly evaluation on validation data
        #score = __class__.evaluate(ds=self.valid_data, model=self.model, num_classes=self.num_classes, noisy_flag=self.noisy_flag)

        # 4. Cosine Similarity evaluation
        #score = __class__.sim_evaluate(train_data=self.eva_data, valid_data=self.valid_data, noisy_flag=self.noisy_flag)
        #return self.get_parameters(), self.num_samples, {'loss': h['loss'], 'accuracy': h['accuracy'], 'client_score': score['client_score'], 'noisy_flag': self.noisy_flag}

        # 5. GradNorm evaluation
        if config['round'] == 1:
            # a. FedAvg, FedTrimmedAvg
#            h = __class__.train(ds=self.data, model=self.model, epochs=config['epochs'], optimizer=optimizer, num_classes=self.num_classes)
#            batch_gnorms = __class__.iterate_batch_gradnorm(data_loader=self.eva_data, model=self.model, num_classes=self.num_classes, client_id=self.cid, temperature=1, noisy_flag=self.noisy_flag)
#            return self.get_parameters(), self.num_samples, {'loss': h['loss'], 'accuracy': h['accuracy'], 'grad_norms': batch_gnorms['grad_norms'], 'noisy_flag': self.noisy_flag, "client_id": self.cid}

            # b. FedProx
#            h = __class__.fedprox_train(ds=self.data, model=self.model, epochs=config['epochs'], optimizer=optimizer, num_classes=self.num_classes, global_params=global_params, proximal_mu=proximal_mu)
#            batch_gnorms = __class__.iterate_batch_gradnorm(data_loader=self.eva_data, model=self.model, num_classes=self.num_classes, client_id=self.cid, temperature=1, noisy_flag=self.noisy_flag)
#            return self.get_parameters(), self.num_samples, {'loss': h['loss'], 'accuracy': h['accuracy'], 'grad_norms': batch_gnorms['grad_norms'], 'noisy_flag': self.noisy_flag, "client_id": self.cid}

            # c. FedNova
            h = __class__.fednova_train(ds=self.data, model=self.model, epochs=config['epochs'], optimizer=optimizer, num_classes=self.num_classes)
            batch_gnorms = __class__.iterate_batch_gradnorm(data_loader=self.eva_data, model=self.model, num_classes=self.num_classes, client_id=self.cid, temperature=1, noisy_flag=self.noisy_flag)
            local_tau = h['local_normalizing_vec'] * self.ratio
            return self.get_parameters({}), self.num_samples, {'loss': h['loss'], 'accuracy': h['accuracy'], 'grad_norms': batch_gnorms['grad_norms'], 'noisy_flag': self.noisy_flag, "client_id": self.cid, "ratio": self.ratio, "tau": local_tau, "local_norm": h['local_normalizing_vec']}

        #elif config['round'] >= 2 and self.noisy_flag:
            # 1. Apply CutMix Augmentation
            #grad_norms = __class__.load_grad_norms(self.cid)
            #augmented_data = __class__.data_aug(data_loader=self.eva_data, threshold=config["threshold"], grad_norms=grad_norms, client_id=self.cid, batch_size=self.eva_data.batch_size, num_classes=self.num_classes)
            #h = __class__.robust_train(ds=augmented_data, model=self.model, epochs=config['epochs'], optimizer=optimizer, scheduler=scheduler, num_classes=self.num_classes, loss=CutMixCrossEntropyLoss(True))

            # 2. Apply logit clipping on noisy clients
            #h = __class__.train_logit_clip(ds=self.data, model=self.model, epochs=config['epochs'], optimizer=optimizer, scheduler=scheduler, num_classes=self.num_classes)

        else:
            # a. FedAvg, FedTrimmedAvg
#            h = __class__.train(ds=self.data, model=self.model, epochs=config['epochs'], optimizer=optimizer, num_classes=self.num_classes)
#            return self.get_parameters(), self.num_samples, {'loss': h['loss'], 'accuracy': h['accuracy'], 'noisy_flag': self.noisy_flag, "client_id": self.cid}

            # b. FedProx
#            h = __class__.fedprox_train(ds=self.data, model=self.model, epochs=config['epochs'], optimizer=optimizer, num_classes=self.num_classes, global_params=global_params, proximal_mu=proximal_mu)
#            return self.get_parameters(), self.num_samples, {'loss': h['loss'], 'accuracy': h['accuracy'], 'noisy_flag': self.noisy_flag, "client_id": self.cid}

            # c. FedNova
            h = __class__.fednova_train(ds=self.data, model=self.model, epochs=config['epochs'], optimizer=optimizer, num_classes=self.num_classes)
            local_tau = h['local_normalizing_vec'] * self.ratio
            return self.get_parameters({}), self.num_samples, {'loss': h['loss'], 'accuracy': h['accuracy'], 'noisy_flag': self.noisy_flag, "client_id": self.cid, "ratio": self.ratio, "tau": local_tau, "local_norm": h['local_normalizing_vec']}

        # 6. Dummy Operation
        # a. FedAvg, FedTrimmedAvg
        #h = __class__.train(ds=self.data, model=self.model, epochs=config['epochs'], optimizer=optimizer, num_classes=self.num_classes)
        #return self.get_parameters({}), self.num_samples, {'loss': h['loss'], 'accuracy': h['accuracy'], 'noisy_flag': self.noisy_flag, "client_id": self.cid}

        # b. FedProx
        #h = __class__.fedprox_train(ds=self.data, model=self.model, epochs=config['epochs'], optimizer=optimizer, scheduler=scheduler, num_classes=self.num_classes, global_params=global_params, proximal_mu=proximal_mu)
        #return self.get_parameters({}), self.num_samples, {'loss': h['loss'], 'accuracy': h['accuracy'], 'noisy_flag': self.noisy_flag, "client_id": self.cid}

        # c. FedNova
#        h = __class__.fednova_train(ds=self.data, model=self.model, epochs=config['epochs'], optimizer=optimizer, num_classes=self.num_classes)
#        local_tau = h['local_normalizing_vec'] * self.ratio
#        return self.get_parameters({}), self.num_samples, {'loss': h['loss'], 'accuracy': h['accuracy'], 'noisy_flag': self.noisy_flag, "client_id": self.cid, "ratio": self.ratio, "tau": local_tau, "local_norm": h['local_normalizing_vec']}

    @staticmethod
    def data_aug(data_loader, threshold, grad_norms, client_id, batch_size, num_workers=4, num_mix=1, beta=1., prob=1.0, num_classes=10, verbose=True):
        np.random.seed(None)
        random.seed(None)
        imgs, lbs = [], []
        for index, (img, lb) in enumerate(data_loader.dataset):
            img = apply_transforms(img)
            lb_onehot = onehot(num_classes, lb)

            # Determine the batch index for the data sample
            batch_index = index // data_loader.batch_size

            # Apply CutMix if the batch's grad norm is below the threshold
            if grad_norms[batch_index] < threshold:
                for _ in range(num_mix):
                    r = np.random.rand(1)
                    if beta <= 0 or r > prob:
                        continue

                    # generate mixed sample
                    lam = np.random.beta(beta, beta)
                    rand_index = random.choice(range(len(data_loader.dataset)))

                    img2, lb2 = data_loader.dataset[rand_index]
                    lb2_onehot = onehot(num_classes, lb2)
                    bbx1, bby1, bbx2, bby2 = rand_bbox(img.size(), lam)
                    img[:, bbx1:bbx2, bby1:bby2] = img2[:, bbx1:bbx2, bby1:bby2]
                    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img.size()[-1] * img.size()[-2]))
                    lb_onehot = lb_onehot * lam + lb2_onehot * (1. - lam)
            imgs.append(img)
            lbs.append(lb_onehot)

        imgs = [torch.tensor(np.array(img), dtype=torch.float32) for img in imgs]
        lbs = [torch.tensor(lb, dtype=torch.float32) for lb in lbs]
        # Create dataset and dataloader
        cutmix_dataset = cifar10dataset(imgs, lbs)
        data_loader = DataLoader(cutmix_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        return data_loader

        # save the cutmix images to local dir, commented if not needed
#        img, soft_label = cutmix_dataset.get_augmented_sample()
#        CIFAR10_LABELS = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
#        import matplotlib.pyplot as plt
#        top_two_probs, top_two_indices = torch.topk(soft_label, 2)
#        # Map these indices to their corresponding CIFAR-10 labels
#        labels = [CIFAR10_LABELS[idx.item()] for idx in top_two_indices]
#        # Combine label names and their corresponding probabilities
#        label_with_probs = [f"{labels[i]} ({top_two_probs[i].item():.2f})" for i in range(2)]
#        mean = np.array([0.4914, 0.4822, 0.4465])
#        std = np.array([0.2023, 0.1994, 0.2010])
#        # Reverse normalization
#        img = img.numpy().transpose((1, 2, 0))  # Convert to HxWxC format
#        img = img * std + mean  # De-normalize
#        img = np.clip(img, 0, 1)
#        plt.figure(figsize=(6, 6))
#        plt.imshow(img)
#        plt.title(", ".join(labels))
#        plt.axis('off')
#        plt.savefig(f"./{top_two_indices}.png")
#        plt.close()

    @staticmethod
    def load_grad_norms(client_id):
        grad_norms_path = f"./cifar10_client_data/grad_norms_client_{client_id}.npy"
        try:
            grad_norms = np.load(grad_norms_path)
            return grad_norms
        except FileNotFoundError:
            print(f"Grad norms file not found for client {client_id}")
            return None


    @staticmethod
    def train(ds, model, epochs, optimizer, num_classes, metrics=None, loss=torch.nn.CrossEntropyLoss(), verbose=False):
        device = next(model.parameters()).device
        if metrics is None:
            metrics = torchmetrics.classification.MulticlassAccuracy(num_classes=num_classes, average='micro').to(device)
        loss_scores = []
        model.train()
        for epoch in range(epochs):
            train_loss = 0.0
            #all_Li = []
            for _, (x, y) in enumerate(ds):
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True).long()
                optimizer.zero_grad()
                preds, feature_outputs = model(x)
                _loss = loss(preds, y)
                _loss.backward()
                optimizer.step()
                #scheduler.step()
                train_loss += _loss.item()
                metrics(preds.max(1)[-1], y)
            train_loss /= len(ds)
            loss_scores.append(train_loss)
            acc = metrics.compute()
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} - Accuracy: {100. * acc:.2f}%")

        return {'loss': loss_scores, 'accuracy': acc}

    @staticmethod
    def fedprox_train(ds, model, epochs, optimizer, scheduler, num_classes, global_params, proximal_mu, metrics=None, loss=torch.nn.CrossEntropyLoss(), verbose=False):
        device = next(model.parameters()).device
        if metrics is None:
            metrics = torchmetrics.classification.MulticlassAccuracy(num_classes=num_classes, average='micro').to(device)
        loss_scores = []
        model.train()
        for epoch in range(epochs):
            train_loss = 0.0
            for _, (x, y) in enumerate(ds):
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True).long().squeeze()
                optimizer.zero_grad()
                preds, feature_outputs = model(x)
                _loss = loss(preds, y)

                # Proximal term calculation
                proximal_term = 0.0
                for param, global_param in zip(model.parameters(), global_params):
                    proximal_term += (param - global_param).norm(2)

                # Total loss with proximal term
                total_loss = _loss + (proximal_mu / 2) * proximal_term
                total_loss.backward()
                optimizer.step()
                #scheduler.step()
                train_loss += total_loss.item()
                metrics(preds.max(1)[-1], y)
            train_loss /= len(ds)
            loss_scores.append(train_loss)
            acc = metrics.compute()
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} - Accuracy: {100. * acc:.2f}%")

        return {'loss': loss_scores, 'accuracy': acc}

    @staticmethod
    def fednova_train(ds, model, epochs, optimizer, num_classes, metrics=None, loss=torch.nn.CrossEntropyLoss(), verbose=False):
        device = next(model.parameters()).device
        # Track the local updates
        local_normalizing_vec = 0
        local_counter = 0
        local_steps = 0
        if metrics is None:
            metrics = torchmetrics.classification.MulticlassAccuracy(num_classes=num_classes, average='micro').to(device)
        loss_scores = []
        model.train()
        for epoch in range(epochs):
            train_loss = 0.0
            for _, (x, y) in enumerate(ds):
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True).long().squeeze()
                optimizer.zero_grad()
                preds, feature_outputs = model(x)
                _loss = loss(preds, y)
                _loss.backward()
                optimizer.step()

                # Update local stats
                local_counter = local_counter * 0.9 + 1 # SGD momentum=0.9
                local_normalizing_vec += local_counter

                train_loss += _loss.item()
                metrics(preds.max(1)[-1], y)
            train_loss /= len(ds)
            loss_scores.append(train_loss)
            acc = metrics.compute()
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} - Accuracy: {100. * acc:.2f}%")

        return {'loss': loss_scores, 'accuracy': acc, 'local_normalizing_vec': local_normalizing_vec}

    @staticmethod
    def robust_train(ds, model, epochs, optimizer, scheduler, num_classes, metrics=None, loss=torch.nn.CrossEntropyLoss(), verbose=False):
        device = next(model.parameters()).device
        if metrics is None:
            metrics = torchmetrics.classification.MulticlassAccuracy(num_classes=num_classes, average='micro').to(device)
        loss_scores = []
        model.train()
        for epoch in range(epochs):
            train_loss = 0.0
            for _, (x, y) in enumerate(ds):
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True).long()
                optimizer.zero_grad()
                preds, feature_outputs = model(x)
                _loss = loss(preds, y)
                _loss.backward()
                optimizer.step()
                scheduler.step()
                train_loss += _loss.item()

                # Convert soft-labels to hard-labels for metric calculation
                hard_labels = torch.argmax(y, dim=1)
                metrics(preds.max(1)[-1], hard_labels)
                #metrics(preds.max(1)[-1], y)
            train_loss /= len(ds)
            loss_scores.append(train_loss)
            acc = metrics.compute()
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} - Accuracy: {100. * acc:.2f}%")

        return {'loss': loss_scores, 'accuracy': acc}

    @staticmethod
    def train_logit_clip(ds, model, epochs, optimizer, scheduler, num_classes, metrics=None, criterion=torch.nn.CrossEntropyLoss(), verbose=False):
        device = next(model.parameters()).device
        temp = 1
        lp = 2
        delta = 1/temp

        if metrics is None:
            metrics = torchmetrics.classification.MulticlassAccuracy(num_classes=num_classes, average='micro').to(device)
        loss_scores = []
        model.train()
        for epoch in range(epochs):
            train_loss = 0.0
            for _, (x, y) in enumerate(ds):
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True).long()
                optimizer.zero_grad()
                logits, feature_outputs = model(x)

                # apply logit clipping
                norms = torch.norm(logits, p=lp, dim=-1, keepdim=True) + 1e-7
                logits_norm = torch.div(logits, norms) * delta
                clip = (norms > temp).expand(-1, logits.shape[-1])
                logits_final = torch.where(clip, logits_norm, logits)
                loss = criterion(logits_final, y)

                loss.backward()
                optimizer.step()
                scheduler.step()
                train_loss += loss.item()
                metrics(logits.max(1)[-1], y)
            train_loss /= len(ds)
            loss_scores.append(train_loss)
            acc = metrics.compute()
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} - Accuracy: {100. * acc:.2f}%")

        return {'loss': loss_scores, 'accuracy': acc}

    @staticmethod
    def evaluate(ds, model, num_classes, metrics=None, loss=torch.nn.CrossEntropyLoss(), verbose=True, noisy_flag=False):
        device = next(model.parameters()).device
        if metrics is None:
            metrics = torchmetrics.classification.MulticlassAccuracy(num_classes=num_classes, average='micro').to(device)
        loss_scores = []
        model.eval()
        eva_loss = 0.0
        all_Li = []
        with torch.no_grad():
            for _, (x, y) in enumerate(ds):
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True).long()
                preds, feature_outputs = model(x)
                _loss = loss(preds, y)
                eva_loss += _loss.item()
                metrics(preds.max(1)[-1], y)
                Li, Aj = calculate_active_features(feature_outputs, epsilon=2.5)
                all_Li.extend(Li)
            eva_loss /= len(ds)
            loss_scores.append(eva_loss)
            acc = metrics.compute()
        all_Aj = torch.zeros(feature_outputs.shape[1])
        for feature_set in all_Li:
            for feature in feature_set:
                all_Aj[feature] += 1

        all_Aj = (all_Aj / len(ds.sampler)) * 100
        #all_Aj = (all_Aj / len(ds.sampler))
        features_idx = collect_D_features(all_Aj, 50, 95)
        features_act = all_Aj[features_idx].numpy()
        # count the No. discriminative features per sample
        from collections import Counter
        sizes = [len(s) for s in all_Li if isinstance(s, set)]
        count_output = str(Counter(sizes))
        # client-wise dps score
        #client_score = noisy_score(all_Li, all_Aj, features_idx, lambda_term=1)
        all_dps_score = rank_D_score(all_Li, all_Aj, features_idx)
        all_dps_score = [x for x in all_dps_score if x != 0]
        if verbose:
            #print(len(all_Li))
            #print("non-zero activation percentage for all features")
            #print(len(Aj))
            #list_aj = []
            #for i in Aj:
            #    if i != 0:
            #        list_aj.append(i)
            #print(len(list_aj))
            #print(f"No. D features for the dataset: {len(features_idx)}")
            #print(features_idx)
            #A_max = np.max(all_Aj[list(features_idx)].numpy())
            print(f"Noisy Flag:{noisy_flag}, No. highly activated Features: {count_output}")
            #print(f"Evaluation ==> Noisy:{noisy_flag} A_max: {A_max:.4f} - Accuracy: {100. * acc:.2f}% - Client Score: {client_score:.5f}")

        return {'client_score': all_dps_score}

    @staticmethod
    def pre_evaluate(ds, verbose=True, noisy_flag=False):
        model, preprocess = clip.load('ViT-B/32', "cpu")
        device = next(model.parameters()).device
        #feature_output = []
        all_Li = []

        with torch.no_grad():
            for _, (x, y) in enumerate(ds):
                x = x.to(device, non_blocking=True)
                f = model.encode_image(x)
                Li, Aj = calculate_active_features(f, epsilon=2.5)
                all_Li.extend(Li)
        all_Aj = torch.zeros(f.shape[1])
        for feature_set in all_Li:
            for feature in feature_set:
                all_Aj[feature] += 1

        all_Aj = (all_Aj / len(ds.sampler)) * 100
        #all_Aj = (all_Aj / len(ds.sampler))
        features_idx = collect_D_features(all_Aj, 50, 95)
        features_act = all_Aj[features_idx].numpy()
        # count the No. discriminative features per sample
        from collections import Counter
        sizes = [len(s) for s in all_Li if isinstance(s, set)]
        count_output = str(Counter(sizes))
        #client_score = noisy_score(all_Li, all_Aj, features_idx, lambda_term=0)
        all_dps_score = rank_D_score(all_Li, all_Aj, features_idx)
        all_dps_score = [x for x in all_dps_score if x != 0]
        if verbose:
            #print(len(all_Li))
            #print("non-zero activation percentage for all features")
            #print(len(Aj))
            #list_aj = []
            #for i in Aj:
            #    if i != 0:
            #        list_aj.append(i)
            #print(len(list_aj))
            #print(f"No. D features for the dataset: {len(features_idx)}")
            #print(features_idx)
            #A_max = np.max(all_Aj[list(features_idx)].numpy())
            print(f"Noisy Flag:{noisy_flag}, No. highly activated Features: {count_output}")
            #print(f"Evaluation ==> Noisy:{noisy_flag} A_max: {A_max:.4f} - Client Score: {client_score:.5f}")

        return {'client_score': all_dps_score}

    @staticmethod
    def iterate_sample_gradnorm(data_loader, model, num_classes, temperature=1, noisy_flag=False, verbose=True):
        gnorms = []
        logsoftmax = torch.nn.LogSoftmax(dim=-1).cuda()
        for b, (x, y) in enumerate(data_loader):
            inputs = x.cuda()

            for i in range(inputs.shape[0]):  # Iterate through each sample
                model.zero_grad()
                input_sample = inputs[i].unsqueeze(0).requires_grad_(True)
                outputs, feature_outputs = model(input_sample)
                targets = torch.ones((1, num_classes)).cuda()
                output_sample = outputs / temperature
                loss = torch.sum(-targets * logsoftmax(output_sample))

                loss.backward(retain_graph=(i != inputs.shape[0] - 1))  # Retain graph except for the last sample
                layer_grad = model.linear.weight.grad.data

                # L1-norm for each sample
                layer_grad_norm = torch.sum(torch.abs(layer_grad)).cpu().numpy()
                # L2-norm
                layer_grad_norm = torch.sqrt(torch.sum(layer_grad**2)).cpu().numpy()
                gnorms.append(layer_grad_norm)

        #if verbose:
        #    print(f"loader size: {len(data_loader)}")

        return {'grad_norms': np.array(gnorms)}

    @staticmethod
    def iterate_batch_gradnorm(data_loader, model, num_classes, client_id, temperature=1, noisy_flag=False, verbose=True):
        #temperature=2
        gnorms = []
        logsoftmax = torch.nn.LogSoftmax(dim=-1).cuda()
        for b, (x, y) in enumerate(data_loader):
            inputs = x.cuda().requires_grad_(True)

            model.zero_grad()
            outputs, feature_outputs = model(inputs)
            targets = torch.ones((inputs.shape[0], num_classes)).cuda()
            outputs = outputs / temperature
            loss = torch.mean(torch.sum(-targets * logsoftmax(outputs), dim=-1))

            loss.backward()
            # UNCOMMENT this line with the last layer in your model
            # 1. ResNet-18
            layer_grad = model.linear.weight.grad.data
            # 2. VGG16
            #layer_grad = model.classifier[-1].weight.grad.data
            # 3. ViT-small
            #layer_grad = model.mlp_head[1].weight.grad.data

            # UNCOMMENT this line with the penultimate layer in the model
            # 1. ConvMixer-256/8
            # layer_grad = model.mixer_layers[-1][1].weight.grad.data
            # 2. ResNet-18
            # layer_grad = model.layer4[-1].conv2.weight.grad.data

            # L1-norm
            layer_grad_norm = torch.sum(torch.abs(layer_grad)).cpu().numpy()
            # L2-norm
            #layer_grad_norm = torch.sqrt(torch.sum(layer_grad**2)).cpu().numpy()
            gnorms.append(layer_grad_norm)

        #if verbose:
            #print(f"Noisy Flag:{noisy_flag}, Mean_GradNorm: {np.mean(gnorms)}, Variance_GradNorm: {np.var(gnorms)}")
            #print(f"loader size: {len(data_loader)} gradnorm size: {len(gnorms)}")

        client_data_dir = "./cifar10_client_data"
        os.makedirs(client_data_dir, exist_ok=True)
        grad_norms_path = f"./cifar10_client_data/grad_norms_client_{client_id}.npy"
        np.save(grad_norms_path, np.array(gnorms))
        return {'grad_norms': np.array(gnorms)}


    @staticmethod
    def sim_evaluate(train_data, valid_data, noisy_flag=False, verbose=True):
        model, preprocess = clip.load('ViT-B/32', "cpu")
        device = next(model.parameters()).device
        valid_feature_outputs = []
        sim_score = []
        with torch.no_grad():
            for _, (x, _) in enumerate(valid_data):
                x = x.to(device, non_blocking=True)
                f_valid = model.encode_image(x)
                valid_feature_outputs.extend(f_valid)
            valid_feature_outputs = torch.stack(valid_feature_outputs)

            for _, (x, _) in enumerate(train_data):
                x = x.to(device, non_blocking=True)
                f = model.encode_image(x)
                if f.ndim == 1:
                    f = f.unsqueeze(0)

                cos_sim = F.cosine_similarity(f, valid_feature_outputs)

                # Compute max, mean, and variance
                max_sim = torch.max(cos_sim).item()
                mean_sim = torch.mean(cos_sim).item()
                var_sim = torch.var(cos_sim).item()
                median_sim = torch.median(cos_sim).item()
                #if verbose:
                    #print(f"Train x Feature Outputs Size: {f.shape}")
                    #print(f"valid Feature Outputs Size: {valid_feature_outputs.shape}")
                    #print(f"Noisy Flag:{noisy_flag} Max similarity: {max_sim}, Mean similarity: {mean_sim}, Variance: {var_sim}")
                sim_score.append(max_sim)

        print(f"Noisy Flag: {noisy_flag} Mean of Similarity Score: {np.mean(sim_score)}")

        return {'client_score': sim_score}


