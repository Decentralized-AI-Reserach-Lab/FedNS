'''
    Part of Copyright from:
        https://github.com/kuangliu/pytorch-cifar
        https://github.com/ildoonet/cutmix
'''
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import Dataset
import seaborn as sns
import matplotlib.pyplot as plt
from torch.nn.modules.module import Module
from scipy.stats import gaussian_kde
import random
from torchvision import transforms
from PIL import Image


def unnormalize(tensor, mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)  # Unnormalize each channel
    return tensor

def apply_transforms(img):
    # Unnormalize and convert to PIL
    tensor = unnormalize(img)

    # Apply additional transformations
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    return transform(img)

class CutMixCrossEntropyLoss(Module):
    def __init__(self, size_average=True):
        super().__init__()
        self.size_average = size_average

    def forward(self, input, target):
        if len(target.size()) == 1:
            target = torch.nn.functional.one_hot(target, num_classes=input.size(-1))
            target = target.float().cuda()
        return cross_entropy(input, target, self.size_average)


def cross_entropy(input, target, size_average=True):
    """ Cross entropy that accepts soft targets
    Args:
         pred: predictions for neural network
         targets: targets, can be soft
         size_average: if false, sum is returned instead of mean

    Examples::

        input = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
        input = torch.autograd.Variable(out, requires_grad=True)

        target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        target = torch.autograd.Variable(y1)
        loss = cross_entropy(input, target)
        loss.backward()
    """
    logsoftmax = torch.nn.LogSoftmax(dim=1)
    if size_average:
        return torch.mean(torch.sum(-target * logsoftmax(input), dim=1))
    else:
        return torch.sum(torch.sum(-target * logsoftmax(input), dim=1))

def onehot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec


def rand_bbox(size, lam):
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[1]
        H = size[2]
    else:
        raise Exception

    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

class CutMixDataset(Dataset):
    def __init__(self, data, grad_norms, threshold, num_mix=1, beta=1., prob=1.0, num_classes=10, batch_size=32):
        self.dataset = data
        self.grad_norms = grad_norms
        self.threshold = threshold
        self.num_mix = num_mix
        self.beta = beta
        self.prob = prob
        self.num_classes = num_classes
        self.batch_size=batch_size
        self.cutmix_counter = 0

    def __getitem__(self, index):
        img, lb = self.dataset[index]
        lb_onehot = onehot(self.num_classes, lb)

        # Determine the batch index for this data point
        batch_index = index // self.batch_size

        # Apply CutMix if the batch's grad norm is below the threshold
        if self.grad_norms[batch_index] < self.threshold:
            for _ in range(self.num_mix):
                r = np.random.rand(1)
                if self.beta <= 0 or r > self.prob:
                    continue

                # generate mixed sample
                lam = np.random.beta(self.beta, self.beta)
                rand_index = random.choice(range(len(self)))

                img2, lb2 = self.dataset[rand_index]
                lb2_onehot = onehot(self.num_classes, lb2)

                bbx1, bby1, bbx2, bby2 = rand_bbox(img.size(), lam)
                img[:, bbx1:bbx2, bby1:bby2] = img2[:, bbx1:bbx2, bby1:bby2]
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img.size()[-1] * img.size()[-2]))
                lb_onehot = lb_onehot * lam + lb2_onehot * (1. - lam)

            self.cutmix_counter += 1
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img).float()
            if isinstance(lb_onehot, np.ndarray):
                lb_onehot = torch.from_numpy(lb_onehot).float()
            return img, lb_onehot
        else:
            # Check if img is a numpy array before converting
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img).float()
            return img, lb

    def __len__(self):
        return len(self.dataset)

    def get_cutmix_count(self):
        return self.cutmix_counter

    def get_augmented_sample(self):
        # Loop until an augmented sample is found
        while True:
            index = random.randint(0, len(self.dataset) - 1)
            img, lb_onehot = self.__getitem__(index)
            batch_index = index // self.batch_size
            # Check if this sample was augmented
            if self.grad_norms[batch_index] < self.threshold:
                return img, lb_onehot

class cifar10dataset(Dataset):
    def __init__(self, data, labels, transforms=None) -> None:
        super().__init__()
        self.data = data
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        label = self.labels[index]
        if self.transforms:
            sample = self.transforms(sample)

        return sample, label

    def update_item(self, idx, new_data, new_label):
        self.data[idx] = new_data
        self.labels[idx] = new_label

class cifar100dataset(Dataset):
    def __init__(self, data, labels, transforms=None) -> None:
        super().__init__()
        self.data = data
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        label = self.labels[index]
        if self.transforms:
            sample = self.transforms(sample)

        return sample, label

    def update_item(self, idx, new_data, new_label):
        self.data[idx] = new_data
        self.labels[idx] = new_label

class cifar10MixedNoisyDataset(Dataset):
    def __init__(self, data, labels, noise_types=None, transforms=None) -> None:
        super().__init__()
        self.data = data
        self.labels = labels
        self.noise_types = noise_types
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        label = self.labels[index]
        noise_type = self.noise_types[index]
        if self.transforms:
            sample = self.transforms(sample)

        return sample, label, noise_type

def calculate_active_features(feature_outputs, epsilon=4):
    """
    Calculate the set of highly activating features (L_i) for each sample in the batch and 
    the percentage of highly activated features (A_j) for each feature.

    Parameters:
    - feature_outputs: A tensor of shape [batch_size, feature_size]
    - epsilon: Hyperparameter for threshold calculation

    Returns:
    - Li: A list of sets, each containing the indices of highly activating features for each sample
    - Aj: A tensor containing the percentage of highly activating samples for each feature
    """
    # Calculate mean and std deviation for each sample
    mu = torch.mean(feature_outputs, dim=1)
    sigma = torch.std(feature_outputs, dim=1)

    Li = []
    Aj = torch.zeros(feature_outputs.shape[1])

    for i in range(feature_outputs.shape[0]):
        # Calculate Li for each sample
        threshold = mu[i] + epsilon * sigma[i]
        highly_activating_features = set((feature_outputs[i] > threshold).nonzero(as_tuple=True)[0].tolist())
        Li.append(highly_activating_features)

        # Update Aj for each feature
        for feature in highly_activating_features:
            Aj[feature] += 1

    # Calculate percentage for Aj
    Aj = (Aj / feature_outputs.shape[0]) * 100
    #Aj = (Aj / feature_outputs.shape[0])

    return Li, Aj

def collect_D_features(all_Aj, lower_bound, upper_bound):
    non_zero_values = np.array(all_Aj)[np.array(all_Aj) != 0]

    # Calculate 50th and 90th percentiles of non-zero values
    if non_zero_values.size > 0:
        p_low = np.percentile(non_zero_values, lower_bound)
        p_high = np.percentile(non_zero_values, upper_bound)
    else:
        # Handle the case
        print("No activated features!\n")
        raise ValueError

    # Find indices where feature activation falls between the lower and higher bound
    selected_indices = np.where((all_Aj >= p_low) & (all_Aj <= p_high))[0]

    return selected_indices

def noisy_score(all_Li, all_Aj, D, lambda_term=1):
    """
    Calculate the noisy score for the client

    Parameters:
    - all_Li: A list of sets, each containing the indices of highly activating features for each sample
    - all_Aj: A tensor containing the percentage of highly activating samples for each feature
    - D: A set containing the indices of discriminative features
    - lambda_term: A hyperparameter to control the importance of regularization term

    Returns:
    - noisy_scores: The noisy scores for the client data
    """
    noisy_score = 0
    D = set(D)
    if len(D) != 0:
        for feature_set in all_Li:
            if len(feature_set.intersection(D)) == 0:
                continue
            else:
                Li = list(feature_set.intersection(D))
                Ai = all_Aj[Li].numpy()
                score = (np.log(1 + np.exp(np.sum(Ai)))) / len(Li)
                #score = (np.log(1 + np.exp(np.sum(Ai) * 3))) / len(Li)
                noisy_score += score

        # add regularization term
        noisy_score /= len(all_Li)
        noisy_score += (np.max(all_Aj[list(D)].numpy()) * lambda_term)

    return noisy_score

def rank_D_score(all_Li, all_Aj, D):
    """
    Calculate the noisy score for each data sample.

    Parameters:
    - all_Li: A list of sets, each containing the indices of highly activating features for each sample
    - all_Aj: A tensor containing the percentage of highly activating samples for each feature
    - D: A set containing the indices of discriminative features

    Returns:
    - all_rank_score: The list of score for all the data samples
    """
    all_rank_score = []
    D = set(D)
    for feature_set in all_Li:
        if len(feature_set.intersection(D)) == 0:
            score = 0
            all_rank_score.append(score)
            continue
        else:
            Li = list(feature_set.intersection(D))
            Ai = all_Aj[Li].numpy()
            score = (np.log(1 + np.exp(np.sum(Ai)))) / len(Li)
            all_rank_score.append(score)

    return all_rank_score

def plot_score_density(scores, label, color):
    scores = [x for x in scores if x != 0]
    density = gaussian_kde(scores)
    x_vals = np.linspace(min(scores), max(scores), 1000)
    # Plot the density
    plt.plot(x_vals, density(x_vals), label=label, color=color, linewidth=2)
    plt.fill_between(x_vals, density(x_vals), alpha=0.4, color=color)

def q_score(feature_outputs, Li, D):
    """
    Calculate the Q-Score for each sample in the batch.

    Parameters:
    - feature_outputs: A tensor of shape [batch_size, feature_size], containing the activations h_ij
    - Li: A list of sets, each containing the indices of highly activating features for each sample
    - D: A set containing the indices of discriminative features

    Returns:
    - Q_scores: A list of Q-Scores for each sample in the batch
    """
    Q_scores = []
    mu = np.mean(feature_outputs, axis=1)  # mean activation for each sample

    for i in range(feature_outputs.shape[0]):
        LiD = Li[i].intersection(D)
        Q_score = (sum(feature_outputs[i, j] - mu[i] for j in LiD)) / len(LiD)
        Q_scores.append(Q_score)

    return Q_scores

def plot_heatmap(Li, Aj, feature_outputs, noise_type, process_name):
    # Initialize a zero matrix for the heatmap
    heatmap_data = torch.zeros(feature_outputs.shape)

    # Fill the matrix: for each sample, set the value of the active features based on Aj
    for i, active_features in enumerate(Li):
        for feature in active_features:
            heatmap_data[i, feature] = Aj[feature]

    # Plotting the heatmap
    plt.figure(figsize=(20, 5))
    single_color_cmap = plt.cm.Blues
    sns.heatmap(heatmap_data, cmap=single_color_cmap)
    plt.title(f'Feature Activation Heatmap during {process_name} with {noise_type} Model')
    plt.xlabel("Features")
    plt.ylabel("Samples")
    plt.savefig(f'./exp6/Heatmap_{noise_type} during {process_name} data.png')
    plt.close()

def energy_score_fn(logits, temperature=1.0):
    return temperature * torch.logsumexp(logits / temperature, dim=-1)


def alpha_req(
    tensor,
    s = None,
    epsilon = 1e-12,
    **_
):
  """Implementation of the Alpha-ReQ metric.

  This metric is defined in "α-ReQ: Assessing representation quality in
  self-supervised learning by measuring eigenspectrum decay". Agrawal et al.,
  NeurIPS 2022.

  Args:
    tensor (dense matrix): Input embeddings.
    s (optional, dense vector): Singular values of `tensor`.
    epsilon (float): Numerical epsilon.

  Returns:
    float: Alpha-ReQ metric value.
  """
  tensor_cpu = tensor.cpu()
  if s is None:
    s = np.linalg.svd(tensor_cpu.detach().numpy(), compute_uv=False)
  n = s.shape[0]
  s = s + epsilon
  features = np.vstack([np.linspace(1, 0, n), np.ones(n)]).T
  a, _, _, _ = np.linalg.lstsq(features, np.log(s), rcond=None)
  return a[0]



# RankeMe score function for embeddings
def rankme(embeddings, normalize=True):
    if normalize:
        embeddings = torch.nn.functional.normalize(embeddings, dim=1, p=2)
    s = torch.linalg.svdvals(embeddings)
    s = s / torch.sum(torch.abs(s))
    s = torch.exp(-torch.sum(s * torch.log(s)))
    return s.item()

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


#_, term_width = os.popen('stty size', 'r').read().split()
#term_width = int(term_width)

#TOTAL_BAR_LENGTH = 65.
#last_time = time.time()
#begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

