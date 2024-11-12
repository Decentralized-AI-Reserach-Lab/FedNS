import numpy as np
import time
import GPUtil

def none_or_str(value):
    if value == 'None':
        return None
    return value

def grab_gpu(memory_limit=0.91, max_wait_time=600):
    start_time = time.time()
    while time.time() - start_time < max_wait_time:
        cuda_device_ids = GPUtil.getAvailable(order='memory', limit=len(GPUtil.getGPUs()), maxLoad=1.0, maxMemory=memory_limit)

        if cuda_device_ids:
            return str(cuda_device_ids[0])
        else:
            print("Waiting for available GPU...")
            time.sleep(10)

    raise RuntimeError("No GPU available within the maximum wait time.")

def create_iid_shards(idxs, num_shards, num_samples, num_classes, seed=42):
    np.random.seed(seed)
    data_distribution = np.random.choice(a=np.arange(0,num_shards), size=num_samples).astype(int)
    return {id:list(np.squeeze(np.argwhere((np.squeeze([data_distribution==id])==True)))) for id in range(num_shards)}

def create_imbalanced_shards(idxs, num_shards, num_samples, num_classes, skewness=0.8, seed=42):
    np.random.seed(seed)
    data_distribution = np.random.choice(a=np.arange(0,num_shards), size=num_samples, p=np.random.dirichlet(np.repeat(skewness, num_shards))).astype(int)
    return {id:list(np.squeeze(np.argwhere((np.squeeze([data_distribution==id])==True)))) for id in range(num_shards)}

def create_noniid_shards(idxs, num_shards, num_samples, num_classes, skewness=0.5, seed=42,):
    np.random.seed(seed)
    partitions = {}
    min_size = 0
    min_require_size = 10
    while min_size < min_require_size:
        idx_batch = [[] for _ in range(num_shards)]
        for k in range(num_classes):
            idx_k = np.where(idxs==k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(skewness, num_shards))
            proportions = np.array([p * (len(idx_j) < num_samples / num_shards) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])
        for j in range(num_shards):
            np.random.shuffle(idx_batch[j])
            partitions[j] = idx_batch[j]
    return partitions

def get_split_fn(name='iid'):
    if name == 'iid':
        return create_iid_shards
    elif name == 'noniid':
        return create_noniid_shards
    elif name == 'imbalanced':
        return create_imbalanced_shards
    else:
        raise ValueError("Invalid name provided. Supported names are 'iid', 'noniid', and 'imbalanced'.")
