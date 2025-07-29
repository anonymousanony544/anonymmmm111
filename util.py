import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch import nn

def loaddata(data_name):
    path = data_name
    data = pd.read_csv(path, header=None)
    return data.values.reshape(-1, 63, 10, 10)

def normalize(data):
    min_val = torch.min(data)
    max_val = torch.max(data)
    if max_val - min_val == 0:
        return torch.zeros_like(data)
    normalized_data = 2 * (data - min_val) / (max_val - min_val) - 1
    return normalized_data

def process_d(speed, demand, inflow):
    speed = torch.clamp(speed, max=140)
    demand = torch.clamp(demand, max=100)
    inflow = torch.clamp(inflow, max=100)

    normalized_speed = normalize(speed)
    normalized_demand = normalize(demand)
    normalized_inflow = normalize(inflow)

    res_speed = normalized_speed.unsqueeze(1).reshape(-1, 63, 100).float()
    res_demand = normalized_demand.unsqueeze(1).reshape(-1, 63, 100).float()
    res_inflow = normalized_inflow.unsqueeze(1).reshape(-1, 63, 100).float()

    return res_speed, res_demand, res_inflow

def process_small(speed, demand, inflow):
    speed = torch.clamp(speed, max=140)
    demand = torch.clamp(demand, max=100)
    inflow = torch.clamp(inflow, max=100)

    normalized_speed = normalize(speed)
    normalized_demand = normalize(demand)
    normalized_inflow = normalize(inflow)

    res_speed = normalized_speed.unsqueeze(1).reshape(-1, 12, 1, 4, 10, 10).float()
    res_demand = normalized_demand.unsqueeze(1).reshape(-1, 12, 1, 4, 10, 10).float()
    res_inflow = normalized_inflow.unsqueeze(1).reshape(-1, 12, 1, 4, 10, 10).float()

    return res_speed, res_demand, res_inflow

def loaddata_city(city, prefix):
    assert city in ['XA', 'CD'], f"Unsupported city: {city}"

    speed = np.load(f'{prefix}speed_{city}.npy')
    inflow = np.load(f'{prefix}inflow_{city}.npy')
    demand = np.load(f'{prefix}demand_{city}.npy')

    speed = torch.tensor(speed).float()
    inflow = torch.tensor(inflow).float()
    demand = torch.tensor(demand).float()

    return speed, inflow, demand

def process_d_city(speed, inflow, demand):
    def normalize(data):
        min_val = torch.min(data)
        max_val = torch.max(data)
        if max_val - min_val == 0:
            return torch.zeros_like(data)
        return 2 * (data - min_val) / (max_val - min_val) - 1

    def process_tensor(tensor, max_val):
        tensor = torch.clamp(tensor, max=max_val)
        train = tensor[:, :, :3]
        test = tensor[:, :, 3:]
        train_norm = normalize(train)
        test_norm = normalize(test)
        return torch.cat([train_norm, test_norm], dim=2)

    speed = process_tensor(speed, 140)
    inflow = process_tensor(inflow, 100)
    demand = process_tensor(demand, 100)

    return speed, inflow, demand
