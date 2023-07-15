from collections import defaultdict
from dataclasses import dataclass
import multiprocessing
import threading
from torch import Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Observation:
    id: int
    step: int
    pstate: Tensor
    nstate: Tensor
    action: float
    reward: float
    done: bool
    value: float
    log_probability: float
    prev: int = None
    next: int = None


def worker(wid, action, output, env, model):
    sim = 0
    step = 0

    obs = env.new()
    while True:
        command = action.get()

        if command is not None:
            cmd, args = command
            if cmd == 'STOP':
                break

        action, log_probability = model.sample(obs)
        value = model.value(obs)
        nobs, reward, done = env.step(action)

        output.push(
            Observation(
                id=(wid, sim),
                step=step,
                pstate=obs,
                nstate=nobs,
                action=action,
                reward=reward,
                done=done,
                value=value,
                log_probability=log_probability,
            )
        )

        obs = nobs
        step += 1

        if done:
            if command:
                if cmd == 'UPDATE_MODEL':
                    model.load_state_dict(args)

            obs = env.new()
            sim += 1
            step = 0


class RLDataset:
    """A RLDataset is a replay buffer that acts as a dataset.
    New samples are created using running environment and a offline agent,
    that gets updated from time to time.

    """

    def __init__(self, env, model, size, count) -> None:
        self.env = env
        self.model = model
        self.size = size
        self.counter = 0
        self.workers = []
        self.output = multiprocessing.Queue()
        self.lock = threading.RLock()
        self.thread = threading.Thread(target=self._fetch_observation)
        self.states = []
        self.trajectory = defaultdict(list)
        self.running = True

        for i in range(count):
            action = multiprocessing.Queue()
            worker = multiprocessing.Process(
                target=worker, 
                args=(i, action, self.output, self.env, self.model)
            )
            self.workers.append((action, worker))

        for _, process in self.workers:
            process.start()
        self.thread.start()

    def _fetch_observation(self):
        while self.running:
            obs: Observation = self.output.get()

            with self.lock:
                if len(self.states) < self.size:
                    self.states.append(obs)
                    self.trajectory[obs.id].append(obs)
                else:
                    self.states[self.counter % self.size] = obs
                self.counter += 1
    
    def update_model(self):
        for q, _ in self.workers:
            q.push(("UPDATE_MODEL", self.model.state_dict()))

    def stop(self):
        for q, _ in self.workers:
            q.push(("STOP", None))
                   
        self.running = False
        self.thread.join()

        for _, p in self.workers:
            p.join()


class TrajectorySampler:

    def __init__(self, dataset) -> None:
        self.dataset = dataset


def astensor(x, device):
    if not x is torch.Tensor:
        x = torch.from_numpy(x).float().to(device)
    return x


class ValueNetwork(torch.nn.Module):
    def __init__(self, in_dim=128):
        super(ValueNetwork, self).__init__()

        self.fc1 = torch.nn.Linear(in_dim, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, 128)
        self.fc4 = torch.nn.Linear(128, 1)
        self.l_relu = torch.nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.l_relu(self.fc1(x))
        x = self.l_relu(self.fc2(x))
        x = self.l_relu(self.fc3(x))
        y = self.fc4(x)

        return y.squeeze(1)

    def state_value(self, state):
        if not state is torch.Tensor:
            state = torch.from_numpy(state).float().to(device)

        if len(state.size()) == 1:
            state = state.unsqueeze(0)

        y = self(state)

        return y.item()
    
class PolicyNetwork(nn.Module):
    def __init__(self, n=4, in_dim=128):
        super(PolicyNetwork, self).__init__()  
        self.seq = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, n) 
        )

    def forward(self, x):              
        y = self.seq(x)        
        y = F.softmax(y, dim=-1)       
        return y    

    def sample_action(self, state):        
        if not state is torch.Tensor:
            state = torch.from_numpy(state).float().to(device)        
            
        if len(state.size()) == 1:
            state = state.unsqueeze(0)        

        y = self.forward(state)        

        dist = nn.Categorical(y)        
        action = dist.sample()        
        log_probability = dist.log_prob(action)        

        return action.item(), log_probability.item()    
        
    def best_action(self, state):        
        if not state is torch.Tensor:
            state = torch.from_numpy(state).float().to(device)      

        if len(state.size()) == 1:
            state = state.unsqueeze(0)    

        y = self(state).squeeze()        
        action = torch.argmax(y)        
        return action.item()    
    
    def evaluate_actions(self, states, actions):
        y = self.forward(states) 
        
        dist = nn.Categorical(y)     
        entropy = dist.entropy()        
        log_probabilities = dist.log_prob(actions)  
          
        return log_probabilities, entropy
    

def main():
    import gym

    env = gym.make('CartPole-v1')
    epoch = 100
    replay_buffer = 100000

    dataset = RLDataset()

    for _ in range(epoch):

        for batch in dataset:


from torchvision.models import vit_l_32