import os
import datetime
import torch
import torch.nn as nn
import gymnasium as gym
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical


class Policy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(Policy, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def train(num_epochs = 1000): 
    env = gym.make("CartPole-v1")
    policy = Policy(env.observation_space.shape[0], env.action_space.n)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)
    gamma = 0.99
    batch_size = 5
    batch_losses = []
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if not os.path.exists('runs'):
        os.makedirs('runs')
    log_dir = f'runs/reinforce_{current_time}'
    writer = SummaryWriter(log_dir)

    
    for epoch in tqdm(range(num_epochs)): 
        state, info = env.reset()
        log_probs = []
        ep_reward = []
        
        # use policy to generate trajectory
        while True: 
            action_prob = policy(torch.tensor(state))
            dist = Categorical(action_prob)
            action = dist.sample()

            next_state, reward, terminated, truncated, info = env.step(action.item())
            
            ep_reward.append(reward)
            log_probs.append(dist.log_prob(action))
            state = next_state

            if terminated or truncated: 
                break
        
        # Compute discounted returns
        returns = []
        G = 0
        for r in reversed(ep_reward): 
            G = r + G * gamma
            returns.insert(0, G)

        # Normalize returns to reduce variance
        tensor_return = torch.tensor(returns)
        return_normalized = (tensor_return - tensor_return.mean()) / (tensor_return.std() + 1e-7)
        
        # Compute policy gradient loss
        ep_loss = 0
        for rew, log_prob in zip(return_normalized, log_probs):
            ep_loss += -log_prob * rew

        batch_losses.append(ep_loss/len(returns))

        writer.add_scalar('train/reward_raw', tensor_return.mean(), epoch)
        writer.add_scalar('train/ep_loss', ep_loss/len(returns), epoch)
        writer.add_scalar('train/ep_length', len(returns), epoch)

        # Update policy after batch_size episodes
        if epoch % batch_size == 0 and batch_size != 0: 
            optimizer.zero_grad()
            loss = sum(batch_losses)/len(batch_losses)
            loss.backward()
            optimizer.step()
            batch_losses = []

            writer.add_scalar('train/batch_loss', loss, epoch)



if __name__ == "__main__": 
    train()