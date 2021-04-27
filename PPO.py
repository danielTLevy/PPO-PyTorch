import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import torch_geometric.nn.conv as graph_conv
import pdb

################################## set device ##################################

print("============================================================================================")


# set device to cpu or cuda
device = torch.device('cpu')

if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
    
print("============================================================================================")




################################## PPO Policy ##################################


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init):
        super(ActorCritic, self).__init__()


        self.action_dim = action_dim
        self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

        # actor
        self.actor = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, action_dim),
                        nn.Tanh()
                    )


        
        # critic
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1)
                    )
        
    def set_action_std(self, new_action_std):
        self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)

    def forward(self):
        raise NotImplementedError
    

    def act(self, state):

        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(action_mean, cov_mat)


        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob.detach()
    

    def evaluate(self, state, action):
        action_mean = self.actor(state)
        
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)
        dist = MultivariateNormal(action_mean, cov_mat)
        
        # For Single Action Environments.
        if self.action_dim == 1:
            action = action.reshape(-1, self.action_dim)


        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy


class Embedder(nn.Module):
    def __init__(self, state_dim, embedding_dim, n_nodes, input_dict,
                                 ob_size_dict, node_type_dict, node_to_type):
        super(Embedder, self).__init__()
        self.n_nodes = n_nodes
        self.embedding_dim = embedding_dim
        self.ob_size_dict = ob_size_dict
        self.node_to_type = node_to_type
        self.embedder_gather = {k:  torch.LongTensor(v).to(device)
                                    for k, v in input_dict.items()}
        self.node_embedders = {node_type:
                            nn.Linear(ob_size_dict[node_type],  embedding_dim)
                            for node_type in node_type_dict}

    def forward(self, state):
        embedded = torch.zeros(state.shape[0], self.n_nodes, self.embedding_dim).to(device)
        for node, obs_idx in self.embedder_gather.items():
            node_input = torch.index_select(state, 1, obs_idx)
            embedded[:, node, :] =  self.node_embedders[self.node_to_type[node]](node_input)
        return embedded
    
    
class GGNN(nn.Module):
    def __init__(self, hidden_dim, edge_idx):
        super(GGNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.edge_idx = edge_idx
        self.n_edges = edge_idx.shape[1]
        self.gnn = graph_conv.GatedGraphConv(out_channels=hidden_dim, num_layers=4,
                                             aggr='add', bias=True)
    
    def forward(self, embedding):
        batch_size = embedding.shape[0]
        n_nodes = embedding.shape[1]
        embedding_dim = embedding.shape[2]
        
        embedding_batched = embedding.view(batch_size*n_nodes, embedding_dim)
        #batch = torch.arange(n_nodes).repeat_interleave(n_nodes)

        batch_offset = n_nodes*torch.arange(batch_size).repeat_interleave(self.n_edges).to(device)
        edge_idx_batched = self.edge_idx.repeat(1, batch_size) + batch_offset
        
        out = self.gnn(embedding_batched, edge_idx_batched)
        return out.view(batch_size, n_nodes, -1)

        
class ActionPredictor(nn.Module):
    def __init__(self, hidden_dim, action_dim, output_list):
        super(ActionPredictor, self).__init__()

        self.action_dim = action_dim
        # List of nodes that have outputs
        self.output_list = torch.LongTensor(output_list).to(device)
        # TODO: this assumes each output body part has only one action
        self.action_output = nn.Linear(hidden_dim, 1)
        self.tanh = nn.Tanh()
    
    
    def forward(self, X):
        X_output = torch.index_select(X, 1, self.output_list)
        return self.tanh(self.action_output(X_output).squeeze())


class NerveNet(nn.Module):
    def __init__(self, state_dim, action_dim, node_info):
        super(NerveNet, self).__init__()

        self.n_nodes = len(node_info['tree'])


        self.embedder = Embedder(state_dim, embedding_dim=5, n_nodes=self.n_nodes,
                                 input_dict=node_info['input_dict'],
                                 ob_size_dict=node_info['ob_size_dict'],
                                 node_type_dict=node_info['node_type_dict'],
                                 node_to_type=node_info['node_to_type']).to(device)

        receive_idx = np.array(node_info['receive_idx'])
        send_idx = np.array(node_info['send_idx'][1])
        self.edge_idx = torch.LongTensor(np.stack((receive_idx, send_idx))).to(device)

        hidden_dim = 64
        self.gnn = GGNN(hidden_dim, self.edge_idx).to(device)

        self.action_predictor = ActionPredictor(hidden_dim, action_dim,
                                                output_list=node_info['output_list']).to(device)

    def forward(self, state):
        if state.ndim == 1:
            state = state.unsqueeze(0)
        embedded = self.embedder(state)
        gnn_out = self.gnn(embedded)
        return self.action_predictor(gnn_out)
    



class ActorCriticNerveNet(ActorCritic):
    def __init__(self, state_dim, action_dim, action_std_init, node_info):
        nn.Module.__init__(self)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_std_init = action_std_init
        self.node_info = node_info

        self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

        # actor            
        self.actor = NerveNet(self.state_dim, self.action_dim, node_info).to(device)


        # critic
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1)
                    )



class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std_init=0.6, node_info=None):


        self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()

    
        self.policy = ActorCriticNerveNet(state_dim, action_dim, action_std_init, node_info).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCriticNerveNet(state_dim, action_dim, action_std_init, node_info).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()


    def set_action_std(self, new_action_std):
        
        self.action_std = new_action_std
        self.policy.set_action_std(new_action_std)
        self.policy_old.set_action_std(new_action_std)


    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")

        self.action_std = self.action_std - action_std_decay_rate
        self.action_std = round(self.action_std, 4)
        if (self.action_std <= min_action_std):
            self.action_std = min_action_std
            print("setting actor output action_std to min_action_std : ", self.action_std)
        else:
            print("setting actor output action_std to : ", self.action_std)
        self.set_action_std(self.action_std)


        print("--------------------------------------------------------------------------------------------")


    def select_action(self, state):

        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob = self.policy_old.act(state)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)

        return action.detach().cpu().numpy().flatten()



    def update(self):

        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)

        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
    
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        
        
       


