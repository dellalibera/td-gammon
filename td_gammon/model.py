import datetime
import random
import time
from itertools import count

import numpy as np
import torch
import torch.nn as nn

from agents import TDAgent, RandomAgent, evaluate_agents
from gym_backgammon.envs.backgammon import WHITE, BLACK

torch.set_default_tensor_type('torch.DoubleTensor')


class BaseModel(nn.Module):
    def __init__(self, lr, lamda, seed=123):
        super(BaseModel, self).__init__()
        self.lr = lr
        self.lamda = lamda  # trace-decay parameter
        self.start_episode = 0

        self.eligibility_traces = None
        self.optimizer = None

        torch.manual_seed(seed)
        random.seed(seed)

    def update_weights(self, p, p_next):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

    def init_weights(self):
        raise NotImplementedError

    def init_eligibility_traces(self):
        self.eligibility_traces = [torch.zeros(weights.shape, requires_grad=False) for weights in list(self.parameters())]

    def checkpoint(self, checkpoint_path, step, name_experiment):
        path = checkpoint_path + "/{}_{}_{}.tar".format(name_experiment, datetime.datetime.now().strftime('%Y%m%d_%H%M_%S_%f'), step + 1)
        torch.save({'step': step + 1, 'model_state_dict': self.state_dict(), 'eligibility': self.eligibility_traces if self.eligibility_traces else []}, path)
        print("\nCheckpoint saved: {}".format(path))

    def load(self, checkpoint_path, optimizer=None, eligibility_traces=None):
        checkpoint = torch.load(checkpoint_path)
        self.start_episode = checkpoint['step']

        self.load_state_dict(checkpoint['model_state_dict'])

        if eligibility_traces is not None:
            self.eligibility_traces = checkpoint['eligibility']

        if optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

    def train_agent(self, env, n_episodes, save_path=None, eligibility=False, save_step=0, name_experiment=''):
        start_episode = self.start_episode
        n_episodes += start_episode

        wins = {WHITE: 0, BLACK: 0}
        network = self

        agents = {WHITE: TDAgent(WHITE, net=network), BLACK: TDAgent(BLACK, net=network)}

        durations = []
        steps = 0
        start_training = time.time()

        for episode in range(start_episode, n_episodes):

            if eligibility:
                self.init_eligibility_traces()

            agent_color, first_roll, observation = env.reset()
            agent = agents[agent_color]

            t = time.time()

            for i in count():
                if first_roll:
                    roll = first_roll
                    first_roll = None
                else:
                    roll = agent.roll_dice()

                p = self(observation)

                actions = env.get_valid_actions(roll)
                action = agent.choose_best_action(actions, env)
                observation_next, reward, done, winner = env.step(action)
                p_next = self(observation_next)

                if done:
                    if winner is not None:
                        loss = self.update_weights(p, reward)

                        wins[agent.color] += 1

                    tot = sum(wins.values())
                    tot = tot if tot > 0 else 1

                    print("Game={:<6d} | Winner={} | after {:<4} plays || Wins: {}={:<6}({:<5.1f}%) | {}={:<6}({:<5.1f}%) | Duration={:<.3f} sec".format(episode + 1, winner, i,
                        agents[WHITE].name, wins[WHITE], (wins[WHITE] / tot) * 100,
                        agents[BLACK].name, wins[BLACK], (wins[BLACK] / tot) * 100, time.time() - t))

                    durations.append(time.time() - t)
                    steps += i
                    break
                else:
                    loss = self.update_weights(p, p_next)

                agent_color = env.get_opponent_agent()
                agent = agents[agent_color]

                observation = observation_next

            if save_path and save_step > 0 and episode > 0 and (episode + 1) % save_step == 0:
                self.checkpoint(checkpoint_path=save_path, step=episode, name_experiment=name_experiment)
                agents_to_evaluate = {WHITE: TDAgent(WHITE, net=network), BLACK: RandomAgent(BLACK)}
                evaluate_agents(agents_to_evaluate, env, n_episodes=20)
                print()

        print("\nAverage duration per game: {} seconds".format(round(sum(durations) / n_episodes, 3)))
        print("Average game length: {} plays | Total Duration: {}".format(round(steps / n_episodes, 2), datetime.timedelta(seconds=int(time.time() - start_training))))

        if save_path:
            self.checkpoint(checkpoint_path=save_path, step=n_episodes - 1, name_experiment=name_experiment)

            with open('{}/comments.txt'.format(save_path), 'a') as file:
                file.write("Average duration per game: {} seconds".format(round(sum(durations) / n_episodes, 3)))
                file.write("\nAverage game length: {} plays | Total Duration: {}".format(round(steps / n_episodes, 2), datetime.timedelta(seconds=int(time.time() - start_training))))

        env.close()


class TDGammonCNN(BaseModel):
    def __init__(self, lr, seed=123, output_units=1):
        super(TDGammonCNN, self).__init__(lr, seed=seed, lamda=0.7)

        self.loss_fn = torch.nn.MSELoss(reduction='sum')

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4),  # CHANNEL it was 3
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.hidden = nn.Sequential(
            nn.Linear(64 * 8 * 8, 80),
            nn.Sigmoid()
        )

        self.output = nn.Sequential(
            nn.Linear(80, output_units),
            nn.Sigmoid()
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def init_weights(self):
        pass

    def forward(self, x):
        # https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
        x = np.dot(x[..., :3], [0.2989, 0.5870, 0.1140])
        x = x[np.newaxis, :]
        x = torch.from_numpy(np.array(x))
        x = x.unsqueeze(0)
        x = x.type(torch.DoubleTensor)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, 64 * 8 * 8)
        x = x.reshape(-1)
        x = self.hidden(x)
        x = self.output(x)
        return x

    def update_weights(self, p, p_next):

        if isinstance(p_next, int):
            p_next = torch.tensor([p_next], dtype=torch.float64)

        loss = self.loss_fn(p_next, p)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss


class TDGammon(BaseModel):
    def __init__(self, hidden_units, lr, lamda, init_weights, seed=123, input_units=198, output_units=1):
        super(TDGammon, self).__init__(lr, lamda, seed=seed)

        self.hidden = nn.Sequential(
            nn.Linear(input_units, hidden_units),
            nn.Sigmoid()
        )

        # self.hidden2 = nn.Sequential(
        #     nn.Linear(hidden_units, hidden_units),
        #     nn.Sigmoid()
        # )

        # self.hidden3 = nn.Sequential(
        #     nn.Linear(hidden_units, hidden_units),
        #     nn.Sigmoid()
        # )

        self.output = nn.Sequential(
            nn.Linear(hidden_units, output_units),
            nn.Sigmoid()
        )

        if init_weights:
            self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            nn.init.zeros_(p)

    def forward(self, x):
        x = torch.from_numpy(np.array(x))
        x = self.hidden(x)
        # x = self.hidden2(x)
        # x = self.hidden3(x)
        x = self.output(x)
        return x

    def update_weights(self, p, p_next):
        # reset the gradients
        self.zero_grad()

        # compute the derivative of p w.r.t. the parameters
        p.backward()

        with torch.no_grad():

            td_error = p_next - p

            # get the parameters of the model
            parameters = list(self.parameters())

            for i, weights in enumerate(parameters):

                # z <- gamma * lambda * z + (grad w w.r.t P_t)
                self.eligibility_traces[i] = self.lamda * self.eligibility_traces[i] + weights.grad

                # w <- w + alpha * td_error * z
                new_weights = weights + self.lr * td_error * self.eligibility_traces[i]
                weights.copy_(new_weights)

        return td_error
