import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch.optim as optim
from collections import deque
from torch.nn.functional import one_hot, log_softmax, softmax, normalize
from torch.distributions import Categorical
from actor_model import ACTNet
from critric_model import CRITRICNet
# from torch.utils.tensorboard import SummaryWriter
class Policy:
    def __init__(self,param):

        self.embed_mat=np.load('kbpdata/vec.npy')
        self.train_token = np.load('kbpdata/train_token.npy')
        self.train_labels = np.load('kbpdata/train_labels.npy')

        self.test_token = np.load('kbpdata/test_token.npy')
        self.test_labels = np.load('kbpdata/test_labels.npy')

        self.actor=ACTNet(embed_dim=param.EMBED_DIM,vocab_size=len(self.embed_mat),hidden_dim=param.HIDDEN_DIM,embed_matrix=self.embed_mat)
        self.actor.cuda()

        self.critic=CRITRICNet(embed_dim=param.EMBED_DIM,vocab_size=len(self.embed_mat),hidden_dim=param.HIDDEN_DIM,embed_matrix=self.embed_mat)
        self.critic.cuda()

        self.adam = optim.Adam(params=self.actor.parameters(), lr=0.001)
        self.total_rewards = deque([], maxlen=100)
        LR = 0.001
        self.optimizer = torch.optim.Adam(self.critic.parameters(), lr=LR)
        #损失函数
        self.loss_function = nn.BCELoss()
        #训练批次大小
        self.act_epochs=param.ACT_NUM_EPOCHS
        self.critic_epochs=param.CRITIC_NUM_EPOCHS
        self.train_batch_size=param.TRAIN_BATCH_SIZE
        #划分训练数据和测试数据
        self.train_x=list(self.train_token)
        self.train_y=list(self.train_labels)

        self.test_x=list(self.test_token)
        self.test_y=list(self.test_labels)

        self.test_batch_size=param.TEST_BATCH_SIZE
        self.BETA = param.BETA
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def solve_environment(self):
        """
            The main interface for the Policy Gradient solver
        """
        # init the episode and the epoch
        epoch = 0

        while epoch < self.act_epochs:
            # init the epoch arrays
            # used for entropy calculation
            epoch_logits = torch.empty(size=(0, 2), device=self.DEVICE)
            epoch_weighted_log_probs = torch.empty(size=(0,), dtype=torch.float, device=self.DEVICE)

            # Sample BATCH_SIZE models and do average
            for i in range(0,(int)(len(self.train_x)/self.train_batch_size)+1):

                # play an episode of the environment
                b_x = Variable(torch.LongTensor(self.train_x[i*self.train_batch_size:i*self.train_batch_size+self.train_batch_size]))
                b_x=b_x.cuda()
                (episode_weighted_log_prob_trajectory,
                    episode_logits,
                    sum_of_episode_rewards) = self.play_episode(b_x)

                # after each episode append the sum of total rewards to the deque
                self.total_rewards.append(sum_of_episode_rewards)

                # append the weighted log-probabilities of actions
                epoch_weighted_log_probs = torch.cat((epoch_weighted_log_probs, episode_weighted_log_prob_trajectory),
                                                        dim=0)
                # append the logits - needed for the entropy bonus calculation
                epoch_logits = torch.cat((epoch_logits, episode_logits), dim=0)

            # calculate the loss
            loss, entropy = self.calculate_loss(epoch_logits=epoch_logits,
                                                weighted_log_probs=epoch_weighted_log_probs)

            # zero the gradient
            self.adam.zero_grad()

            # backprop
            loss.backward()

            # update the parameters
            self.adam.step()

            # feedback
            print("\r", f"Epoch: {epoch}, Avg Return per Epoch: {np.mean(self.total_rewards):.3f}",
                    end="",
                    flush=True)


            # check if solved
            # if np.mean(self.total_rewards) > 200:
            #     print('\nSolved!')
            #     break
            epoch += 1
    # close the writer

# train_x=list(train_token)
# train_y=list(train_labels)

# test_x=list(test_token)
# test_y=list(test_labels)
    def play_episode(self,x):
        """
            Plays an episode of the environment.
            episode: the episode counter
            Returns:
                sum_weighted_log_probs: the sum of the log-prob of an action multiplied by the reward-to-go from that state
                episode_logits: the logits of every step of the episode - needed to compute entropy for entropy bonus
                finished_rendering_this_epoch: pass-through rendering flag
                sum_of_rewards: sum of the rewards for the episode - needed for the average over 200 episode statistic
        """
        # get the action logits from the agent - (preferences)
        episode_logits = self.actor(x)

        # sample an action according to the action distribution
        action_index = Categorical(logits=episode_logits).sample().unsqueeze(1)

        mask = one_hot(action_index, num_classes=2)

        episode_log_probs = torch.sum(mask.float() * log_softmax(episode_logits, dim=1), dim=1)

        # append the action to the episode action list to obtain the trajectory
        # we need to store the actions and logits so we could calculate the gradient of the performance
        #episode_actions = torch.cat((episode_actions, action_index), dim=0)

        # Get action actions
        # generate a submodel given predicted actions
        # relabel()

        
        # print(train_y)
        re_train_x,re_train_y=self.relabel()

        net = self.critic
        #net = Net()

        criterion = nn.BCELoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(self.critic_epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            for i in range(0,(int)(len(re_train_x)/self.train_batch_size)+1):

                # play an episode of the environment
                train_x_batch = Variable(torch.LongTensor(re_train_x[i*self.train_batch_size:i*self.train_batch_size+self.train_batch_size]))
                train_x_batch=train_x_batch.cuda()
                
                train_y_batch = Variable(torch.FloatTensor(re_train_y[i*self.train_batch_size:i*self.train_batch_size+self.train_batch_size]))
                train_y_batch=train_y_batch.cuda()
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(train_x_batch)
                loss = criterion(outputs, train_y_batch)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                            (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

        print('Finished Training')

        # load best performance epoch in this training session
        # model.load_weights('weights/temp_network.h5')

        # evaluate the model
        correct = 0
        total = 0
        with torch.no_grad():
            for i in range(0,(int)(len(self.test_x)/self.test_batch_size)+1):
                # play an episode of the environment
                test_x_batch = Variable(torch.LongTensor(self.test_x[i*self.test_batch_size:i*self.test_batch_size+self.test_batch_size]))
                test_x_batch=test_x_batch.cuda()
                labels= Variable(torch.LongTensor(self.test_y[i*self.test_batch_size:i*self.test_batch_size+self.test_batch_size]))
                labels=labels.cuda()
                outputs = self.critic(test_x_batch)
                predicted = torch.round(outputs.data)
                print(predicted)
                # predicted = torch.LongTensor(predicted)
                predicted= predicted.cuda()
                total += labels.size(0)
                # correct += (predicted.cpu().numpy() == labels.cpu().numpy()).sum().item()
                y_true= labels.cpu().numpy()
                y_pred= predicted.cpu().numpy()
                TP = np.sum(np.logical_and(np.equal(y_true, 1), np.equal(y_pred, 1)))   # 10
                # #TP = np.sum(np.multiply(y_true, y_pred)) #同样可以实现计算TP
                # # False Positive:即y_true中为0但是在y_pred中被识别为1的个数
                # FP = np.sum(np.logical_and(np.equal(y_true, 0), np.equal(y_pred, 1)))   #  0
                # # False Negative:即y_true中为1但是在y_pred中被识别为0的个数
                # FN = np.sum(np.logical_and(np.equal(y_true, 1), np.equal(y_pred, 0)))  # 6
                # # True Negative:即y_true与y_pred中同时为0的个数
                TN = np.sum(np.logical_and(np.equal(y_true, 0), np.equal(y_pred, 0)))  # 34


        acc = 100 * (TP+TN) / total
        print((TP+TN),total)
        print('Accuracy of the network on the 10000 test images: {}'.format(acc))

        # compute the reward
        reward = acc

        episode_weighted_log_probs = episode_log_probs * reward
        sum_weighted_log_probs = torch.sum(episode_weighted_log_probs).unsqueeze(dim=0)

        return  sum_weighted_log_probs, episode_logits, reward

    def calculate_loss(self,epoch_logits: torch.Tensor, weighted_log_probs: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
            Calculates the policy "loss" and the entropy bonus
            Args:
                epoch_logits: logits of the policy network we have collected over the epoch
                weighted_log_probs: loP * W of the actions taken
            Returns:
                policy loss + the entropy bonus
                entropy: needed for logging
        """
        policy_loss = -1 * torch.mean(weighted_log_probs)

        # add the entropy bonus
        p = softmax(epoch_logits, dim=1)
        log_p = log_softmax(epoch_logits, dim=1)
        entropy = -1 * torch.mean(torch.sum(p * log_p, dim=1), dim=0)
        entropy_bonus = -1 * 0.1 * entropy

        return policy_loss + entropy_bonus, entropy
# solve_environment()
    def relabel(self):
        act_label=[]
        with torch.no_grad():
            for i in range(0,(int)(len(self.train_x)/self.train_batch_size)+1):

                # play an episode of the environment
                train_x_batch = Variable(torch.LongTensor(self.train_x[i*self.train_batch_size:i*self.train_batch_size+self.train_batch_size]))
                train_x_batch=train_x_batch.cuda()
                act_logits = self.actor(train_x_batch)

        # sample an action according to the action distribution
                action_i = Categorical(logits=act_logits).sample().unsqueeze(1).cpu().numpy()
                action_i=list(action_i)
                act_label+=action_i
        act_label=np.array(act_label)
        # act_label= act_label.reshape(-1)
        # print(train_labels)
        # print(act_label)
        train_y = np.concatenate((self.train_labels,act_label),axis=1)
        return self.train_x,train_y