#TODO: rework history as an array from the start
#      softmax action selection (output vs. softmax after?)
#      l1 regularization??
#      anneal epsilon, learning rate
#      keep track of avg reward in a running average/sum instead of a list
#      consider increasing or lowering penalty for failure
#      reshape network???
#      change the single render parameter into 2: render_now and render_always


import random
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras import regularizers
from keras import optimizers
from keras import backend
import keras.utils  

import gym

def choose_e_greedy(actions, epsilon):
    choice = random.random()
    if choice < epsilon:
        return random.randrange(len_action)
    else:
        return actions.argmax()
    
#takes a keras net and an array of s,s',a,r rows, and trains the net
#also needs: the last row in the history that is valid
#how many entries to train on
def train_on_history(model, history, h_len, train_num, gamma):
    #print('history:', history.shape)
    #print(history)
    
    #first pick episodes to learn from
    learn_list = [random.randrange(0, h_len+1) for _ in range(train_num)]
    #print('h_len', h_len)
    #print('learn_list')
    #print(learn_list)
    #convert history to training data
    observation_array = np.empty( (train_num, len_obs) )
    #print('obs:', observation_array.shape)
    for i, step_i in enumerate(learn_list):
        observation_array[i] = history[step_i][:len_obs]
    #print('observation_array')
    #print(observation_array)
    
    target_array = np.empty( (train_num, len_action) )
    for i,step_i in enumerate(learn_list):
        next_state = history[step_i][len_obs:2*len_obs].reshape(-1,len_obs)
        #h[-2]: reward   
        q_val = history[step_i][-2] + gamma*(model.predict(next_state).max()) #belman equation: reward + next_q
        q_val_array = np.zeros((1, len_action))
        #h[-3]: action we took
        #print("action we took:", history[step_i][-3])
        q_val_array[0][int(history[step_i][-3])] = q_val
        target_array[i] = q_val_array
    #finally, train the model on our data
    model.fit(observation_array, target_array, batch_size=1, epochs=1, verbose=0, shuffle=True)



#initialize the gym
env = gym.make('CartPole-v0')
print('action space:', env.action_space)
len_obs = env.observation_space.shape[0] #note: env observation space only works when obs are 1D
len_action = env.action_space.n

    #initialize the neural net
model = Sequential([
    Dense(24, input_shape=(len_obs,), kernel_regularizer=regularizers.l1(0.0000)),
    Activation('tanh'),
    Dense(24),
    Activation('tanh'),
    Dense(len_action),
    Activation('linear'),
])

opt = optimizers.Adam()
model.compile(optimizer=opt,
              loss = 'mean_squared_error')



#how long to run for
epochs = 250
epoch_length = 200
epoch_reward = 0

#keep track of avg. reward per epoch and over the run
partial_reward = 0
partial_reward_list = []
total_reward = 0
total_eps = epochs*epoch_length

#initialize the history
hist_len = 100000
  #this is an array, the first n values are the state we were in, the next n, the state we got to next
  #the last 3 entries are respectively: action, reward, done
  #done is a -1 if the game did not end, a 1 if it did
len_obs = env.observation_space.shape[0] #note: env observation space only works when obs are 1D
h_obs = np.zeros( (hist_len, len_obs*2 + 3) ) 
  #this is the index of the next place to put something into the history
h_ptr = 0
h_loop = False #whether we have looped and are overwriting history
learn_every = 12 #how many actions to take before learning
learn_num = 10 #how many experiences to learn from each time

#rendering settings
render = False #whether to render at all
render_now = render #whether we are currently rendering
render_num = 0 #how many games per epoch to render


#algorithm settings
e_start = 1
e_final = .05
final_epoch = int(.85 * epochs)
e_change = (e_final - e_start) / final_epoch
epsilon = e_start
gamma = .99

for i_episode in range(epochs*epoch_length):
    if i_episode % epoch_length < render_num:
        render_now = render
    if i_episode %learn_every == 0 and i_episode != 0:
        #if we've filled the history learn from all of it
        learn_len = hist_len-1 if h_loop else h_ptr 
        train_on_history(model, h_obs, learn_len, learn_num, gamma)
    if i_episode%epoch_length == 0 and i_episode != 0:
        print('episode:', i_episode, 'reward avg. last', epoch_length, 'episodes:\n', partial_reward/epoch_length)
        partial_reward_list.append(partial_reward/epoch_length)
        partial_reward = 0
        if i_episode < final_epoch*epoch_length:
            epsilon += e_change
            print("new epsilon:", epsilon)
        else:
            epsilon = e_final
        

        
    episode_reward = 0
    observation = env.reset()
    observation = observation.reshape(-1, len_obs)
    for t in range(10000):
        if render_now:
            env.render()

        #print(observation)
        # action = env.action_space.sample() #to randomly choose an action

        action_weights = model.predict(observation, batch_size = 1)
        #print('output', action_weights, 'type', type(action_weights))
        action = choose_e_greedy(action_weights, epsilon)
        if render_now and t==0:
            print('taking action', action, 'based on', action_weights)
        
        #advance the environment itself
        last_obs = observation
        observation, reward, done, info = env.step(action)
        observation = observation.reshape(-1, len_obs)        
        if done:
            reward = -10
            #this is to resolve the fact that we don't learn when the game ends
            
        #record everything into our history array
        h_obs[h_ptr][:len_obs] = last_obs #the state we took the action in
        h_obs[h_ptr][len_obs:2*len_obs] = observation #the state we got to
        h_obs[h_ptr][-3] = action
        h_obs[h_ptr][-2] = reward
        h_obs[h_ptr][-1] = 1 if done else -1

        h_ptr += 1
        
        if h_ptr >= hist_len:
            #wrap around:
            h_ptr = 0
            h_loop = True #note that we have wrapped around
        
        episode_reward += reward

        if done:
            #print("Episode finished after {} timesteps".format(t+1))
            #print("Total reward accrued:", total_reward)
            partial_reward += episode_reward
            total_reward += episode_reward
            break
    if not done:
        print('time limit exceeded')
        print("Total reward accrued:", episode_reward)
        partial_reward += episode_reward
        total_reward += episode_reward        
    render_now = False

print("average reward accrued:", total_reward / total_eps)
plt.plot(partial_reward_list)
plt.ylabel('avg. reward')
plt.show()
#this code is purely to resolve a minor error in a destructor in Tensorflow
#https://github.com/tensorflow/tensorflow/issues/3388
backend.clear_session()
