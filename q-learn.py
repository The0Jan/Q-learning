import gym
import numpy as np
import matplotlib.pyplot as plt
from frozen_lake import FrozenLakeEnv
from numpy import exp
from random import Random

T_CYAN = '\u001b[36m'
RESET = '\u001b[0m'


# Epsilon Greedy Strategy
def choose_action(Q, pos, rng, e ):
    prob_tab = [0, 0, 0, 0]
    direct = [0,1,2,3]

    max_v = max( Q[pos][act] for act in range(4))

    for i in direct:
        prob = 0
        if Q[pos][i] == max_v:
            prob += 1 - e
        prob += e/4
        prob_tab[i] = prob

    result = rng.choices(direct, prob_tab)
    return result[0]

# Adjusting the Q parameter according to the Q-learning algorithm
def adjust_Q(Q, pos, new_pos, action, reward, Bt, y):

    result = Bt*(reward + y*max(Q[new_pos]) - Q[pos][action])
    return result


# A single trip made by our agent through ice
def single_try(frost, Q, Bt, y, e, rng, show ):
    observed = frost.reset()
    done = False

    for i in range(10000):
        if show == True:
            frost.render()
        
        current_position = observed
        action = choose_action(Q, current_position, rng, e)
        observed, reward, done, info = frost.step(action)
        Q[current_position, action] += adjust_Q(Q, current_position, observed, action, reward, Bt, y)
        
        if done == True:
            break
    return Q, reward

# Performs the full Q-learning algorithm. Additionaly shows two plots for the results. Returns the Q table.
def perform_the_walks(num_of_sets, set_time, Bt, y, e, name, rng,seed, show = False):

    frost = FrozenLakeEnv(map_name = '8x8', is_slippery= True)
    frost.seed(seed)
    Q = np.zeros((64,4))

    rewards_list = []
    total_rewards_list = []
    times_list = []
    total_reward = 0
    full_set_times = 0

    for _ in range(num_of_sets):
        set_reward = 0

        for _ in range(set_time):
            Q, reward = single_try(frost, Q, Bt, y, e, rng, show)
            set_reward += reward
        
        full_set_times += set_time
        total_reward += set_reward

        rewards_list.append(set_reward)
        times_list.append(full_set_times)
        total_rewards_list.append(total_reward)
        print(total_reward, full_set_times)

    goal_per_set = 'Goal reached per set:' + name
    goals_total = 'Total_times goal was achieved:' + name
    draw_result(times_list, rewards_list, goal_per_set, 0)
    draw_result(times_list, total_rewards_list, goals_total, 1)

    return Q

# Draws graph for given parameters
def draw_result(x, y, name, mode):

    if mode == 0:
        plt.plot(x, y)
        plt.ylabel('Times goal was reached')
    elif mode == 1:
        plt.plot(x,y)
        plt.ylabel('Total amount goal was reached')

    plt.xlabel('Number of episodes')

    plt.title(name)
    plt.show()

# Gives out the politics array in a readable manner
def politics (q):
    m = [
            "SFFFFFFF",
            "FFFFFFFF",
            "FFFHFFFF",
            "FFFFFHFF",
            "FFFHFFFF",
            "FHHFFFHF",
            "FHFFHFHF",
            "FFFHFFFG"
        ]

    st = ""
    for i in range(len(q)):
        if(i % 8 == 0):
            st += "\n"
            print(st)
            st = ""
        st += m[i // 8][i % 8]
        direct = max(enumerate(q[i]), key=lambda t: t[1])[0]
        if(direct == 0):
            st += T_CYAN + "< "  + RESET 
        if(direct == 1):
            st += T_CYAN + "v " + RESET
        if(direct == 2):
            st += T_CYAN + "> "+ RESET
        if(direct == 3):
            st += T_CYAN + "^ "+ RESET
    st += "\n"
    print(st)


# Setting the rng seed
seed = 1
rng = Random(seed)

#first = perform_the_walks(100, 1000, 0.8, 0.96 ,0.1, 'test_1', rng, seed, False)
#politics(first)

second = perform_the_walks(100, 100, 0.8, 0.96 ,0.5, 'test_2', rng, seed)
third = perform_the_walks(100, 100, 0.8, 0.96 ,0.9, 'test_3', rng, seed)
politics(second)
politics(third)


