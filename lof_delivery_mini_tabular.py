import gym 
import envs 
from algorithms.value_iteration import ValueIteration
from time import sleep
from task_specifications import *
import numpy as np

env = gym.make("DeliveryMini-v0")
global exit_states_idxs

def learn_options(env):

    OPTIONS = {}
    algorithm = ValueIteration(env, gamma=1)
     
    for e in env.unwrapped.exit_states.values():

        Q, V, To = algorithm.train(e)

        OPTIONS[e] = {'Q':Q, 'Ro': V, 'To':To}

    return OPTIONS

def lof_value_iteration(env, OPTIONS, fsa):

    F = fsa.states 
    Sdim = env.unwrapped.s_dim 
    exit_states = list(env.unwrapped.exit_states.values())

    Q = np.zeros((len(F), Sdim, len(OPTIONS)))
    V = np.zeros((len(F), Sdim))

    obstacles_idxs = list(map(lambda x: env.unwrapped.coords_to_state[x], env.unwrapped.obstacles)) 

    T = fsa.get_transition_matrix()

    T = np.tile(T[:, :, None], (1, 1, Sdim))

    T = np.zeros(T.shape)

    T[0, 0, :] = 1
    T[0, 0, obstacles_idxs] = 0
    T[0, 0, exit_states_idxs[0]] = 0
    T[0, 1, exit_states_idxs[0]] = 1 

    T[1, 1, :] = 1
    T[1, 1, obstacles_idxs] = 0
    T[1, 1, exit_states_idxs[1]] = 0
    T[1, 2, exit_states_idxs[1]] = 1

    T[2, 2, :] = 1
    T[2, 2, obstacles_idxs] = 0
    T[2, 2, exit_states_idxs[2]] = 0
    T[2, 3, exit_states_idxs[2]] = 1

    T[3, 3, :] = 1
    T[3, 3, obstacles_idxs] = 0
    T[3, 3, exit_states_idxs[3]] = 0
    T[3, 4, exit_states_idxs[3]] = 1

    T[4, 4, :] = 1


    R = np.ones(len(F))
    R[-1] = 0

    # LOF-VI
    iters = 0

    for _ in range(50):

        for oidx, o in enumerate(OPTIONS):

            # Qo = OPTIONS[o]['Q']
            To = OPTIONS[o]['To']
            Ro = OPTIONS[o]['Ro'] #- 1

            # Eq. 3 LOF paper

            rewards = R * (np.tile(Ro[:, None], [1, len(F)]) - 1)

            aux = np.dot(To, V.T)

            preQ = rewards + aux

            # print(oidx, ((rewards + aux).T).shape)

            Q[:, :, oidx] = preQ.T

        # print('Rewards', rewards[217, :])
        # print('Next V', np.round(V.T[217, :], 2))

        V = Q.max(axis=2)

        print(V.shape)

        preV = np.tile(V[None, ...], (len(F), 1, 1))
        
        V = np.sum(T*preV, axis=1)
    
    for (f, s) in np.ndindex(V.shape):
        print(f, s, V[f, s])
    mu = Q.argmax(axis=2)

    # print(mu)

    
    for f, s in np.ndindex(mu.shape):
        # if f in [2]:
        #     continue
        state =  np.unravel_index(s, (env.unwrapped.height, env.unwrapped.width))
        print(F[f], state, env.unwrapped.MAP[state],  mu[f, s], Q[f, s, :])



if __name__ == "__main__":

    env = gym.make("Delivery-v0")

    

 
    # STEP 1: Learn options
    OPTIONS = learn_options(env)

    # STEP 2: learn the metapolicy for a given FSA

    fsa = fsa_delivery1()

    lof_value_iteration(env, OPTIONS, fsa)


    for option in OPTIONS:

        RO = OPTIONS[option]["Ro"]
        print(RO[exit_states_idxs + [0]])

        
