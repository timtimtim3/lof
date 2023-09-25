import gym 
import envs 
from algorithms.value_iteration import ValueIteration
from time import sleep
from task_specifications import *
import numpy as np

env = gym.make("DeliveryMini-v0")

def learn_options(env):

    OPTIONS = {}
    algorithm = ValueIteration(env)
     
    for e in env.unwrapped.exit_states.values():

        Q, V, To = algorithm.train(e)

        OPTIONS[e] = {'Q':Q, 'Ro': V, 'To':To}


    return OPTIONS

def lof_value_iteration(env, OPTIONS, fsa):

    F = fsa.states 
    Sdim = env.unwrapped.s_dim 
    exit_states = env.unwrapped.exit_states
    
    Q = np.zeros((len(F), Sdim, len(OPTIONS)))
    V = np.zeros((len(F), Sdim))

    # LOF-VI
    iters = 0

    while True:

        iters+=1

        Q_old = Q.copy()
        V_old = V.copy()

        for fidx, sidx in np.ndindex((len(F), Sdim)):
            
            f = F[fidx]

            if f in ('u0', 'u1', 'u3'):
                continue

            for oidx, o in enumerate(OPTIONS):

                # Qo = OPTIONS[o]['Q']
                To = OPTIONS[o]['To']
                Ro = OPTIONS[o]['Ro']

                # Eq. 3 LOF paper
                next_fsa_states = fsa.get_neighbors(F[fidx])[0]

                exit_state_option = exit_states[oidx]
                exit_state_idx = env.unwrapped.coords_to_state[exit_state_option]                

                aux =  To[sidx, exit_state_idx] * V_old[F.index(next_fsa_states), exit_state_idx] 

                print(To[sidx, exit_state_idx])
                            
                Q[fidx, sidx, oidx] = Ro[sidx] + np.sum(aux)
        
        V = Q.max(axis=2)

        if np.allclose(V, V_old):
            print("Done", iters)
            break


    mu = Q.argmax(axis=2)

    print(mu)

    
    for f, s in np.ndindex(mu.shape):
        if f in (0,1,3):
            continue
        state =  np.unravel_index(s, (env.unwrapped.height, env.unwrapped.width))
        print(F[f], state, env.unwrapped.MAP[state],  mu[f, s], Q[f, s, :])



if __name__ == "__main__":

    env = gym.make("DeliveryMini-v0")
 
    # STEP 1: Learn options
    OPTIONS = learn_options(env)

    # STEP 2: learn the metapolicy for a given FSA

    fsa = fsa_delivery_mini1()

    lof_value_iteration(env, OPTIONS, fsa)


        
