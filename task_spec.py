from algorithms.tasks import *  

def load_fsa(name: str, env):    

    if name == "delivery_task1":
        return fsa_delivery1(env)
    elif name == "delivery_task2":
        return fsa_delivery2(env)
    else:
        raise NotImplementedError