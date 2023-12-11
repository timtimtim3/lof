from algorithms.tasks import *
import networkx as nx
from matplotlib import pyplot as plt

def load_fsa(name: str, env):    

    if name == "delivery_task1":
        init_fun = fsa_delivery1
    elif name == "delivery_task2":
        init_fun = fsa_delivery2
    elif name == "delivery_task3":
        init_fun = fsa_delivery3
    elif name == "office_task1":
        init_fun = fsa_office1
    elif name == "office_task2":
        init_fun = fsa_office2
    elif name == "office_task3":
        init_fun = fsa_office3
    elif name == "double_slit_task1":
        init_fun = fsa_double_slit1
    else:
        raise NameError()
    g = init_fun(env)
    nx.draw(g[0].graph, with_labels=True)
    plt.savefig("test.png")
    return g