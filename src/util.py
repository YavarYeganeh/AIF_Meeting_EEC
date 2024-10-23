import numpy as np
import random
import tensorflow as tf

np_precision = np.float32

pi_one_hot = np.array([[1.0,0.0,0.0,0.0,0.0,0.0,0.0], [0.0,1.0,0.0,0.0,0.0,0.0,0.0], [0.0,0.0,1.0,0.0,0.0,0.0,0.0], [0.0,0.0,0.0,1.0,0.0,0.0,0.0], [0.0,0.0,0.0,0.0,1.0,0.0,0.0], [0.0,0.0,0.0,0.0,0.0,1.0,0.0], [0.0,0.0,0.0,0.0,0.0,0.0,1.0]], dtype=np_precision)
pi_repeated = np.tile(pi_one_hot,(1, 1))


def softmax_multi_with_log(x, single_values=7, eps=1e-20, temperature=10.0): # actions are seven
    """Compute softmax values for each sets of scores in x."""
    x = x.reshape(-1, single_values)
    x = x - np.max(x,1).reshape(-1,1) # Normalization
    e_x = np.exp(x/temperature)
    SM = e_x / e_x.sum(axis=1).reshape(-1,1)
    logSM = x - np.log(e_x.sum(axis=1).reshape(-1,1) + eps) # to avoid infs
    return SM, logSM


def compare_reward(o1, po1):
    ''' Using MSE. '''
    logpo1 = np.square(o1[:,-1] - po1[:,-1]).mean(axis=-1)
    return logpo1


def buffer_recon(o1, po1):
    ''' Difference in the buffer level prediction.
     Po1 is a distribution (normalized Bernoulli) and should be treated accordingly or be sampled! Right now it utlizes the expectation of the prediction.! 
       '''
    
    bn = np.arange(11) # Despite the assumption of buffcap=10, there will be an error.
    
    bo1 = bn * o1[:,:11]
    bo1 = bo1.sum(1)
    bpo1 = bn * po1[:,:11]
    bpo1 = tf.reduce_sum(bpo1, axis=1)
    
    distance = np.linalg.norm(bo1 - bpo1) # Euclidean distance
    mae = tf.reduce_mean(tf.abs(bo1 - bpo1))
    return distance, mae


def machine_state_recon(o1, po1):
    ''' calculates for how many machines the state is predicted correctly.
     This function is designed with an assumption of po_dim = 10  + 1 + 6*5 + 3 '''

    po_dim = 10  + 1 + 6*5 + 3
    if po1.shape[-1] != po_dim:
        raise Exception("The dimension of the p_observation is not 10  + 1 + 5*6 + 3, which is the assumption for the function, machine_state_recon")

    # removing buffer and reward with copy to make sure not modifying the original array
    co1 = np.copy(o1)[:,11:-3]
    cpo1 = np.copy(po1)[:,11:-3]

    recon = np.multiply(co1,cpo1).sum(1) # for how many machines the state is predicted correctly
    recon = recon.mean() # average over batch
    return recon 
   

def prod_planner(o, batch_id, model):

    if model.decision == 'aif':

        o = np.array(o, dtype=np_precision).reshape(1,-1)
        o_repeated = o.repeat(7,0) # The 0th dimensionS

        sum_G, sum_terms, po2 = model.calculate_G_prod(o_repeated)

        terms1 = -sum_terms[0]
        terms12 = -sum_terms[0]+sum_terms[1]

        Ppi = sum_G.numpy() # new hybrid aif, which is already similar to Ppi
        log_Ppi = np.log(Ppi + 1e-20)  # 1e-20 is added for numerical stability

        pi_choice = np.random.choice(7,p=Ppi)

        # One hot version..
        pi0 = np.zeros(7, dtype=np_precision)
        pi0[pi_choice] = 1.0

        model.o_i[batch_id].append(o)
        model.o_t[batch_id].append(o)

        model.a_t[batch_id].append((pi0, log_Ppi))
        model.a_i[batch_id].append(pi_choice)

        # for the seven action
        model.efe[batch_id]['G'].append(sum_G)
        model.efe[batch_id]['q_term'].append(sum_terms[0])  # for hybrid
        model.efe[batch_id]['term0'].append(-sum_terms[1])  # G = - term0 + term1 + term2
        model.efe[batch_id]['term1'].append(sum_terms[2])
        model.efe[batch_id]['term2'].append(sum_terms[3])

        return pi_choice
    
    elif model.decision == 'random':

        pi_choice = np.random.choice(7)
        return pi_choice
    
    else:

        raise ValueError(f'The selected decision type, i.e., {model.decision} is not defined!')



class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def record_interaction(self, o0, o1, pi0, r, os0, os1, pis):
        experience = (o0, o1, pi0, r, os0, os1, pis)
        self.add(experience)

    def clear_buffer(self):
        self.buffer = []
        self.position = 0

    def get_last_interaction(self):
        if len(self.buffer) < 2:
            return None, None, None, None

        last_experience = self.buffer[self.position - 1]
        second_last_experience = self.buffer[self.position - 2]

        return second_last_experience, last_experience

    def add(self, experience):
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position % self.capacity] = experience
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def size(self):
        return len(self.buffer)


def batch_observe(env, model, sim_interval, batch_size, steps):

    if min(len(model.o_i[i]) for i in range(batch_size)) < steps:
        
        existing_steps_o = False

        s = 0
        sim_time = env.now
        while not existing_steps_o:

            env.run(until=sim_time + sim_interval)
            sim_time += sim_interval

            # check whether at least `steps` number of observations exist in the recordings
            if min(len(model.o_i[i]) for i in range(batch_size)) >= steps:
                existing_steps_o = True
            else:
                s += 1
                if s >= 1000 * steps:
                    for j in range(batch_size):
                        print(f"\nThe workstation {j} doesn't have {steps} observations after {s * sim_interval} seconds. It only has {len(model.o_i[j])} observations.\n")
                    raise Exception('Too long running the simulation, but without having enough observations!')


    # empty the lists to obtain new observations 
    for i in range(batch_size):
        model.o_t[i] = []
        model.a_t[i] = []

    new_o_for_all = False

    s = 0

    sim_time = env.now
    while not new_o_for_all:

        env.run(until=sim_time + sim_interval)
        sim_time += sim_interval

        # we check whether 2nd element of observations is obtained for all
        try: 
            [model.o_t[b][1] for b in range(batch_size)] # check
            new_o_for_all = True
            # s = 0
        except:
            s += 1
            if s>=1000:
                for j in range(batch_size):
                    try:
                        model.o_t[j][1]
                    except:
                        print("\nThe following workstation hasn't called decsion for at least 999 steps:\n", len(model.o_i[j]),model.o_i[j][-1], 'with action:', model.a_i[j][-1],'\n')
                        # print(system.systems[j].__dict__, '\n', system.systems[j].workstations[0].__dict__, '\n', system.systems[j].workstations[0].machines[0].__dict__, '\n',system.systems[j].workstations[0].machines[0].__dict__, '\n',system.systems[j].workstations[0].machines[1].__dict__, '\n',system.systems[j].workstations[0].machines[2].__dict__, '\n',system.systems[j].workstations[0].machines[3].__dict__, '\n',system.systems[j].workstations[0].machines[4].__dict__, '\n',system.systems[j].workstations[0].machines[5].__dict__, '\n')
                # print(model.o_t)
                raise Exception('Too long for running events but without having all batches!')

    o0 = [model.o_t[b][0] for b in range(batch_size)]
    o0 = np.array(o0, dtype=np_precision).reshape(batch_size,-1)

    o1 = [model.o_t[b][1] for b in range(batch_size)]  
    o1 = np.array(o1, dtype=np_precision).reshape(batch_size,-1)
    r = o1[:,-1].reshape(batch_size,-1)  # reward/preference for updating the q

    pi0 = [model.a_t[b][0][0] for b in range(batch_size)]
    pi0 = np.array(pi0, dtype=np_precision).reshape(batch_size,-1)
    
    log_Ppi = [model.a_t[b][0][1] for b in range(batch_size)]
    log_Ppi = np.array(log_Ppi, dtype=np_precision).reshape(batch_size,-1)

    # collecting sequence observations and actions with size of number of steps
    os0, os1, pis = [], [], []
    for i in range(batch_size):
        os0.append(model.o_i[i][-(steps+1)])
        os1.append(model.o_i[i][-1])
        pis.append(model.a_i[i][-(steps+1):-1])

    os0 = np.array(os0, dtype=np_precision).reshape(batch_size, -1)
    os1 = np.array(os1, dtype=np_precision).reshape(batch_size, -1)
    pis = np.array(pis, dtype=np_precision).reshape(batch_size, -1)  

    return o0, o1, pi0, log_Ppi, r, os0, os1, pis

""" creating/resting dictionary for observations and efe """
def reset_recordings(model, batch_size):

    model.o_t = {}
    model.o_i = {}
    model.a_t = {}
    model.a_i = {}
    model.efe = {}
    for i in range(batch_size):
        model.o_t[i] = []
        model.o_i[i] = []
        model.a_t[i] = []
        model.a_i[i] = []
        model.efe[i] = {'G': [], 'q_term': [], 'term0': [], 'term1': [], 'term2': []} # this step is necessary  as EFE for the training and training should be recorded separately