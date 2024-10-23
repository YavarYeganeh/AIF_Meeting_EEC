
import os, time, numpy as np, argparse, random
import pickle
import simpy
import datetime
import numpy as np

np_precision = np.float32

# tf only on cpu 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
# errors
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# If the machine used does not have enough memory
memory_restriction = False
if memory_restriction:
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession
    config = ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.8
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

# Import agent modules
from src.prod_environment import System
import src.util as u
import src.tfloss as loss
from src.tfmodel import ActiveInferenceModel
from src.tfutils import *


parser = argparse.ArgumentParser(description='Training script.')
parser.add_argument('-b', '--batch', type=int, default=1, help='Select batch size.')
parser.add_argument('-g', '--gamma', type=float, default=0.0, help='Select gamma for balance in the hybrid planning') # gamma_hybrid hyperparameter responsible for controlling the balance between short and long horizon in EFE
parser.add_argument('-s', '--steps', type=int, default=30, help='How many actions the transition considers') 
parser.add_argument('--samples', type=int, default=10, help='How many samples should be used to calculate EFE') 
parser.add_argument('--calc_mean', action='store_true', help='Whether mean should be considered during calculation of EFE')
args = parser.parse_args()


'''
a: The sum a+d show the maximum value of omega
b: This shows the average value of D_kl[pi] that will cause half sigmoid (i.e. d+a/2)
c: This moves the steepness of the sigmoid
d: This is the minimum omega (when sigmoid is zero)
'''
var_a = 1.0;         var_b = 25.0;          var_c = 5.0;         var_d = 1.5
s_dim = 512;          pi_dim = 7;            beta_s = 1.0;        beta_o = 1.0;
gamma = 0.0;         gamma_rate = 0.01;     gamma_max = 0.8;     gamma_delay = 30
l_rate_top = 1e-04;  l_rate_mid = 1e-04;    l_rate_down = 1e-04
ROUNDS = 1000;       epochs = 40;  # 8 hours of time span is ~ 3000 steps
replay_capacity = 500; replay_size = 200 #200;  ## replay capacity of initial 500!
discount = 0.99; update_target_frequency = 500;
test_frequency = 1; record_frequency = 1;


random_seed = 0

sim_interval = 0.5 #1 #4
o_dim = 10 + 1 + 6*5 + 3 
po_dim = o_dim  # all the times are removed in the observation and prediction
random_init_time = 1 * 24 * 60 * 60  # 10 days
test_time = 1 * 24 * 60 * 60  # 1 days

signature = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
signature += '_gamma-EFE-' +str(args.gamma) + '_' + str(args.steps) + '_' + str(args.samples) + '_' + str(s_dim) + '_' + str(args.calc_mean) + '_' + str(args.batch) + '_' + str(l_rate_down) + '_' + str(l_rate_top) + '_' + str(ROUNDS)
signature = signature.replace('.', '-')
folder = './results'
folder_results = folder + '/' + signature
folder_chp = folder + '/checkpoints'

try: os.mkdir(folder)
except: print('Folder already exists!!')
try: os.mkdir(folder_results)
except: print('Folder results creation error')
try: os.mkdir(folder_chp)
except: print('Folder chp creation error')

# fixing the random seeds before initializing the model
random.seed(random_seed)
np.random.seed(random_seed)
tf.random.set_seed(random_seed)


print(f'gamma_hybrid (balance between in hybrid EFE) = {args.gamma}')
print(f'Transition steps = {args.steps}')

model = ActiveInferenceModel(s_dim=s_dim, pi_dim=pi_dim, gamma=gamma, steps=args.steps, beta_s=beta_s, beta_o=beta_o, o_dim=o_dim, po_dim=po_dim, gamma_hybrid=args.gamma, samples=args.samples, calc_mean=args.calc_mean) 
replay_buffer = u.ReplayBuffer(capacity=replay_capacity)

# training performance will be recorded here; EFE will also be retrieved from the the model parameters
stats_start = {
    'F': {},
    'F_top': {},
    'F_mid': {},
    'F_down': {},
    'kl_div_s': {},
    'kl_div_s_anal': {},
    'kl_div_s_naive': {},
    'kl_div_s_naive_anal': {},
    'bin_cross_entr_o': {},
    'kl_div_pi': {},
    'reward_mse': {},
    'buffer_recon': {},
    'machine_state_recon': {},
    'reward': {},
    'test_G': {},
    'test_q_term': {},
    'test_term0': {},
    'test_term1': {},
    'test_term2': {},
    'test_G_mean': {},
    'test_q_term_mean': {},
    'test_term0_mean': {},
    'test_term1_mean': {},
    'test_term2_mean': {},
    'test_reward': {},
    'test_reward_energy': {},
    'test_reward_prod': {},
    'test_reward_average': {},
    'test_throughput_loss': {},
    'test_energy_saving': {},
    'test_improving_consum_part': {},
    'best_test' : {},
}
stats = stats_start

optimizers = {}
optimizers['top'] = tf.keras.optimizers.Adam(learning_rate=l_rate_top)
optimizers['mid'] = tf.keras.optimizers.Adam(learning_rate=l_rate_mid)
optimizers['down'] = tf.keras.optimizers.Adam(learning_rate=l_rate_down)

best_test_reward = 0
best_test_epoch = 0


start_epoch = 1
start_time = time.time()
for epoch in range(start_epoch, epochs + 1):

    u.reset_recordings(model, batch_size=args.batch)

    # Stats epoch init
    stats['F'][epoch] = []
    stats['F_top'][epoch] = []
    stats['F_mid'][epoch] = []
    stats['F_down'][epoch] = []
    stats['bin_cross_entr_o'][epoch] = []
    stats['kl_div_s'][epoch] = []
    stats['kl_div_s_anal'][epoch] = []
    stats['kl_div_s_naive'][epoch] = []
    stats['kl_div_s_naive_anal'][epoch] = []
    stats['kl_div_pi'][epoch] = []
    stats['reward_mse'][epoch] = []
    stats['buffer_recon'][epoch] = []
    stats['machine_state_recon'][epoch] = [] 
    stats['reward'][epoch] = []
    stats['test_G'][epoch] = []

    
    if epoch > gamma_delay and model.model_down.gamma < gamma_max:
            model.model_down.gamma.assign(model.model_down.gamma+gamma_rate)

    env=simpy.Environment()
    env_test=simpy.Environment()

    system = System(env=env, number_of_systems=args.batch, dmodel=model, warmup=False)  
    system_test = System(env=env_test, number_of_systems=1, dmodel=model, warmup=False)  

    # systems warmup, which also removes the profile
    system.warmup()  
    system_test.warmup() 

    # triggering decisions from the AIF module/algorithm rather than "ALL ON" of the system
    system.dmodel_from_now()
    system_test.dmodel_from_now()

    # systems' initialization with random agent; this will create a good preference start (i.e., ~0.67) to improve
    model.decision = 'random'
    env.run(until=env.now + random_init_time)
    env_test.run(until=env_test.now + random_init_time)
    model.decision = 'aif'
    print(f'Epoch {epoch}: Random init is now done! Decisions are now based the AIF model!')
    
    # pre-filling of the replay buffer with the model (which is random at this beginning)
    while replay_buffer.size() < replay_size:
        o0, o1, pi0, log_Ppi, r, os0, os1, pis = u.batch_observe(env, model, sim_interval, batch_size=args.batch, steps=args.steps)     
        replay_buffer.record_interaction(o0=o0, o1=o1, pi0=pi0, r=r, os0=os0, os1=os1, pis=pis)

    # untrained testing and stat recording 
    if epoch == 1:  
        
        print('Initial testing')
        epoch = 0  # after recording performance of untrained agent it makes the the epoch 1

        u.reset_recordings(model, batch_size=args.batch)

        env_test.run(until = env_test.now + test_time)

        # creating a query form the last time window of the testing system
        rewards = system_test.systems[0].reward()
        performances = system_test.systems[0].performance()

        print(f'Testing in epoch {epoch} (untrained): Rewards -> {rewards} - Performances -> {performances}')
        
        # recording efe for the testing 
        # one test env batch -> it's id is 0 then
        stats['test_G'][epoch] = np.mean(model.efe[0]['G'], axis=0)
        stats['test_q_term'][epoch] = np.mean(model.efe[0]['q_term'], axis=0)
        stats['test_term0'][epoch] = np.mean(model.efe[0]['term0'], axis=0)
        stats['test_term1'][epoch] = np.mean(model.efe[0]['term1'], axis=0)
        stats['test_term2'][epoch]= np.mean(model.efe[0]['term2'], axis=0)
        stats['test_G_mean'][epoch] = np.mean(model.efe[0]['G'])
        stats['test_q_term_mean'][epoch] = np.mean(model.efe[0]['q_term'])
        stats['test_term0_mean'][epoch] = np.mean(model.efe[0]['term0'])
        stats['test_term1_mean'][epoch] = np.mean(model.efe[0]['term1'])
        stats['test_term2_mean'][epoch] = np.mean(model.efe[0]['term2'])
        stats['test_reward'][epoch] = rewards[-1]
        stats['test_reward_energy'][epoch] = rewards[-2]
        stats['test_reward_prod'][epoch] = rewards[-3]
        stats['test_reward_average'][epoch] = np.mean(np.array(model.o_t[0]).reshape(-1, o_dim)[:, -1]) # model.o_t[i] was rest at the beginning of testing
        stats['test_throughput_loss'][epoch] = performances[-3]
        stats['test_energy_saving'][epoch] = performances[-2]
        stats['test_improving_consum_part'][epoch] = performances[-1]

        if stats['test_reward'][epoch] > best_test_reward:

            best_test_reward = stats['test_reward'][epoch]
            best_test_epoch = epoch
            stats['best_test'][epoch] = epoch
            
        stats['best_test'][epoch] = best_test_epoch

        u.reset_recordings(model, batch_size=args.batch)
        
        # saving the initial stats dictionary for further analysis during training
        stats_path = folder_results + '/stats_epoch_' + str(epoch) + '.pkl'

        # Save the dictionary to a binary file using pickle
        with open(stats_path, 'wb') as file:
            pickle.dump(stats, file)

        epoch = 1  # now this make the epoch 1 for the first training

    for round in range(ROUNDS):

        """" 
        This part of the code is responsible for having a new observation for all the systems in the batch
        we check whether i_th element of observation is obtained for all
        sim_interval is better to be small to prevent having more than one observation in either case before update
        """
  
        o0, o1, pi0, log_Ppi, r, os0, os1, pis = u.batch_observe(env, model, sim_interval, batch_size=args.batch, steps=args.steps) # obtaining new observation for all systems in the batch

        batch = replay_buffer.sample(replay_size -1)
        o0b, o1b, pi0b, rb, os0b, os1b, pisb = map(list, zip(*batch)) #zip(*batch)
        update_size = args.batch*replay_size

        # ensuring the last interaction is added
        o0b.append(o0) 
        o1b.append(o1)
        pi0b.append(pi0) 
        rb.append(r)
        os0b.append(os0)
        os1b.append(os1)
        pisb.append(pis)
        
        # now recording the interaction in the replay buffer
        replay_buffer.record_interaction(o0=o0, o1=o1, pi0=pi0, r=r, os0=os0, os1=os1, pis=pis)

        # creating np arrays for training 
        o0b = np.array(o0b, dtype=np_precision).reshape(update_size,-1)
        o1b = np.array(o1b, dtype=np_precision).reshape(update_size,-1)
        pi0b = np.array(pi0b, dtype=np_precision).reshape(update_size,-1)
        rb = np.array(rb, dtype=np_precision).reshape(update_size,-1)
        os0b = np.array(os0b, dtype=np_precision).reshape(update_size,-1)
        os1b = np.array(os1b, dtype=np_precision).reshape(update_size,-1)
        pisb = np.array(pisb, dtype=np_precision).reshape(update_size,-1)
        
    
        # -- TRAIN TOP LAYER --------------------------------------------------- 
        """"
            can be trained with replay buffer; however, omega can't be computed with replay buffer as it is following the current decision/policy (i.e., log_Ppi) and therefore should be computed with the latest policy
           """

        # dqn
        F_top = loss.train_model_top(model_top=model.model_top, o0=o0b, o1=o1b, pi0=pi0b, r=rb, optimizer=optimizers['top'], gamma=discount)

        if round % update_target_frequency == 0:
            synchronize_target_network(model.model_top)
              
        current_omega, kl_div_pi = loss.compute_omega(model=model, o0=o0, log_Ppi=log_Ppi, a=var_a, b=var_b, c=var_c, d=var_d)

        current_omega = current_omega.reshape(-1,1)

        current_omega_b = np.repeat(current_omega, repeats=(o0b.shape[0]/args.batch), axis=0)  # broadcasting current omega for the batch as omega can't be computed with replay buffer as it is following the current decision/policy (i.e., log_Ppi) and therefore should be computed with the latest policy

        # -- TRAIN MIDDLE LAYER ------------------------------------------------ can be trained with replay buffer
        qss0b, _, _ = model.model_down.encoder_with_sample(os0b)
        qss1b_mean, qss1b_logvar = model.model_down.encoder(os1b)
        pss1b_mean, pss1b_logvar = loss.train_model_mid(model_mid=model.model_mid, s0=qss0b, qs1_mean=qss1b_mean, qs1_logvar=qss1b_logvar, Ppi_sampled=pisb, omega=current_omega_b, optimizer=optimizers['mid'])  

        # -- TRAIN DOWN LAYER -------------------------------------------------- can be trained with replay buffer
        loss.train_model_down(model_down=model.model_down, o1=os1b, ps1_mean=pss1b_mean, ps1_logvar=pss1b_logvar, omega=current_omega_b, optimizer=optimizers['down'])
        
        # -- COMPUTING LOSS TERMS -------------------------------------------------- 
        F_mid, loss_terms_mid, pss1b, pss1b_mean, pss1b_logvar = loss.compute_loss_mid(model_mid=model.model_mid, s0=qss0b, Ppi_sampled=pisb, qs1_mean=qss1b_mean, qs1_logvar=qss1b_logvar, omega=current_omega_b)
        F_down, loss_terms, pos1b, qss1b = loss.compute_loss_down(model_down=model.model_down, o1=os1b, ps1_mean=pss1b_mean, ps1_logvar=pss1b_logvar, omega=current_omega_b)

        # # train-level stats
        stats['F'][epoch].append(np.mean(F_down) + np.mean(F_mid) + np.mean(F_top))
        stats['F_top'][epoch].append(np.mean(F_top))
        stats['F_mid'][epoch].append(np.mean(F_mid))
        stats['F_down'][epoch].append(np.mean(F_down))
        stats['bin_cross_entr_o'][epoch].append(np.mean(loss_terms[0]))
        stats['kl_div_s'][epoch].append(np.mean(loss_terms[1]))
        stats['kl_div_s_anal'][epoch].append(np.mean(loss_terms[2],axis=0))
        stats['kl_div_s_naive'][epoch].append(np.mean(loss_terms[3]))
        stats['kl_div_s_naive_anal'][epoch].append(np.mean(loss_terms[4],axis=0))
        stats['kl_div_pi'][epoch].append(np.mean(kl_div_pi))
        stats['reward_mse'][epoch].append(np.mean(u.compare_reward(o1b, pos1b)))
        stats['buffer_recon'][epoch].append(u.buffer_recon(o1b, pos1b)[0]) # using Euclidean distance over batches
        stats['machine_state_recon'][epoch].append(u.machine_state_recon(o1b, pos1b))
        stats['reward'][epoch].append(np.mean(o1[:,-1]))  # batch isn't necessary

    # Only recording means of each epoch of train-level stats (to reduce the size of the file)
    stats['F'][epoch] = np.mean(stats['F'][epoch])
    stats['F_top'][epoch] = np.mean(stats['F_top'][epoch])
    stats['F_mid'][epoch] = np.mean(stats['F_mid'][epoch])
    stats['F_down'][epoch] = np.mean(stats['F_down'][epoch])
    stats['bin_cross_entr_o'][epoch] = np.mean(stats['bin_cross_entr_o'][epoch])
    stats['kl_div_s'][epoch] = np.mean(stats['kl_div_s'][epoch])
    stats['kl_div_s_anal'][epoch] = np.mean(stats['kl_div_s_anal'][epoch] )
    stats['kl_div_s_naive'][epoch] = np.mean(stats['kl_div_s_naive'][epoch])
    stats['kl_div_s_naive_anal'][epoch] = np.mean(stats['kl_div_s_naive_anal'][epoch])
    stats['kl_div_pi'][epoch] = np.mean(stats['kl_div_pi'][epoch])
    stats['reward_mse'][epoch] = np.mean(stats['reward_mse'][epoch])
    stats['buffer_recon'][epoch] = np.mean(stats['buffer_recon'][epoch])
    stats['machine_state_recon'][epoch] = np.mean(stats['machine_state_recon'][epoch])
    stats['reward'][epoch] = np.mean(stats['reward'][epoch])

    # testing
    if epoch % test_frequency == 0: 

        u.reset_recordings(model, batch_size=args.batch)

        env_test.run(until = env_test.now + test_time)

        # creating a query form the last time window of the testing system
        rewards = system_test.systems[0].reward()
        performances = system_test.systems[0].performance()

        print(f'Testing in epoch {epoch}: Rewards -> {rewards} - Performances -> {performances}')
        
        # recording efe for the testing 
        # one test env batch -> it's id is 0 then
        # stats['test_G'][epoch].append(model.efe[0]['G']) # excluded due to the stats file size
        stats['test_G'][epoch] = np.mean(model.efe[0]['G'], axis=0)         
        stats['test_q_term'][epoch] = np.mean(model.efe[0]['q_term'], axis=0)
        stats['test_term0'][epoch] = np.mean(model.efe[0]['term0'], axis=0)
        stats['test_term1'][epoch] = np.mean(model.efe[0]['term1'], axis=0)
        stats['test_term2'][epoch]= np.mean(model.efe[0]['term2'], axis=0)
        stats['test_G_mean'][epoch] = np.mean(model.efe[0]['G'])
        stats['test_q_term_mean'][epoch] = np.mean(model.efe[0]['q_term'])
        stats['test_term0_mean'][epoch] = np.mean(model.efe[0]['term0'])
        stats['test_term1_mean'][epoch] = np.mean(model.efe[0]['term1'])
        stats['test_term2_mean'][epoch] = np.mean(model.efe[0]['term2'])
        stats['test_reward'][epoch] = rewards[-1]
        stats['test_reward_energy'][epoch] = rewards[-2]
        stats['test_reward_prod'][epoch] = rewards[-3]
        stats['test_reward_average'][epoch] = np.mean(np.array(model.o_t[0]).reshape(-1, o_dim)[:, -1]) # model.o_t[i] was rest at the beginning of testing
        stats['test_throughput_loss'][epoch] = performances[-3]
        stats['test_energy_saving'][epoch] = performances[-2]
        stats['test_improving_consum_part'][epoch] = performances[-1]

        if stats['test_reward'][epoch] > best_test_reward:

            best_test_reward = stats['test_reward'][epoch]
            best_test_epoch = epoch
            stats['best_test'][epoch] = epoch
            
        stats['best_test'][epoch] = best_test_epoch

        u.reset_recordings(model, batch_size=args.batch)

    if epoch % record_frequency == 0:

        # saving the stats dictionary for further analysis during training
        stats_path = folder_results + '/stats_epoch_' + str(epoch) + '.pkl'

        # Save the dictionary to a binary file using pickle
        with open(stats_path, 'wb') as file:
            pickle.dump(stats, file)


# saving the stats dictionary for further analysis 
stats_path = folder_results + '/stats_final.pkl'

# save the dictionary to a binary file using pickle
with open(stats_path, 'wb') as file:
    pickle.dump(stats, file)
