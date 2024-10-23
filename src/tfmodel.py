import tensorflow as tf
import pickle
import numpy as np
from shutil import copyfile
from src.tfutils import *
from src.util import softmax_multi_with_log


np_precision = np.float32

@tf.function
def state_activation(x): 
    """""
    Ensures mu is [-1 1] and var [0 1]. 
    """

    # Split the input tensor into two halves
    half_size = x.shape[-1] // 2
    first_half = x[:, :half_size]
    second_half = x[:, half_size:]

    # Apply tanh to the first half
    first_half_tanh = tf.tanh(first_half)

    # # Apply sigmoid followed by log to the second half
    # second_half_log = tf.math.log(tf.nn.sigmoid(second_half) + 1e-10)  # Adding a small epsilon for numerical stability
    # for lambda_s = 1.5
    second_half_sigmoid = tf.nn.sigmoid(second_half)
    second_half_sigmoid_scaled = tf.multiply(second_half_sigmoid, 1.5)
    second_half_log = tf.math.log(second_half_sigmoid_scaled + 1e-10)  # Adding a small epsilon for numerical stability


    # Concatenate the results
    result = tf.concat([first_half_tanh, second_half_log], axis=-1)

    return result


""""" modified for dqn; the top model (q) here maps the observations rather than states to actions in order to prevent another sampling step 
"""
class ModelTop(tf.keras.Model):
    def __init__(self, o_dim, pi_dim, tf_precision, precision):
        super(ModelTop, self).__init__()
        # For activation function we used ReLU.
        # For weight initialization we used He Uniform

        self.tf_precision = tf_precision
        self.precision = precision
        tf.keras.backend.set_floatx(self.precision)

        self.o_dim = o_dim
        self.pi_dim = pi_dim

        self.qpi_net = tf.keras.Sequential([
              tf.keras.layers.InputLayer(input_shape=(o_dim,)),
              tf.keras.layers.Dense(units=64, activation=tf.nn.relu, kernel_initializer='he_uniform'),
              tf.keras.layers.Dropout(0.1),
              tf.keras.layers.Dense(pi_dim),]) # No activation
        
        # for dqn
        self.qpi_net_target = tf.keras.Sequential([
              tf.keras.layers.InputLayer(input_shape=(o_dim,)),
              tf.keras.layers.Dense(units=64, activation=tf.nn.relu, kernel_initializer='he_uniform'),
              tf.keras.layers.Dropout(0.1),
              tf.keras.layers.Dense(pi_dim),]) # No activation

    @tf.function
    def encode_o(self, o0):
        logits_pi = self.qpi_net(o0)
        # q_pi = tf.nn.softmax(logits_pi)
        q_pi = tf.nn.softmax(-logits_pi)
        log_q_pi = tf.math.log(q_pi+1e-20)
        return logits_pi, q_pi, log_q_pi

    # for dqn
    @tf.function
    def encode_o_target(self, o0):
        logits_pi = self.qpi_net_target(o0)
        q_pi = tf.nn.softmax(logits_pi)
        log_q_pi = tf.math.log(q_pi+1e-20)
        return logits_pi, q_pi, log_q_pi

    
class ModelMid(tf.keras.Model):
    def __init__(self, s_dim, steps, tf_precision, precision):
        super(ModelMid, self).__init__()

        self.tf_precision = tf_precision
        self.precision = precision
        tf.keras.backend.set_floatx(self.precision)
        self.s_dim = s_dim

        self.ps_net = tf.keras.Sequential([
            #   tf.keras.layers.InputLayer(input_shape=(pi_dim + s_dim + 1,)),  # delta_t is removed!
              tf.keras.layers.InputLayer(input_shape=(steps + s_dim,)),
              tf.keras.layers.Dense(units=128, activation=tf.nn.relu, kernel_initializer='he_uniform'),
              tf.keras.layers.Dropout(0.1),
              tf.keras.layers.Dense(units=(s_dim + s_dim), activation=state_activation)]) # No activation

    @tf.function
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    @tf.function
    def transition(self, pi, s0):
        mean, logvar = tf.split(self.ps_net(tf.concat([pi,s0],1)), num_or_size_splits=2, axis=1)
        return mean, logvar
    
    @tf.function
    def transition_with_sample(self, pi, s0):
        ps1_mean, ps1_logvar = self.transition(pi, s0)
        ps1 = self.reparameterize(ps1_mean, ps1_logvar)
        return ps1, ps1_mean, ps1_logvar

class ModelDown(tf.keras.Model):
    def __init__(self, s_dim, tf_precision, precision, o_dim, po_dim):
        super(ModelDown, self).__init__()

        self.tf_precision = tf_precision
        self.precision = precision
        tf.keras.backend.set_floatx(self.precision)

        self.s_dim = s_dim
        self.o_dim = o_dim
        self.po_dim = po_dim

        self.qs_net = tf.keras.Sequential([
              tf.keras.layers.InputLayer(input_shape=o_dim),
              tf.keras.layers.Dense(128, activation=tf.nn.relu, kernel_initializer='he_uniform'),
              tf.keras.layers.Dropout(0.1),
              tf.keras.layers.Dense(units=(s_dim + s_dim), activation=state_activation),]) # No activation
        
        self.po_net_base = tf.keras.Sequential([
              tf.keras.layers.InputLayer(input_shape=s_dim),
              tf.keras.layers.Dense(128, activation=tf.nn.relu, kernel_initializer='he_uniform'),
              tf.keras.layers.Dropout(0.1),
              tf.keras.layers.Dense(po_dim),]) # No activation

    @tf.function
    def po_net(self, x):
        o = self.po_net_base(x)
        return separate_softmax_sigmoid(o)     

    @tf.function
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    @tf.function
    def encoder(self, o):
        # tf.print('o.shape',o.shape)
        grad_check = tf.debugging.check_numerics(self.qs_net(o), 'check_numerics: Caught bad numerics for self.qs_net(o)!')
        mean_s, logvar_s = tf.split(self.qs_net(o), num_or_size_splits=2, axis=1)
        return mean_s, logvar_s

    @tf.function
    def decoder(self, s):
        po = self.po_net(s)
        return po

    @tf.function
    def encoder_with_sample(self, o):
        mean, logvar = self.encoder(o)
        s = self.reparameterize(mean, logvar)
        return s, mean, logvar

class ActiveInferenceModel:

    def __init__(self, s_dim, pi_dim, gamma, steps, beta_s, beta_o, o_dim, po_dim, gamma_hybrid, samples, calc_mean):

        self.tf_precision = tf.float32
        self.precision = 'float32'

        self.s_dim = s_dim
        self.pi_dim = pi_dim
        tf.keras.backend.set_floatx(self.precision)

        # self.model_top = ModelTop(s_dim, pi_dim, self.tf_precision, self.precision)
        self.model_top = ModelTop(o_dim, pi_dim, self.tf_precision, self.precision)
        self.model_mid = ModelMid(s_dim, steps, self.tf_precision, self.precision)
        self.model_down = ModelDown(s_dim, self.tf_precision, self.precision, o_dim=o_dim, po_dim=po_dim)

        self.model_down.beta_s = tf.Variable(beta_s, trainable=False, name="beta_s")
        self.model_down.gamma = tf.Variable(gamma, trainable=False, name="gamma")
        self.model_down.beta_o = tf.Variable(beta_o, trainable=False, name="beta_o")
        self.pi_one_hot = tf.Variable([[1.0,0.0,0.0,0.0,0.0,0.0,0.0],
                                       [0.0,1.0,0.0,0.0,0.0,0.0,0.0], 
                                       [0.0,0.0,1.0,0.0,0.0,0.0,0.0], 
                                       [0.0,0.0,0.0,1.0,0.0,0.0,0.0], 
                                       [0.0,0.0,0.0,0.0,1.0,0.0,0.0], 
                                       [0.0,0.0,0.0,0.0,0.0,1.0,0.0], 
                                       [0.0,0.0,0.0,0.0,0.0,0.0,1.0]], trainable=False, dtype=self.tf_precision)
        
        self.decision = 'random'  # this will impact prod_planner in util to initialize with random then it will be changed to self.decision = 'aif' to engage the aif model

        self.gamma_hybrid = gamma_hybrid
        self.samples = samples
        self.calc_mean = calc_mean
        self.steps = steps
        self.pi_steps_repeated = tf.constant(np.tile(np.arange(pi_dim).reshape(-1, 1), (1, steps)), dtype=tf.float32)

        

    def save_weights(self, folder_chp):
        self.model_down.qs_net.save_weights(folder_chp+'/checkpoint_qs')
        self.model_down.po_net.save_weights(folder_chp+'/checkpoint_po')
        if self.pi_dim > 0:
            self.model_top.qpi_net.save_weights(folder_chp+'/checkpoint_qpi')
            self.model_mid.ps_net.save_weights(folder_chp+'/checkpoint_ps')

    def load_weights(self, folder_chp):
        self.model_down.qs_net.load_weights(folder_chp+'/checkpoint_qs')
        self.model_down.po_net.load_weights(folder_chp+'/checkpoint_po')
        if self.pi_dim > 0:
            self.model_top.qpi_net.load_weights(folder_chp+'/checkpoint_qpi')
            self.model_mid.ps_net.load_weights(folder_chp+'/checkpoint_ps')

    def save_all(self, folder_chp, stats, script_file="", optimizers={}):
        self.save_weights(folder_chp)
        with open(folder_chp+'/stats.pkl','wb') as ff:
            pickle.dump(stats,ff)
        with open(folder_chp+'/optimizers.pkl','wb') as ff:
            pickle.dump(optimizers,ff)
        copyfile('src/tfmodel.py', folder_chp+'/tfmodel.py')
        copyfile('src/tfloss.py', folder_chp+'/tfloss.py')
        if script_file != "":
            copyfile(script_file, folder_chp+'/'+script_file)

    def load_all(self, folder_chp):
        self.load_weights(folder_chp)
        with open(folder_chp+'/stats.pkl','rb') as ff:
            stats = pickle.load(ff)
        try:
            with open(folder_chp+'/optimizers.pkl','rb') as ff:
                optimizers = pickle.load(ff)
        except:
            optimizers = {}
        if len(stats['var_beta_s'])>0: self.model_down.beta_s.assign(stats['var_beta_s'][-1])
        if len(stats['var_gamma'])>0: self.model_down.gamma.assign(stats['var_gamma'][-1])
        if len(stats['var_beta_o'])>0: self.model_down.beta_o.assign(stats['var_beta_o'][-1])
        return stats, optimizers


    # modified for the production systems with mean along the batches
    def check_reward(self, o):
        # return tf.reduce_mean(calc_reward_prod(o),axis=[0]) * 10.0 # Incorrect as this not a batch
        reward = calc_reward_prod(o) * 10.0
        return reward


    @tf.function
    def calculate_G_prod(self, o, depth=1):
        """
        We simultaneously calculate G for the seven policies of repeating each
        one of the seven actions continuously..
        """

        # Calculate current s_t
        qs0_mean, qs0_logvar = self.model_down.encoder(o)
        qs0 = self.model_down.reparameterize(qs0_mean, qs0_logvar)

        sum_terms = [tf.zeros([o.shape[0]], self.tf_precision), tf.zeros([o.shape[0]], self.tf_precision), tf.zeros([o.shape[0]], self.tf_precision), tf.zeros([o.shape[0]], self.tf_precision)]
        sum_G = tf.zeros([o.shape[0]], self.tf_precision)

        # Predict s_t+1 for various policies
        if self.calc_mean:
            s0_temp = qs0_mean
        else:
            s0_temp = qs0
        
        
        for t in range(depth):
            # print('h1')
            # G, terms, s1, ps1_mean, po1 = self.calculate_G(s0_temp, pi, samples=samples)
            G, terms, s1, ps1_mean, po1 = self.calculate_G(o, s0_temp, pis=self.pi_steps_repeated, samples=self.samples) # q networks maps o to actions

            sum_terms[0] += terms[0]
            sum_terms[1] += terms[1]
            sum_terms[2] += terms[2]
            sum_terms[3] += terms[3]
            sum_G += G

            if self.calc_mean:
                s0_temp = ps1_mean
            else:
                s0_temp = s1

        return sum_G, sum_terms, po1


    """" Hybrid EFE (q/habit net receives o) """    
    @tf.function
    def calculate_G(self, o0, s0, pis, samples=10):

        # Hybrid
        # log_q_pi = self.model_top.encode_o(o0)[-1]
        # # print("pi0", pi0)
        # q_prod = tf.multiply(log_q_pi, pi0) # pi0 is one-hot vector representing the action 
        q_pi = self.model_top.encode_o(o0)[-2]
        # print("pi0", pi0)
        q_prod = tf.multiply(q_pi, self.pi_one_hot) # pi0 is one-hot vector representing the action 
        # q_term = - tf.reduce_sum(q_prod, axis=1) # controlling the system in opposite direction
        q_term = tf.reduce_sum(q_prod, axis=1)
        # print("q...", log_q_pi, q_prod, q_term, "\n")

        term0 = tf.zeros([s0.shape[0]], self.tf_precision)
        term1 = tf.zeros([s0.shape[0]], self.tf_precision)
        for _ in range(samples):
            ps1, ps1_mean, ps1_logvar = self.model_mid.transition_with_sample(pis, s0)
            po1 = self.model_down.decoder(ps1)
            # print('__________po1', po1.shape)
            # tf.print('__________po1', po1.shape)
            spo1 = po_max_sampler(po1) # po is a distribution with different structure with o1 and can't be directly fed into the encoder
            qs1, _, qs1_logvar = self.model_down.encoder_with_sample(spo1)

            # E [ log P(o|pi) ]  # Eq. 7a
            # print('hhhhh___hhhh>>>>',po1.shape)
            logpo1 = self.check_reward(po1)
            term0 += logpo1

            # E [ log Q(s|pi) - log Q(s|o,pi) ]  # Eq. 7b  # why different from ...
            term1 += - tf.reduce_sum(entropy_normal_from_logvar(ps1_logvar) + entropy_normal_from_logvar(qs1_logvar), axis=1)
        term0 /= float(samples)
        term1 /= float(samples)

        term2_1 = tf.zeros(s0.shape[0], self.tf_precision)
        term2_2 = tf.zeros(s0.shape[0], self.tf_precision)
        for _ in range(samples):
            # Term 2.1: Sampling different thetas, i.e. sampling different ps_mean/logvar with dropout!
            s1_temp1 = self.model_mid.transition_with_sample(pis, s0)[0]
            # tf.print('__________s1_temp1', s1_temp1)
            po1_temp1 = self.model_down.decoder(s1_temp1)
            # tf.print('__________po1_temp1', po1_temp1)#.eval(session=sess)))
            # term2_1 += tf.reduce_sum(entropy_bernoulli(po1_temp1),axis=[1,2,3])
            term2_1 += tf.reduce_sum(entropy_bernoulli(po1_temp1),axis=[1])
            # tf.print('term2_1',term2_1)
            # Term 2.2: Sampling different s with the same theta, i.e. just the reparametrization trick!
            s1_temp2 = self.model_down.reparameterize(ps1_mean, ps1_logvar)
            # tf.print('__________s1_temp2', s1_temp2)
            po1_temp2 = self.model_down.decoder(s1_temp2)
            # tf.print('__________po1_temp2', po1_temp2)
            # term2_2 += tf.reduce_sum(entropy_bernoulli(po1_temp2),axis=[1,2,3])
            term2_2 += tf.reduce_sum(entropy_bernoulli(po1_temp2),axis=[1])
            # tf.print('term2_2',term2_2)
        term2_1 /= float(samples)
        term2_2 /= float(samples)
        # E [ log [ H(o|s,th,pi) ] - E [ H(o|s,pi) ]
        term2 = term2_1 - term2_2
        # print('((((((-------))))))', term2_1, term2_2)

        # G = - term0 + term1 + term2

        # G = self.gamma_hybrid*q_term + (1-self.gamma_hybrid)*(- term0 + term1 + term2)
        efe_term = tf.nn.softmax(-(- term0 + term1 + term2))
        G = self.gamma_hybrid*q_term + (1-self.gamma_hybrid)*(efe_term)
        # tf.print('efe_term ->', efe_term, 'q_term ->', q_term, 'G ->', G)

        # return G, [q_term, term0, term1], ps1, ps1_mean, po1
        return G, [q_term, term0, term1, term2], ps1, ps1_mean, po1  # for hybrid
