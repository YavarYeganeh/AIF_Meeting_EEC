import tensorflow as tf
import numpy as np


log_2_pi = np.log(2.0*np.pi)
log_2_pi_e = np.log(2.0*np.pi*np.e)


def kl_div_loss_analytically_from_logvar_and_precision(mu1, logvar1, mu2, logvar2, omega):
    return 0.5*(logvar2 - tf.math.log(omega) - logvar1) + (tf.exp(logvar1) + tf.math.square(mu1 - mu2)) / (2.0 * tf.exp(logvar2) / omega) - 0.5


def kl_div_loss_analytically_from_logvar_and_precision_mstable(mu1, logvar1, mu2, logvar2, omega):
    term1 = 0.5 * (logvar2 - tf.math.log(omega + 1e-10) - logvar1)
    term2 = (tf.exp(logvar1) + tf.square(mu1 - mu2)) / (1e-10 + 2.0 * tf.exp(logvar2) / (omega + 1e-10)) - 0.5

    return term1 + term2


def kl_div_loss_analytically_from_logvar(mu1, logvar1, mu2, logvar2):
    return 0.5*(logvar2 - logvar1) + (tf.exp(logvar1) + tf.math.square(mu1 - mu2)) / (2.0 * tf.exp(logvar2)) - 0.5


def kl_div_loss(mu1, var1, mu2, var2, axis=1):
    return tf.reduce_sum(kl_div_loss_analytically(mu1, var1, mu2, var2), axis)


@tf.function
def entropy_normal_from_logvar(logvar):
    return 0.5*(log_2_pi_e + logvar)


def entropy_bernoulli(p, displacement = 0.00001):
    return - (1-p)*tf.math.log(displacement + 1 - p) - p*tf.math.log(displacement + p)


def log_bernoulli(x, p, displacement = 0.00001):
    return x*tf.math.log(displacement + p) + (1-x)*tf.math.log(displacement + 1 - p)


def calc_reward_prod(o):
    perfect_reward = np.ones((1),dtype=np.float32)
    return log_bernoulli(o[:,-1], perfect_reward)


def total_correlation(data):
    Cov = np.cov(data.T)
    return 0.5*(np.log(np.diag(Cov)).sum() - np.linalg.slogdet(Cov)[1])


""" Activation for "po" with the structure 10 + 1 + 6*5 + 3""" 
@tf.function
def separate_softmax_sigmoid(x): 
    buffer_index = 10 + 1 
    buffer_softmax = tf.nn.softmax(x[:, :buffer_index])
    for i in range(6):
        m = tf.nn.softmax(x[:, buffer_index + i*5:buffer_index + (i+1)*5])
        try:
            machine_cat_softmax = tf.concat([machine_cat_softmax, m], axis=-1)
        except:
            machine_cat_softmax = m
    reward_sigmoid = tf.reshape(tf.nn.sigmoid(x[:,-3:]),[-1,3])
    return tf.concat([buffer_softmax, machine_cat_softmax, reward_sigmoid], axis=-1)


@tf.function
def po_max_sampler(po): 
    '''This function is designed with an assumption of po_dim = 10 + 1 + 6*5 + 3 
    It samples based on the max Bernoulli probabilities and no time exists! 
    '''

    po_dim = 10 + 1 + 6*5 + 3
    if po.shape[-1] != po_dim:
        raise Exception("The dimension of po is not 10+1+(6*5)+3, which is the assumption for the function, po_max_sampler")

    buffer_index = 10 + 1

    max_buff = tf.argmax(po[:, :buffer_index], axis=1)
    spo = tf.one_hot(max_buff, depth=po[:, :buffer_index].shape[1])

    machine_shape = po[:, buffer_index: buffer_index + 5].shape[1]

    for i in range(6):
        max_state = tf.argmax(po[:, buffer_index + i*5: buffer_index + (i+1)*5], axis=1)
        spo = tf.concat([spo, tf.one_hot(max_state, depth=machine_shape)], axis=1)

    # reward
    spo = tf.concat([spo, tf.cast(tf.reshape(po[:, -3:], (-1, 3)), dtype=tf.float32)], axis=1)

    return spo
    

def synchronize_target_network(model_top):
    model_top.qpi_net_target.set_weights(model_top.qpi_net.get_weights())


@tf.function
def column_remover(t, columns_to_remove):
    
    columns_to_keep = [i for i in range(t.shape[1]) if i not in columns_to_remove]
    # Use tf.gather to select the desired columns
    output_tensor = tf.gather(t, columns_to_keep, axis=1)

    return output_tensor
