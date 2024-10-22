import tensorflow as tf
import numpy as np

from src.tfutils import *

o_dim = 10 + 1 + 6*5 + 3 


"""" this version is modified to have the encoder of q working with o """
@tf.function
def compute_kl_div_pi(model, o0, log_Ppi):
    _, Qpi, log_Qpi = model.model_top.encode_o(o0)

    # TERM: Eqs D_kl[Q(pi|s1,s0)||P(pi)], Categorical K-L divergence
    # --------------------------------------------------------------------------
    return tf.reduce_sum(Qpi*(log_Qpi-log_Ppi), 1)


"""" modified version """
def compute_omega(model, o0, log_Ppi, a, b, c, d):
    o0_stopped = tf.stop_gradient(o0)
    log_Ppi_stopped = tf.stop_gradient(log_Ppi)    
    kl_pi = compute_kl_div_pi(model=model, o0=o0_stopped, log_Ppi=log_Ppi_stopped).numpy()
    return a * ( 1.0 - 1.0/(1.0 + np.exp(- (kl_pi-b) / c)) ) + d, kl_pi


# for dqn
@tf.function
def compute_loss_top(model_top, o0, o1, pi0, r, gamma):
    
    # Q-values for the current state and next state
    q_values_current = model_top.qpi_net(o0)
    q_values_next = model_top.qpi_net_target(o1)  # using the same network for q-values and target q-values

    # Q-value for the chosen actions
    batch_size = tf.shape(q_values_current)[0]
    actions = pi0[:, 0]  # pi0 is already an array of actions 
    actions = tf.cast(actions, tf.int32)
    indices = tf.range(batch_size)
    indices = tf.stack([indices, actions], axis=1)
    q_values_current_selected = tf.gather_nd(q_values_current, indices)

    # Compute target Q-values
    target_q_values = r + gamma * tf.reduce_max(q_values_next, axis=1)

    # Compute loss (MSE)
    loss = tf.reduce_mean(tf.square(q_values_current_selected - target_q_values))

    return loss


@tf.function
def compute_loss_mid(model_mid, s0, Ppi_sampled, qs1_mean, qs1_logvar, omega):
    ps1, ps1_mean, ps1_logvar = model_mid.transition_with_sample(Ppi_sampled, s0)

    # broadcasting omega
    
    # TERM: Eqpi D_kl[Q(s1)||P(s1|s0,pi)]
    # ----------------------------------------------------------------------
    kl_div_s_anal = kl_div_loss_analytically_from_logvar_and_precision(qs1_mean, qs1_logvar, ps1_mean, ps1_logvar, omega)
    kl_div_s = tf.reduce_sum(kl_div_s_anal, 1)

    F_mid = kl_div_s
    loss_terms = (kl_div_s, kl_div_s_anal)

    # tf.print("F_mid, loss_terms, ps1, ps1_mean, ps1_logvar",F_mid, loss_terms, ps1, ps1_mean, ps1_logvar)
    return F_mid, loss_terms, ps1, ps1_mean, ps1_logvar


""" designed for the case time is completely excluded for 11 + 6*5 + 3 o/po """  
@tf.function
def compute_loss_down(model_down, o1, ps1_mean, ps1_logvar, omega, displacement = 0.00001):
    
    qs1_mean, qs1_logvar = model_down.encoder(o1)
    qs1 = model_down.reparameterize(qs1_mean, qs1_logvar)
    po1 = model_down.decoder(qs1)

    # 'bm' for buffer and machine, while 'r' for rewards, parts of the o1/po1  
    o1_bm = column_remover(o1, columns_to_remove=[i for i in range(o_dim - 3)]) 
    o1_r = column_remover(o1, columns_to_remove=[-3,-2,-1]) 
    po1_bm = column_remover(po1, columns_to_remove=[i for i in range(o_dim - 3)]) 
    po1_r = column_remover(po1, columns_to_remove=[-3,-2,-1]) 

    # TERM: Eq[log P(o1|s1)]
    # --------------------------------------------------------------------------
    # bin_cross_entr = o1 * tf.math.log(displacement + po1) + (1 - o1) * tf.math.log(displacement + 1 - po1) # Binary Cross Entropy
    bin_cross_entr = o1_bm * tf.math.log(displacement + po1_bm) + (1 - o1_bm) * tf.math.log(displacement + 1 - po1_bm) # Binary Cross Entropy

    # logpo1_s1 = tf.reduce_sum(bin_cross_entr, axis=[1,2,3])
    logpo1_s1_bm = tf.reduce_sum(bin_cross_entr, axis=[1])

    # Calculate RMSE between o1_r and po_r
    mse = tf.reduce_mean(tf.square(o1_r - po1_r), axis=1)

    # Calculate scaling factor
    mean_bce = tf.reduce_mean(bin_cross_entr)
    mean_mse = tf.reduce_mean(mse)
    scaling_factor = mean_bce / mean_mse

    # Scale RMSE to match magnitude of BCE
    scaled_mse = scaling_factor * mse

    logpo1_s1 = logpo1_s1_bm + scaled_mse


    # TERM: Eqpi D_kl[Q(s1)||N(0.0,1.0)]
    # --------------------------------------------------------------------------
    kl_div_s_naive_anal = kl_div_loss_analytically_from_logvar_and_precision(qs1_mean, qs1_logvar, 0.0, 0.0, omega)
    kl_div_s_naive = tf.reduce_sum(kl_div_s_naive_anal, 1)

    # TERM: Eqpi D_kl[Q(s1)||P(s1|s0,pi)]
    # ----------------------------------------------------------------------
    kl_div_s_anal = kl_div_loss_analytically_from_logvar_and_precision(qs1_mean, qs1_logvar, ps1_mean, ps1_logvar, omega)
    kl_div_s = tf.reduce_sum(kl_div_s_anal, 1)

    # Compute the loss components without considering the condition
    base_loss = - model_down.beta_o * logpo1_s1
    naive_loss_term = model_down.beta_s * kl_div_s_naive
    loss_term = model_down.beta_s * kl_div_s

    # Compute the loss with the condition using tf.where
    F = tf.where(model_down.gamma <= 0.05, base_loss + naive_loss_term,
                tf.where(model_down.gamma >= 0.95, base_loss + loss_term,
                        base_loss + model_down.beta_s * (model_down.gamma * kl_div_s + (1.0 - model_down.gamma) * kl_div_s_naive)))

        
    loss_terms = (-logpo1_s1, kl_div_s, kl_div_s_anal, kl_div_s_naive, kl_div_s_naive_anal)
    return F, loss_terms, po1, qs1




""""" For dqn """
@tf.function
def train_model_top(model_top, o0, o1, pi0, r, optimizer, gamma):
    
    o0_stopped = tf.stop_gradient(o0)
    o1_stopped = tf.stop_gradient(o1)
    pi0_stopped = tf.stop_gradient(pi0)
    r_stopped = tf.stop_gradient(r)
    
    with tf.GradientTape() as tape:
        loss = compute_loss_top(model_top=model_top, o0=o0_stopped, o1=o1_stopped, pi0=pi0_stopped, r=r_stopped, gamma=gamma)
    
    gradients = tape.gradient(loss, model_top.qpi_net.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model_top.qpi_net.trainable_variables))

    return loss


@tf.function
def train_model_mid(model_mid, s0, qs1_mean, qs1_logvar, Ppi_sampled, omega, optimizer):
    s0_stopped = tf.stop_gradient(s0)
    qs1_mean_stopped = tf.stop_gradient(qs1_mean)
    qs1_logvar_stopped = tf.stop_gradient(qs1_logvar)
    Ppi_sampled_stopped = tf.stop_gradient(Ppi_sampled)
    omega_stopped = tf.stop_gradient(omega)
    with tf.GradientTape() as tape:
        F, loss_terms, ps1, ps1_mean, ps1_logvar = compute_loss_mid(model_mid=model_mid, s0=s0_stopped, Ppi_sampled=Ppi_sampled_stopped, qs1_mean=qs1_mean_stopped, qs1_logvar=qs1_logvar_stopped, omega=omega_stopped)
        gradients = tape.gradient(F, model_mid.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model_mid.trainable_variables))
    return ps1_mean, ps1_logvar


@tf.function
def train_model_down(model_down, o1, ps1_mean, ps1_logvar, omega, optimizer):
    ps1_mean_stopped = tf.stop_gradient(ps1_mean)
    ps1_logvar_stopped = tf.stop_gradient(ps1_logvar)
    omega_stopped = tf.stop_gradient(omega)
    with tf.GradientTape() as tape:
        F, _, _, _ = compute_loss_down(model_down=model_down, o1=o1, ps1_mean=ps1_mean_stopped, ps1_logvar=ps1_logvar_stopped, omega=omega_stopped)
        gradients = tape.gradient(F, model_down.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model_down.trainable_variables))
