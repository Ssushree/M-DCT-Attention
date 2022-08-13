from tensorflow.keras import backend as K
import cv2
import tensorflow as tf
import numpy.random as random
import numpy as np
from numpy import r_
from tensorflow.keras import layers
from tensorflow.keras.layers import *


def n_mode_product(x, u, n):
    n = int(n)
    if n > 26:
        raise ValueError('n is too large.')
    ind = ''.join(chr(ord('a') + i) for i in range(n))

    return tf.einsum(f'L{ind}K...,JK->L{ind}J...', x, u)

t_dct = np.array([[1, 1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, -1, -1, -1, -1],
                  [1, 1, -1, -1, -1, -1, 1, 1],
                  [1, -1, -1, -1, 1, 1, 1, -1],
                  [1, -1, -1, 1, 1, -1, -1, 1],
                  [1, -1, 1, 1, -1, -1, 1, -1],
                  [1, -1, 1, -1, -1, 1, -1, 1],
                  [1, -1, 1, -1, 1, -1, 1, -1]],  dtype = 'float')
s_dct = np.diag([1/np.sqrt(8), 1/np.sqrt(8), 1/np.sqrt(8), 1/np.sqrt(8), 1/np.sqrt(8), 1/np.sqrt(8), 1/np.sqrt(8), 1/np.sqrt(8)])

c_dct = np.matmul(s_dct, t_dct)
if (np.matmul(np.transpose(c_dct), c_dct)==np.identity(8)).any():
    c_dct_inv = np.matmul(np.transpose(t_dct), s_dct)
else:
    c_dct_inv = np.matmul(np.linalg.inv(t_dct), np.linalg.inv(s_dct))

def cal_tensor_dct_coeff(inp):
    bs, w, h, c = inp.shape
    
    dct_coeff = K.ones_like(K.variable(np.random.random((bs, w, h, c))))
    dct_coeff = tf.Variable(dct_coeff)
    block_size = 8
    #for b in range(bs):
    for i in r_[:w:block_size]:
        for j in r_[:h:block_size]:
            for k in r_[:c:block_size]:
                coe = Lambda(lambda x: n_mode_product(n_mode_product(n_mode_product(n_mode_product(n_mode_product(n_mode_product(x[0], x[1], 0), x[1], 1), x[1], 2), x[2], 0), x[2], 1), x[2], 2))([inp[:, i:(i+block_size), j:(j+block_size), k:(k+block_size)], t_dct, s_dct])
                dct_coeff[:, i:(i+block_size), j:(j+block_size), k:(k+block_size)].assign(coe)
    dct_coeff = tf.convert_to_tensor(dct_coeff)
    return dct_coeff

    
def cal_thresholded_coeff(dct_coeff, energy_retained):
    bs, w, h, c = dct_coeff.shape
    
    dct_coeff_thresh_selected = K.zeros_like(K.variable(np.random.random((bs, w, h, c))))
    dct_coeff_thresh_selected = tf.Variable(dct_coeff_thresh_selected)
    percent_energy_thresholded = K.zeros_like(K.variable(np.random.random((bs))))
    percent_energy_thresholded = tf.Variable(percent_energy_thresholded)
    
    thresh_range = [0.8, 0.6, 0.4, 0.2, 0.1, 0.08, 0.06, 0.04, 0.02, 0.01, 0.008, 0.006, 0.004, 0.002, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]
    
    for b in range(bs):
        for th in range(len(thresh_range)):
            thresh = thresh_range[th]
            coeff_thresh = tf.math.multiply(dct_coeff[b], (tf.cast(tf.math.abs(dct_coeff[b]) > tf.math.multiply(thresh, tf.math.reduce_max(dct_coeff[b], axis = [1, 2], keepdims = True)), tf.float32)))
            percent_energy = (tf.reduce_sum(tf.math.square(coeff_thresh))/ tf.reduce_sum(tf.math.square(dct_coeff[b])))*100
            percent_energy_thresholded[b].assign(percent_energy)
            if percent_energy_thresholded[b] >= energy_retained:
                dct_coeff_thresh_selected[b].assign(coeff_thresh)
                break
    dct_coeff_thresh_selected = tf.convert_to_tensor(dct_coeff_thresh_selected)
    percent_energy_thresholded = tf.convert_to_tensor(percent_energy_thresholded)
    return dct_coeff_thresh_selected
        
def cal_reconstructed_feat(dct_coeff_thresh_selected):
    bs, w, h, c = dct_coeff_thresh_selected.shape    
    recon_feat_vis_threshold = K.zeros_like(K.variable(np.random.random((bs, w, h, c))))
    recon_feat_vis_threshold = tf.Variable(recon_feat_vis_threshold)
    block_size = 8
    for i in r_[:w:block_size]:
        for j in r_[:h:block_size]:
            for k in r_[:c:block_size]:
                recon = Lambda(lambda x: n_mode_product(n_mode_product(n_mode_product(x[0], x[1], 0), x[1], 1), x[1], 2))([dct_coeff_thresh_selected[:, i:(i+block_size), j:(j+block_size), k:(k+block_size)], c_dct_inv])
                recon_feat_vis_threshold[:, i:(i+block_size), j:(j+block_size), k:(k+block_size)].assign(recon)
    recon_feat_vis_threshold = tf.convert_to_tensor(recon_feat_vis_threshold)
    return recon_feat_vis_threshold 

def tensor_dct(inp, thresh):
    dct_coeff = cal_tensor_dct_coeff(inp)
    dct_coeff_thresh_selected = cal_thresholded_coeff(dct_coeff, thresh)
    recon_feat_vis_threshold = cal_reconstructed_feat(dct_coeff_thresh_selected)
    return recon_feat_vis_threshold
    
def att_dct(inp, thresh):
    recon_feat_vis_threshold = tensor_dct(inp, thresh)
    error_score = Lambda(lambda x: 1 - K.softmax(tf.subtract(x[0], x[1])))([inp, recon_feat_vis_threshold])
    att_output = Lambda(lambda x: tf.math.multiply(x[0], tf.math.multiply(x[0], x[1])))([inp, error_score])
    return att_output
