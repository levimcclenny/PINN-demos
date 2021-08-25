"""
PINN solution using direct implementation in Tensorflow (Adam training only)
Authors: Ulisses Braga-Neto and Levi McClenny, Texas A&M University

Tensorflow 2.0 port of Raissi's HFM examples:
    - https://science.sciencemag.org/content/367/6481/1026.abstract
    - https://github.com/maziarraissi/HFM

This script can be re-used for your own research with explicit reference/citation to the authors
"""

import numpy as np
import tensorflow as tf
import time
import gdown
import math
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.keras import layers, activations
import scipy.io
from scipy.interpolate import griddata
import tensorflow_addons as tfa
import json

# Can be run once, once the files on on your local instance or machine these gdown commands can be removed
url = 'https://drive.google.com/uc?id=1HUO2taQcQlAj1KcHEXaPznfK-nz78HQu'
output = 'Cylinder2D_C_rectangular.json'
gdown.download(url, output, quiet=False)

url = 'https://drive.google.com/uc?id=1pq22MZIWRRreMnrb1wF8UibAxjsheuWk'
output = 'Cylinder2D_P_rectangular.json'
gdown.download(url, output, quiet=False)

url = 'https://drive.google.com/uc?id=1aVbEvX5jxL72DiaOq4HnWQyHjrfz9VZV'
output = 'Cylinder2D_U_rectangular.json'
gdown.download(url, output, quiet=False)

url = 'https://drive.google.com/uc?id=16JuaM1e8Mb7mx-T6n34vTirFMwDDcCHN'
output = 'Cylinder2D_V_rectangular.json'
gdown.download(url, output, quiet=False)

url = 'https://drive.google.com/uc?id=1pcrJgXqREztpMHFnO4W9SuoJazzzNPtE'
output = 'Cylinder2D_x_rectangular.json'
gdown.download(url, output, quiet=False)

url = 'https://drive.google.com/uc?id=1f3V2C7Ar21KS9RNOOYRhfFbYau5p5yNl'
output = 'Cylinder2D_y_rectangular.json'
gdown.download(url, output, quiet=False)


# Open data for each variable, as well as pre-configured x and y data points
file = open('Cylinder2D_C_rectangular.json','r')
C_rect = np.array(json.load(file))
file.close()
file = open('Cylinder2D_U_rectangular.json','r')
U_rect = np.array(json.load(file))
file.close()
file = open('Cylinder2D_V_rectangular.json','r')
V_rect = np.array(json.load(file))
file.close()
file = open('Cylinder2D_P_rectangular.json','r')
P_rect = np.array(json.load(file))
file.close()
file = open('Cylinder2D_x_rectangular.json','r')
x_rect = np.array(json.load(file))
file.close()
file = open('Cylinder2D_y_rectangular.json','r')
y_rect = np.array(json.load(file))
file.close()

layer_sizes = [3] + 10 * [250] + [4] # outputs will be c, u ,v, p

def neural_net(layer_sizes):
    model = Sequential()
    model.add(layers.InputLayer(input_shape=(layer_sizes[0],)))
    for width in layer_sizes[1:-1]:
        model.add(tfa.layers.WeightNormalization(layers.Dense(
            width, activation=tf.nn.silu,
            kernel_initializer="glorot_normal")))
    model.add(tfa.layers.WeightNormalization(layers.Dense(
        layer_sizes[-1], activation=None,
        kernel_initializer="glorot_normal")))
    return model


nnet = neural_net(layer_sizes)

@tf.function
def f(x,y,t):
    out  = nnet(tf.concat([x,y,t],1)) # compute u,v,p,c,d
    u = out[:, 0:1]
    v = out[:, 1:2]
    p = out[:, 2:3]
    c = out[:, 3:4]
    d = out[:, 4:5]

    u_t = tf.gradients(u,t)[0]
    u_x = tf.gradients(u,x)[0]
    u_y = tf.gradients(u,y)[0]
    L_u = tf.gradients(u_x,x)[0] + tf.gradients(u_y,y)[0]

    v_t = tf.gradients(v,t)[0]
    v_x = tf.gradients(v,x)[0]
    v_y = tf.gradients(v,y)[0]
    L_v = tf.gradients(v_x,x)[0] + tf.gradients(v_y,y)[0]

    p_x = tf.gradients(p,x)[0]
    p_y = tf.gradients(p,y)[0]

    c_t = tf.gradients(c,t)[0]
    c_x = tf.gradients(c,x)[0]
    c_y = tf.gradients(c,y)[0]
    L_c = tf.gradients(c_x,x)[0] + tf.gradients(c_y,y)[0]

    d_t = tf.gradients(d,t)[0]
    d_x = tf.gradients(d,x)[0]
    d_y = tf.gradients(d,y)[0]
    L_d = tf.gradients(d_x,x)[0] + tf.gradients(d_y,y)[0]

    f_1 = u_t + u*u_x + v*u_y + p_x - 0.01*L_u
    f_2 = v_t + u*v_x + v*v_y + p_y - 0.01*L_v
    f_3 = u_x + v_y
    f_4 = c_t + u*c_x + v*c_y - 0.01*L_c
    f_5 = d_t + u*d_x + v*d_y - 0.01*L_d

    return f_1, f_2, f_3, f_4, f_5

@tf.function
def grad(model,xcl,ycl,tcl,xb,yb,tb,ub,vb,xtr,ytr,ttr,ctr):
    with tf.GradientTape(persistent=True) as tape:
        loss_value, mse_b, mse_trc, mse_trd, mse_f = loss(xcl,ycl,tcl,xb,yb,tb,ub,vb,xtr,ytr,ttr,ctr)
        grads = tape.gradient(loss_value,model.trainable_variables)
    return loss_value, mse_b, mse_trc, mse_trd, mse_f, grads


def loss(xcl, ycl, tcl, xb, yb, tb, ub, vb, xtr, ytr, ttr, ctr):
    b_pred = nnet(tf.concat([xb, yb, tb], 1))
    tr_pred = nnet(tf.concat([xtr, ytr, ttr], 1))
    f_1, f_2, f_3, f_4, f_5 = f(xcl, ycl, tcl)

    # Dirichlet boundary loss
    mse_b = tf.reduce_mean(tf.pow(ub - b_pred[:, 0:1], 2) + tf.pow(vb - b_pred[:, 1:2], 2))

    # Training data loss on c and d=1-c
    mse_trc = tf.reduce_mean(tf.pow(ctr - tr_pred[:, 3:4], 2))
    mse_trd = tf.reduce_mean(tf.pow(1 - ctr - tr_pred[:, 4:5], 2))

    # Residual loss
    mse_f = tf.reduce_mean(tf.pow(f_1, 2) + tf.pow(f_2, 2) + tf.pow(f_3, 2) + tf.pow(f_4, 2) + tf.pow(f_5, 2))

    return mse_b + mse_trc + mse_trd + mse_f, mse_b, mse_trc, mse_trd, mse_f

def fit(xcl,ycl,tcl,xb,yb,tb,ub,vb,xtr,ytr,ttr,ctr,tf_iter,lr):

    tf_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    N_data = ttr.shape[0]
    N_eqns = tcl.shape[0]

    batch_size = 10000
    n_batches =  len(xcl) // batch_size

    # For random sampling of mini-batches
    idx_data = np.random.choice(N_data, min(batch_size, N_data))
    idx_eqns = np.random.choice(N_eqns, batch_size)

    print("starting Adam training")
    start_time = time.time()
    for epoch in range(tf_iter):
      for i in range(n_batches):
        xcl_batch = xcl[i*batch_size:(i*batch_size + batch_size),]
        ycl_batch = ycl[i*batch_size:(i*batch_size + batch_size),]
        tcl_batch = tcl[i*batch_size:(i*batch_size + batch_size),]
        xtr_batch = xtr[i*batch_size:(i*batch_size + batch_size),]
        ytr_batch = ytr[i*batch_size:(i*batch_size + batch_size),]
        ttr_batch = ttr[i*batch_size:(i*batch_size + batch_size),]
        ctr_batch = ctr[i*batch_size:(i*batch_size + batch_size),]
        # For random sampling of mini-batches
        # xcl_batch = xcl[idx_eqns,:]
        # ycl_batch = ycl[idx_eqns,:]
        # tcl_batch = tcl[idx_eqns,:]
        # xtr_batch = xtr[idx_data,:]
        # ytr_batch = ytr[idx_data,:]
        # ttr_batch = ttr[idx_data,:]
        # ctr_batch = ctr[idx_data,:]

        loss_value,mse_b,mse_trc,mse_trd,mse_f,grads = grad(nnet,
            xcl_batch,ycl_batch,tcl_batch,xb,yb,tb,ub,vb,xtr_batch,ytr_batch,ttr_batch,ctr_batch)
        tf_optimizer.apply_gradients(zip(grads,nnet.trainable_variables))
        if (i % 20 == 0):
          elapsed = time.time()-start_time
          print('ep. {:-3}, it. {:-3}: time = {:2.9} mse_b = {:2.9} mse_trc = {:2.9} mse_f = {:2.9}'.format(epoch,i,elapsed,mse_b,mse_trc,mse_f))
          start_time = time.time()

NS = C_rect.shape[0] # number of spatial points
NT = C_rect.shape[1] # number of time points

t_rect = np.linspace(0,16,NT)

# inlet boundary condition points
# sample Nb points from len(idb)*NT = 20301 total points
Nb = 20000
idb = np.where(x_rect==x_rect.min())[0]
idbs = np.random.choice(len(idb)*NT,Nb,replace=False)
xb = tf.expand_dims(tf.convert_to_tensor(np.tile(x_rect[idb],NT)[idbs],dtype=tf.float32),axis=1)
yb = tf.expand_dims(tf.convert_to_tensor(np.tile(y_rect[idb],NT)[idbs],dtype=tf.float32),axis=1)
tb = tf.expand_dims(tf.convert_to_tensor(np.tile(t_rect,(len(idb),1)).T.flatten()[idbs],dtype=tf.float32),axis=1)
ub = tf.expand_dims(tf.convert_to_tensor(U_rect[idb,:].T.flatten()[idbs],dtype=tf.float32),axis=1)
vb = tf.expand_dims(tf.convert_to_tensor(V_rect[idb,:].T.flatten()[idbs],dtype=tf.float32),axis=1)

# collocation points
# must remove points in the cylinder region (x^2+y^2<=0.5^2)
# this removes about 1.57% of the points
Ncl = 100000
idcl = np.random.choice(NS*NT,Ncl,replace=False)
xcl = tf.expand_dims(tf.convert_to_tensor(np.tile(x_rect,NT)[idcl],dtype=tf.float32),axis=1)
ycl = tf.expand_dims(tf.convert_to_tensor(np.tile(y_rect,NT)[idcl],dtype=tf.float32),axis=1)
tcl = tf.expand_dims(tf.convert_to_tensor(np.tile(t_rect,(NS,1)).T.flatten()[idcl],dtype=tf.float32),axis=1)

# concentration training data points
# sample Ntr points from NS*NT = 4017588
Ntr = 100000
idtr = np.random.choice(NS*NT,Ntr,replace=False)
xtr = tf.expand_dims(tf.convert_to_tensor(np.tile(x_rect,NT)[idtr],dtype=tf.float32),axis=1)
ytr = tf.expand_dims(tf.convert_to_tensor(np.tile(y_rect,NT)[idtr],dtype=tf.float32),axis=1)
ttr = tf.expand_dims(tf.convert_to_tensor(np.tile(t_rect,(NS,1)).T.flatten()[idtr],dtype=tf.float32),axis=1)
ctr = tf.expand_dims(tf.convert_to_tensor(C_rect.T.flatten()[idtr],dtype=tf.float32),axis=1)

# concentration testing data points
# points not selected for training
idts =  np.setdiff1d(np.arange(NS*NT),idtr,assume_unique=True)
xts = tf.expand_dims(tf.convert_to_tensor(np.tile(x_rect,NT)[idts],dtype=tf.float32),axis=1)
yts = tf.expand_dims(tf.convert_to_tensor(np.tile(y_rect,NT)[idts],dtype=tf.float32),axis=1)
tts = tf.expand_dims(tf.convert_to_tensor(np.tile(t_rect,(NS,1)).T.flatten()[idts],dtype=tf.float32),axis=1)
cts = tf.expand_dims(tf.convert_to_tensor(C_rect.T.flatten()[idts],dtype=tf.float32),axis=1)

Ri = 0.01 # Reynolds number = 100, viscosity = 0.01
Peci = 0.01 # Peclet number = 100

normalizer = layers.experimental.preprocessing.Normalization()
normalizer.adapt([xtr, ytr, ttr])
nnet = neural_net(layer_sizes)

fit(xcl,ycl,tcl,xb,yb,tb,ub,vb,xtr,ytr,ttr,ctr,tf_iter=10000,lr=0.001)

# evaluate the trained PINN

def mydot(p,q):
  dot = 0
  for i in range(NS*NT):
    dot += p[i]*q[i]
  return dot



def L2error(p,b):
    """calculates ||prediction-baseline||/||baseline||
    for tensors prediction (p) and baseline (b)"""
    pmb = p-b
    num = mydot(pmb,pmb)
    denom = mydot(b,b)
    return np.sqrt(num/denom)

def relative_error(pred, exact):
    if type(pred) is np.ndarray:
        return np.sqrt(np.mean(np.square(pred - exact))/np.mean(np.square(exact - np.mean(exact))))
    return tf.sqrt(tf.reduce_mean(tf.square(pred - exact))/tf.reduce_mean(tf.square(exact - tf.reduce_mean(exact))))
i = 1
# batch prediction due to large size
x = tf.expand_dims(tf.convert_to_tensor(x_rect,dtype=tf.float32),1)
y = tf.expand_dims(tf.convert_to_tensor(y_rect,dtype=tf.float32),1)
for i in range(NT):
  t = tf.convert_to_tensor(t_rect[i]*np.ones([NS,1]),dtype=tf.float32)
  out  = nnet(tf.concat([x,y,t],1)) # u,v,p,c prediction for at time t
  if i==0:
    U_pred = tf.expand_dims(out[:,0],1)
    V_pred = tf.expand_dims(out[:,1],1)
    P_pred = tf.expand_dims(out[:,2],1)
    C_pred = tf.expand_dims(out[:,3],1)
  else:
    U_pred = tf.concat([U_pred,tf.expand_dims(out[:,0],1)],1)
    V_pred = tf.concat([V_pred,tf.expand_dims(out[:,1],1)],1)
    P_pred = tf.concat([P_pred,tf.expand_dims(out[:,2],1)],1)
    C_pred = tf.concat([C_pred,tf.expand_dims(out[:,3],1)],1)

# L2 error on U,V,P,C on entire data
print('L2 Error for U = %e' % L2error(U_pred.numpy().T.flatten(),U_rect.T.flatten()))
print('L2 Error for V = %e' % L2error(V_pred.numpy().T.flatten(),V_rect.T.flatten()))
print('L2 Error for P = %e' % L2error(P_pred.numpy().T.flatten(),P_rect.T.flatten()))
print('L2 Error for C = %e' % L2error(C_pred.numpy().T.flatten(),C_rect.T.flatten()))

# L2 testing error using concentration testing
# points not selected for training
idts =  np.setdiff1d(np.arange(NS*NT),idtr,assume_unique=True)
xts = tf.expand_dims(tf.convert_to_tensor(np.tile(x_rect,NT)[idts],dtype=tf.float32),axis=1)
yts = tf.expand_dims(tf.convert_to_tensor(np.tile(y_rect,NT)[idts],dtype=tf.float32),axis=1)
tts = tf.expand_dims(tf.convert_to_tensor(np.tile(t_rect,(NS,1)).T.flatten()[idts],dtype=tf.float32),axis=1)
cts = tf.expand_dims(tf.convert_to_tensor(C_rect.T.flatten()[idts],dtype=tf.float32),axis=1)
out  = nnet(tf.concat([xts,yts,tts],1)) # u,v,p,c prediction on test data


N = C_rect.shape[0]
T = C_rect.shape[1]

t = np.expand_dims(np.linspace(0, 16, T), axis=1)

step = 0.05  # grid quantization

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.animation as animation

tep = 0.05  # grid quantization


def get_imgseq(D):
    """
    - Normalize between 0 and 1
    - Insert val in gaps corresponding to jumps in y (this is the space occupied by the cylinder)
    - Reshape to 201x101x101
    """
    mx = np.max(D)
    mn = np.min(D)
    Dout = (D - mn) / (mx - mn)
    yi = y_rect
    j = 0
    for i in range(NS - 1):
        gap = yi[i + 1] - yi[i]
        if gap > step + 0.01:
            for k in range(1, np.around(gap / step).astype(int)):
                j += 1
                Dout = np.insert(Dout, j, 0, axis=0)
        j += 1
    return Dout.reshape([201, 101, 201])


imgseq_U_pred = get_imgseq(U_pred.numpy())
imgseq_V_pred = get_imgseq(V_pred.numpy())
imgseq_P_pred = get_imgseq(P_pred.numpy())
imgseq_C_pred = get_imgseq(C_pred.numpy())

imgseq_U = get_imgseq(U_rect)
imgseq_V = get_imgseq(V_rect)
imgseq_P = get_imgseq(P_rect)
imgseq_C = get_imgseq(C_rect)

fig, ax = plt.subplots(figsize=(8, 4), nrows=2, ncols=2, dpi=100)
ax[0, 0].imshow(imgseq_U[:, :, 100].T, cmap='seismic')
ax[0, 1].imshow(imgseq_V[:, :, 100].T, cmap='seismic')
ax[1, 0].imshow(imgseq_P[:, :, 100].T, cmap='seismic')
ax[1, 1].imshow(imgseq_C[:, :, 100].T, cmap='seismic')
for i in range(2):
    for j in range(2):
        ax[i, j].add_patch(Circle((50, 50), 11, facecolor='gray'))
plt.show()

fig, ax = plt.subplots(figsize=(8, 4), nrows=2, ncols=2, dpi=100)
ax[0, 0].imshow(imgseq_U_pred[:, :, 100].T, cmap='seismic')
ax[0, 1].imshow(imgseq_V_pred[:, :, 100].T, cmap='seismic')
ax[1, 0].imshow(imgseq_P_pred[:, :, 100].T, cmap='seismic')
ax[1, 1].imshow(imgseq_C_pred[:, :, 100].T, cmap='seismic')
for i in range(2):
    for j in range(2):
        ax[i, j].add_patch(Circle((50, 50), 11, facecolor='gray'))
plt.show()


fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
img = imgseq_V_pred

ims = []
for i in range(NT):
    ims.append([ax.imshow(img[:, :, i].T, cmap='seismic', animated=True),
                ax.add_patch(Circle((50, 50), 11, facecolor='gray'))])

# Call the animator:
anim = animation.ArtistAnimation(fig, ims, interval=25, blit=True)

# Save the animation as an mp4. This requires ffmpeg to be installed.
anim.save('Cylinder2D_V_Demo.mp4', fps=15)

# Save network weights to a folder called "model"
nnet.save("model")