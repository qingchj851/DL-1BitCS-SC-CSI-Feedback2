import scipy.io as sio
import os
import numpy as np
import tensorflow as tf
print()
from keras import optimizers
from keras.layers import Input, Dense, BatchNormalization, Reshape, ReLU,regularizers,MaxPooling2D,LeakyReLU,Lambda
from keras.activations import tanh
from keras.models import Model
from keras.callbacks import TensorBoard, Callback,ModelCheckpoint,ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
import time
from matplotlib import pyplot as plt
from keras.utils.vis_utils import plot_model
tf.reset_default_graph()  #清除默认图形堆栈并重置全局默认图形


US_len = 512
CSI_len = 64
K = 8
rou = 0.10
C = 2.5
M = int( C * CSI_len + CSI_len / 2)


regular_rate = 10**(-5)
epochs = 50
batch_size = 500

node_gain = 2
layer_num = 1
path = 'result/' + time.strftime('%Y-%m-%d-%H_%M') + '/'+ '_'+str(M) + '_'+ str(int(rou * 100))
if not os.path.exists(path):
    os.makedirs(path)

SNR = np.linspace(0, 18, 10)
NMSE = []

mat = sio.loadmat('data/8trainsetfix.mat')
# 输入训练集与验证集
# Data loading
trainset_CSI_BIHT = mat['CSI_BIHT5set'].astype('float32')
trainset_CSI_support = mat['Address_s'].astype('float32')
trainset_CSI_label = mat['CSI_s'].astype('float32')



mat1 = sio.loadmat('data/8valsetfix.mat')
valset_CSI_BIHT = mat1['CSI_BIHT5_set'].astype('float32')
valset_CSI_support = mat1['Address_set'].astype('float32')
valset_CSI_label = mat1['CSI_set'].astype('float32')




# Bulid the autoencoder model of CsiNet
def refine_network(x, support, regular_rate,node_gain,layer_num):
    def sub_net(x, support, hid_nodes_1, out_dim):
        def multip_support(support, x):
            real = tf.multiply(support, x[:, :CSI_len])
            imag = tf.multiply(support, x[:, CSI_len:])
            x_c = tf.concat([real, imag], axis=1)

            return x_c
        for i in range(layer_num):
            x = BatchNormalization()(x)
            x = Lambda(lambda x: multip_support(*x))([support, x])
            x = Dense(hid_nodes_1, activation='linear', kernel_regularizer=regularizers.l2(regular_rate))(x)
            x = LeakyReLU()(x)

        y = Dense(out_dim, activation='linear')(x)

        return y

    # 整个流程开始

    CSI_bit_hat = sub_net(x, support, node_gain * 2 * CSI_len, 2*CSI_len)
    return CSI_bit_hat



image_tensor1 = Input(shape=(CSI_len*2,))
image_tensor2 = Input(shape=(CSI_len,))
network_output = refine_network(image_tensor1, image_tensor2, regular_rate, node_gain, layer_num)
autoencoder = Model(inputs=[image_tensor1,image_tensor2], outputs=network_output)
adam = optimizers.Adam(lr=0.001)
autoencoder.compile(optimizer=adam, loss='mse')
print(autoencoder.summary())
plot_model(autoencoder, to_file='model.png', show_shapes=True)


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses_train = []
        self.losses_val = []

    def on_batch_end(self, batch, logs={}):
        self.losses_train.append(logs.get('loss'))

    def on_epoch_end(self, epoch, logs={}):
        self.losses_val.append(logs.get('val_loss'))


history = LossHistory()
filepath = "weights.best.hdf5"
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,patience=5, mode='min')
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

autoencoder.fit([trainset_CSI_BIHT, trainset_CSI_support], trainset_CSI_label,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                verbose=2,
                validation_data=([valset_CSI_BIHT, valset_CSI_support],valset_CSI_label),
                callbacks=[history,checkpoint,reduce_lr])

# 绘制loss曲线并保存

loss_history_train = np.array(history.losses_train)
loss_history_val = np.array(history.losses_val)
loss_file = path + '/loss_train_val.mat'
sio.savemat(loss_file, {'loss_train': loss_history_train, 'loss_val': loss_history_val})

plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
plt.plot(loss_history_train, color="blue", linewidth=2.0, linestyle="-", label='Train loss')
plt.xlabel('Iter')
plt.ylabel('MSE')
plt.title('Train loss.')
plt.legend(loc='best')

plt.subplot(1, 2, 2)
plt.plot(loss_history_val, color="orange", linewidth=2.0, linestyle="-", label='Val loss')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('Val loss.')
plt.legend(loc='best')

plt.savefig(path + '/loss.png')
plt.close()

for ii in range(len(SNR)):
    # Testing data

    mat2 = sio.loadmat('data/testset['+str(ii)+']_' + str(M) + '_' + str(int(rou*100)) + '_8.mat')
    # 输入训练集与验证集
    # Data loading
    testset_CSI_BIHT = mat2['CSI_BIHT5_set'].astype('float32')
    testset_CSI_support = mat2['Address_set'].astype('float32')
    testset_CSI_label = mat2['CSI_set'].astype('float32')

    # Testing data
    tStart = time.time()
    dl_csi = autoencoder.predict([testset_CSI_BIHT, testset_CSI_support])


    tEnd = time.time()
    print("It cost %f sec" % ((tEnd - tStart) / testset_CSI_BIHT.shape[0]))



    # Calcaulating the NMSE and rho
    def calc_nmse(ce_h, h_source):
        h_source_norm = h_source / (np.linalg.norm(h_source, ord=2, axis=0))
        ce_h_norm = ce_h / (np.linalg.norm(ce_h, ord=2, axis=0))

        temp = np.linalg.norm((ce_h_norm - h_source_norm), ord=2, axis=0) ** 2
        NMSE = np.mean(temp / (np.linalg.norm(h_source, ord=2, axis=0) ** 2))


        return NMSE

    nmse = []
    for i in range(len(dl_csi)):
        pre_H_dl = dl_csi[i,:]
        H = testset_CSI_label[i,:]
        nmse_t = calc_nmse(pre_H_dl , H)
        nmse.append(nmse_t)

    nmse_mean =np. mean(nmse)


    print("NMSE is ", nmse_mean)
    NMSE.append(nmse_mean)


NMSE = np.array(NMSE)
AMNMSE_file = path + '/8NMSE'+ str(M)+'_'+ str(int(rou * 100))+'.mat'
sio.savemat(AMNMSE_file, {'NMSE': NMSE})