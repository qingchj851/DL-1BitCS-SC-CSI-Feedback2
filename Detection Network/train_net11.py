# 较终的模型，只得到检测网络
import numpy as np
import os
import scipy.io as sio
import tensorflow as tf
from keras import regularizers
from keras.models import Model
from keras.layers import Input, BatchNormalization, Dense, LeakyReLU, Activation,Conv2D,Reshape
from keras.callbacks import Callback, ReduceLROnPlateau, ModelCheckpoint
import keras.layers as KL
import time
# from data_gen2 import rou, CSI_len, US_len, walsh, SNR, M, K, data_gen,C,phi
import copy
import matplotlib.pyplot as plt
import h5py

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

rou = 0.1  #(0.05\0.10\0.15)
CSI_len = 64
US_len = 512
K = 8
C = 2 #（2，2.5，3）
M = int(C*CSI_len + CSI_len/2)

#读取walsh矩阵(.mat)
walsh_file = 'walsh/walsh_'+str(US_len) +'_'+str(M)+'.mat'
mat = sio.loadmat(walsh_file)
walsh = mat['aim_walsh'].astype('float32')


node_gain = 2
epoch = 50
iter_num = 3
batch_size = 200

testset_num = int(2e4)

regular_rate = np.array([(10**(-5))])


# 定义的重构模型
def Recon_Net(regular_rate):
    def sub_net(x, hid_nodes_1, out_dim):
        x = BatchNormalization()(x)
        x = Dense(hid_nodes_1, activation='linear', kernel_regularizer=regularizers.l2(regular_rate))(x)
        x = LeakyReLU()(x)
        y = Dense(out_dim, activation='linear')(x)
        return y

    def despread(x):
        Real = tf.matmul(x[:, :US_len], walsh)
        Imag = tf.matmul(x[:, US_len:], walsh)
        return tf.reshape(tf.stack([Real, Imag], axis=1), [-1, 2 * M])

    def spread(x):
        Real = tf.matmul(x[:, :M], np.transpose(walsh))
        Imag = tf.matmul(x[:, M:], np.transpose(walsh))
        return tf.reshape(tf.stack([Real, Imag], axis=1), [-1, 2 * US_len])

    def interfer_CSI_reduce(x, y):
        Const_CSI = (np.sqrt(rou / M)).astype('float32')
        return tf.subtract(y, tf.multiply(Const_CSI, x))

    def interfer_US_reduce(x, y):
        Const_US = (np.sqrt(1 - rou)).astype('float32')
        return tf.subtract(y, tf.multiply(Const_US, x))


    def my_tanh(x):
        y = (0.5 ** 0.5) * tf.nn.tanh(x)
        return y


    xs = Input(shape=(2 * US_len,))
    x_update = xs
    for ii in range(iter_num):
        CSI_out_name = 'CSI_'+str(ii+1)
        US_out_name = 'US_'+str(ii+1)
        x_update_name = 'x_'+str(ii+1)
        # CSI-Neti:
        CSI_bit_wave = KL.Lambda(despread)(x_update)
        CSI_bit_hat = sub_net(CSI_bit_wave, node_gain * 2 * M, 2 * M)
        CSI_bit_hat = KL.Lambda(my_tanh,name=CSI_out_name)(CSI_bit_hat)

        # CSI IR:
        US_wave = KL.Lambda(spread)(CSI_bit_hat)
        US_wave = KL.Lambda(lambda x: interfer_CSI_reduce(*x))([US_wave, xs])

        # Det-Neti：
        US_hat = sub_net(US_wave, node_gain * 2 * US_len, 2 * US_len)
        US_hat = KL.Lambda(my_tanh,name=US_out_name)(US_hat)

        # UL-US IR：
        x_update = KL.Lambda(lambda x: interfer_US_reduce(*x),name=x_update_name)([US_hat, xs])
    my_model = Model([xs], [US_hat, CSI_bit_hat])
    my_model = Model([xs], [my_model.get_layer('CSI_1').output, my_model.get_layer('US_1').output,
                            my_model.get_layer('CSI_2').output, my_model.get_layer('US_2').output,
                            my_model.get_layer('CSI_3').output, my_model.get_layer('US_3').output])
    my_model.compile(optimizer='adam', loss='mse', loss_weights=[1. / (2 * iter_num)] * (2 * iter_num))

    print(my_model.summary())
    from keras.utils.vis_utils import plot_model
    plot_model(my_model, to_file="my_model.png", show_shapes=True)

    return my_model



def my_train(path,weights_file,regular_rate,):
    my_model = Recon_Net(regular_rate)
    # 定义的loss回传函数，保存train_loss/val_loss
    class LossHistory(Callback):
        def on_train_begin(self, logs={}):
            self.losses_train = []
            self.losses_val = []

        def on_batch_end(self, batch, logs={}):
            self.losses_train.append(logs.get('loss'))

        def on_epoch_end(self, epoch, logs={}):
            self.losses_val.append(logs.get('val_loss'))

    history = LossHistory()


    # 保存最佳模型的参数
    # checkpoint = ModelCheckpoint(weights_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min', period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=2, mode='min')
    my_model.fit([trainset_y], [trainset_CSI_bit, trainset_US_bit,
                            trainset_CSI_bit, trainset_US_bit,
                            trainset_CSI_bit, trainset_US_bit],
             batch_size=batch_size,
             epochs=epoch,
             shuffle=True,
             verbose=2,  # 每个epoch输出一次loss等训练信息
             validation_data=([valset_y], [valset_CSI_bit, valset_US_bit,
                                           valset_CSI_bit, valset_US_bit,
                                           valset_CSI_bit, valset_US_bit]),
             callbacks=[history,reduce_lr])
    my_model.save_weights(weights_file)


    # 绘制loss曲线并保存
    loss_history_train = np.array(history.losses_train)
    loss_history_val = np.array(history.losses_val)
    loss_file = path + '/loss_train_val' + str(CSI_len) + '.mat'
    sio.savemat(loss_file, {'loss_train': loss_history_train, 'loss_val': loss_history_val})

    plt.figure(figsize=(14,7))
    plt.subplot(1,2,1)
    plt.plot(loss_history_train, color="blue", linewidth=2.0, linestyle="-", label='Train loss')
    plt.xlabel('Iter')
    plt.ylabel('MSE')
    plt.title('1-Bit Train loss.')
    plt.legend(loc='best')

    plt.subplot(1, 2, 2)
    plt.plot(loss_history_val, color="orange", linewidth=2.0, linestyle="-", label='Val loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('1-Bit Val loss.')
    plt.legend(loc='best')

    plt.savefig(path + '/loss.png')
    plt.close()




    ############################################
    ###################TEST#####################
    ############################################
def my_test(path,weights_file,regular_rate):
    def data_yingshe(x):
        temp = copy.copy(x)
        shape = np.shape(temp)
        temp = np.reshape(temp, [1, -1])
        for ii in range(np.size(temp)):
            if (temp[0, ii] <= 0.0):
                temp[0, ii] = 0.0
            else:
                temp[0, ii] = 1.0
        return np.reshape(temp, shape)

    def BER(x, y):
        num_x = np.size(x)
        temp = x - y
        num_temp = sum(sum(temp ** 2))
        return num_temp / num_x

    my_model = Recon_Net(regular_rate)
    my_model.load_weights(weights_file)

    SNR = np.linspace(0, 18, 10)
    # 测试BER性能
    test_US_BER = []
    test_CSI_BER = []

    for ii in range(len(SNR)):
        # 测试各SNR下的BER性能：
        # 依次测试不同BER下的MSE，并保存
        print('读取测试数据中...')
        test_file = 'data/testdata_' + str(int(SNR[ii])) + '.mat'
        mat = h5py.File(test_file)
        testset_y = np.transpose(mat['yR'][:])
        testset_US_bit = np.transpose(mat['US_bitR'][:])
        testset_CSI_bit = np.transpose(mat['CSI_bitR'][:])
        _, _, _, _, testset_CSI_bit_hat, testset_US_bit_hat = my_model.predict([testset_y])
        US_ber = BER(data_yingshe(testset_US_bit), data_yingshe(testset_US_bit_hat))
        CSI_ber = BER(data_yingshe(testset_CSI_bit), data_yingshe(testset_CSI_bit_hat))


        print('When SNR is %d dB, UL-US BER is %f' % (SNR[ii], US_ber))
        print('When SNR is %d dB, DL-CSI BER is %f' % (SNR[ii], CSI_ber))
        print('-' * 50)

        test_US_BER.append(US_ber)
        test_CSI_BER.append(CSI_ber)

    result_file = path + '/test_BER_CSI_dim' + str(M)+'_'+str(int(rou * 100))+ '.mat'
    sio.savemat(result_file,{'US_BER':test_US_BER,'CSI_BER':test_CSI_BER})


if __name__ == '__main__':
    print('读取训练数据中...')

    mat = h5py.File('data/train_data.mat')
    trainset_y = np.transpose(mat['yR'][:])
    trainset_US_bit = np.transpose(mat['US_bitR'][:])
    trainset_CSI_bit = np.transpose(mat['CSI_bitR'][:])



    print('读取校验数据中...')
    mat = h5py.File('data/val_data.mat')
    valset_y = np.transpose(mat['yR'][:])
    valset_US_bit = np.transpose(mat['US_bitR'][:])
    valset_CSI_bit = np.transpose(mat['CSI_bitR'][:])

    path = 'result/'+ time.strftime('%Y-%m-%d-%H_%M')
    if not os.path.exists(path):
        os.makedirs(path)
    weights_file = path + '/[1]bestmodel_CSI_dim' + str(CSI_len) + '.h5'
    my_train(path,weights_file,regular_rate)
    my_test(path,weights_file,regular_rate)