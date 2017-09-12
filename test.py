# keras
test keras


def get_window_data(price, window_len):
    price = price.T
    n = price.shape[1]
    print n
    ips = []
    for i in range(30):
        p = price[i, :].reshape((n,))
        ps = []
        
        for i in range(window_len):
            if i == (window_len-1):
                ps.append(p[i:])
            else:
                ps.append(p[i:-((window_len-1)-i)])
        ip = np.vstack(ps)
        ip = ip.T
        ip = ip.reshape([1, ip.shape[0], ip.shape[1]])
        ips.append(ip)
    ips = np.concatenate(ips, axis=0)
    return ips
    
def window_data(data):
    price = data['xs'][20]
    vol = data['vs'][20]
    wp = get_window_data(price, 15)
    wv = get_window_data(vol, 15)
    wy = (data['ys'][20].T)[:, (15-1):]
    return wp, wv, wy

def window_data_instr(data, instr):
    wp, wv, wy = window_data(data)
    return np.squeeze(wp[instr, :, :]), np.squeeze(wv[instr, :, :]), np.squeeze(wy[instr, :])
    
def remove_zero_and_get_delta(p, v, y):
    good_price = np.all(p > 0, axis=1)
    good_v = np.all(v > 0, axis=1)
    valid_row = (good_price & good_v)

    p_ = p[valid_row, :]
    v_ = v[valid_row, :]
    y_ = y[valid_row]

    delta_p = p_[:, 1:] - p_[:, :-1] 
    delta_v = v_[:, 1:] - v_[:, :-1] 

    good_v = np.all(delta_v >= 0., axis=1)

    delta_p_ = delta_p[good_v, :]
    delta_v_ = delta_v[good_v, :]

    my_x = np.concatenate([delta_p_, delta_v_], axis=1)
    my_y = y_[good_v]

    print my_x.shape
    print my_y.shape
    
    return my_x, my_y
    
p, v, y = window_data_instr(data_is, 26)
my_x, my_y = remove_zero_and_get_delta(p, v, y)


from keras.models import Sequential
from keras.layers import Merge, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam


my_batch_size = 12800
my_nb_epoch = 300
my_layer_len_price = 14
my_layer_len_vol = 14
my_layer_total = 14
my_lr = 0.01


encoder_a = Sequential()
encoder_a.add(Dense(my_layer_len_price, activation='relu', input_shape=(14,), init="glorot_normal") )
encoder_a.add(BatchNormalization())

encoder_b = Sequential()
encoder_b.add(Dense(my_layer_len_vol, activation='relu', input_shape=(14,), init="glorot_normal") )
encoder_b.add(BatchNormalization())

m = Sequential()
m.add(Merge([encoder_a, encoder_b], mode='concat'))
m.add(Dense(my_layer_total, activation='relu', init="glorot_normal"))
m.add(BatchNormalization())
m.add(Dropout(0.2))

m.add(Dense(my_layer_total, activation='relu', init="glorot_normal"))
m.add(BatchNormalization())
m.add(Dropout(0.2))

m.add(Dense(1, init="glorot_normal"))

import keras.backend as K

def one_minus_r_squared(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return (SS_res/(SS_tot + K.epsilon()) )

def r_square(y_true, y_pred):
    return 1 - one_minus_r_squared(y_true, y_pred)

my_adam = Adam(lr = my_lr)

m.compile(optimizer=my_adam, loss=one_minus_r_squared, metrics=[r_square])

# lr=0.001

m.fit([my_x_norm[:, :14], my_x_norm[:, -14:]], my_y, batch_size=my_batch_size, nb_epoch=my_nb_epoch,
     validation_data=([my_x_os1_norm[:, :14], my_x_os1_norm[:, -14:]], my_y_os1))
