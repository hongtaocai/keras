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
