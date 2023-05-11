import os
import time
import torch
import re
import matplotlib.pyplot as plt

def path2data(args_config):
    for r, d, f in os.walk(args_config.data_root):
        req_data = [i for i in f if i.startswith(args_config.data_name)]
    return os.path.join(r, req_data[0])

def save_net(path, state):
    tt = str(time.asctime())
    img_name_save = 'net' + " " + str(re.sub('[:!@#$]', '_', tt))
    img_name_save = img_name_save.replace(' ', '_') + '.pt'
    _dir = os.path.abspath('../')
    path = os.path.join(_dir, path)
    t = datetime.datetime.now()
    datat = t.strftime('%m/%d/%Y').replace('/', '_')
    dir = os.path.join(path, 'net' + '_' + datat)
    if not os.path.exists(dir):
        try:
            os.makedirs(dir, exist_ok=True)
            print("Directory '%s' created successfully" % ('net' + '_' + datat))
        except OSError as error:
            print("Directory '%s' can not be created" % ('net' + '_' + datat))

    net_path = os.path.join(dir, img_name_save)
    print(net_path)
    torch.save(state, net_path)
    return net_path

def plot_bar(data, category):
    counts = data[category].value_counts()
    plt.bar(counts.index, counts.values)
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.show()