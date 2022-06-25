from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import yaml
from lib import utils
from lib.utils import load_graph_data
from model.pytorch.dcrnn_supervisor import DCRNNSupervisor
import random
import numpy as np
import os


def main(args):
    with open(args.config_filename) as f:
        supervisor_config = yaml.load(f)

        graph_pkl_filename = supervisor_config['data'].get('graph_pkl_filename')
        sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data(graph_pkl_filename)
        
        for i in range(3):
            np.random.seed(i)
            random.seed(i)
            max_itr = 12 #12
            data, search_data_x, search_data_y = utils.load_dataset(**supervisor_config.get('data'))
            # itr_supervisor_config = supervisor_config.copy()
            # itr_supervisor_config['data_itr'] = data
            supervisor = DCRNNSupervisor(random_seed=i, iteration=0, max_itr = max_itr, 
                    adj_mx=adj_mx, **supervisor_config)

            for itr in range(max_itr):
                supervisor.iteration = itr
                supervisor._data = data
                supervisor.train()

                sort_ind = np.arange(len(search_data_x))
                np.random.shuffle(sort_ind)
                selected_data_x = [search_data_x[i] for i in sort_ind[-8:]]
                selected_data_y = [search_data_y[i] for i in sort_ind[-8:]]

                selected_data = {}
                selected_data['x'] = selected_data_x
                selected_data['y'] = selected_data_y
                search_config = supervisor_config.get('data').copy()
                search_config['selected_data'] = selected_data
                search_config['previous_data'] = data

                data = utils.generate_new_trainset(**search_config)

                search_data_x = [search_data_x[i] for i in sort_ind[:-8]]
                search_data_y = [search_data_y[i] for i in sort_ind[:-8]]
                print('remained scenarios:', len(search_data_x))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default='data/model/dcrnn_cov.yaml', type=str,
                        help='Configuration filename for restoring the model.')
    parser.add_argument('--use_cpu_only', default=False, type=bool, help='Set to true to only use cpu.')
    args = parser.parse_args()
    main(args)

