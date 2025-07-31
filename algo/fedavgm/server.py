import random
import torch

def get_clients_this_round(fed_args, round):
    if (fed_args.fed_alg).startswith('local'):
        clients_this_round = [int((fed_args.fed_alg)[-1])]
    else:
        if fed_args.num_clients < fed_args.sample_clients:
            clients_this_round = list(range(fed_args.num_clients))
        else:
            random.seed(round)
            clients_this_round = sorted(random.sample(range(fed_args.num_clients), fed_args.sample_clients))
    return clients_this_round

def global_aggregate(fed_args, global_dict, local_dict_list, sample_num_list, clients_this_round, round_idx, proxy_dict=None):
    sample_this_round = sum([sample_num_list[client] for client in clients_this_round])
    global_auxiliary = None
    # Momentum-based FedAvg
    for key in global_dict.keys():
        delta_w = sum([(local_dict_list[client][key] - global_dict[key]) * sample_num_list[client] / sample_this_round for client in clients_this_round])
        proxy_dict[key] = fed_args.fedopt_beta1 * proxy_dict[key] + (1 - fed_args.fedopt_beta1) * delta_w if round_idx > 0 else delta_w
        global_dict[key] = global_dict[key] + proxy_dict[key]
    return global_dict, global_auxiliary