import random
import torch

def get_clients_this_round(fed_args, round):
    if fed_args.num_clients < fed_args.sample_clients:
        clients_this_round = list(range(fed_args.num_clients))
    else:
        random.seed(round)
        clients_this_round = sorted(random.sample(range(fed_args.num_clients), fed_args.sample_clients))
    return clients_this_round
    
def global_aggregate(fed_args, global_dict, local_dict_list, clients_this_round, proxy_dict=None, opt_proxy_dict=None):
    global_auxiliary = None

    for key, param in opt_proxy_dict.items():
        delta_w = sum([(local_dict_list[client][key] - global_dict[key]) for client in clients_this_round]) / len(clients_this_round)
        # In paper 'adaptive federated optimization', momentum is not used
        proxy_dict[key] = delta_w
        opt_proxy_dict[key] = param + torch.square(proxy_dict[key])
        global_dict[key] += fed_args.fedopt_eta * torch.div(proxy_dict[key], torch.sqrt(opt_proxy_dict[key])+fed_args.fedopt_tau)

    return global_dict, global_auxiliary