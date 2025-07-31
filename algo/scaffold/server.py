import random

def get_clients_this_round(fed_args, round):
    if fed_args.num_clients < fed_args.sample_clients:
        clients_this_round = list(range(fed_args.num_clients))
    else:
        random.seed(round)
        clients_this_round = sorted(random.sample(range(fed_args.num_clients), fed_args.sample_clients))
        
    return clients_this_round

def global_aggregate(fed_args, global_dict, local_dict_list, sample_num_list, clients_this_round, auxiliary_info=None):
    sample_this_round = sum([sample_num_list[client] for client in clients_this_round])
    global_auxiliary = None

    for key in global_dict.keys():
        global_dict[key] = sum([local_dict_list[client][key] * sample_num_list[client] / sample_this_round for client in clients_this_round])
        global_auxiliary, auxiliary_delta_dict = auxiliary_info
        for key in global_auxiliary.keys():
            delta_auxiliary = sum([auxiliary_delta_dict[client][key] for client in clients_this_round]) 
            global_auxiliary[key] += delta_auxiliary / fed_args.num_clients

    return global_dict, global_auxiliary