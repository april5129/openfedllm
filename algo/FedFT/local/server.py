
def global_aggregate(global_dict, local_dict_list, sample_num_list, clients_this_round):
    sample_this_round = sum([sample_num_list[client] for client in clients_this_round])
    global_auxiliary = None

   # Normal dataset-size-based aggregation
    for key in global_dict.keys():
        global_dict[key] = sum([local_dict_list[client][key] * sample_num_list[client] / sample_this_round for client in clients_this_round])
       
    return global_dict, global_auxiliary