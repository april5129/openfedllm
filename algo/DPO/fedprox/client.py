import torch
from trl import DPOTrainer

def get_fed_local_dpo_trainer(script_args, fed_args, model, model_ref, tokenizer, training_args, local_dataset, global_dict, local_auxiliary, global_auxiliary):

    trainer = DPOTrainerFedProx(
            model=model,
            ref_model=model_ref,
            args=training_args,
            beta=script_args.dpo_beta,
            train_dataset=local_dataset,
            tokenizer=tokenizer,
            global_state=global_dict,
            prox_mu=fed_args.prox_mu,
        )
    return trainer

class DPOTrainerFedProx(DPOTrainer):
    def __init__(self, global_state, prox_mu, **kwargs):
        super(DPOTrainerFedProx, self).__init__(**kwargs)
        self.global_state = global_state
        self.mu = prox_mu
    
    def compute_loss(self, model, inputs, return_outputs=False):

        return_values = super(DPOTrainerFedProx, self).compute_loss(model, inputs, return_outputs=return_outputs)

        if return_outputs:
            loss, outputs = return_values
        else:
            loss = return_values

        # Apply FedProx Loss
        for name, param in model.named_parameters():
            name = name.replace(".default", "")     # TODO: May need changes. to accord with peft
            # only trainable parameters
            if not param.requires_grad:
                continue
            else:
                loss += self.mu / 2 * torch.norm(param - self.global_state[name]) ** 2

        return (loss, outputs) if return_outputs else loss
