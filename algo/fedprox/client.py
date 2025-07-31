import torch
from trl import SFTTrainer

def get_fedprox_local_trainer(script_args, fed_args, model, tokenizer, training_args, local_dataset, formatting_prompts_func, data_collator, global_dict):
    
    trainer = SFTTrainerFedProx(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            max_seq_length=script_args.seq_length,
            train_dataset=local_dataset,
            formatting_func=formatting_prompts_func,
            data_collator=data_collator,
            global_state=global_dict,
            prox_mu=fed_args.prox_mu,
        )

    return trainer

class SFTTrainerFedProx(SFTTrainer):
    def __init__(self, global_state, prox_mu, **kwargs):
        super(SFTTrainerFedProx, self).__init__(**kwargs)
        self.global_state = global_state
        self.mu = prox_mu
    
    def compute_loss(self, model, inputs, return_outputs=False):

        return_values = super(SFTTrainerFedProx, self).compute_loss(model, inputs, return_outputs=return_outputs)

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
