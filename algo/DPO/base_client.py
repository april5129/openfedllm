from trl import DPOTrainer

def get_fed_local_dpo_trainer(script_args, model, model_ref, tokenizer, training_args, local_dataset):
    
    trainer = DPOTrainer(
            model=model,
            ref_model=model_ref,
            args=training_args,
            beta=script_args.dpo_beta,
            train_dataset=local_dataset,
            tokenizer=tokenizer,
        )
    return trainer
