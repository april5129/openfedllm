from trl import SFTTrainer

def get_base_local_trainer(script_args, model, tokenizer, training_args, local_dataset, formatting_prompts_func, data_collator):
    trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            max_seq_length=script_args.seq_length,
            train_dataset=local_dataset,
            formatting_func=formatting_prompts_func,
            data_collator=data_collator,
        )
    return trainer