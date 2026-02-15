import os
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq

class AdapterTrainer:
    def __init__(self, model, tokenizer, train_dataset, output_dir: str = "./artifacts/adapters"):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.output_dir = output_dir

        # Llama 3 specifically lacks a pad token by default. 
        # We assign it to the EOS token to prevent tensor shape errors during batching.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # DataCollatorForSeq2Seq handles dynamic padding of batches.
        # It pads 'input_ids' with pad_token_id and 'labels' with -100.
        self.data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
            return_tensors="pt"
        )

    def get_training_args(self, adapter_name: str) -> TrainingArguments:
        """Configures the hardware utilization and learning hyperparameters."""
        save_path = os.path.join(self.output_dir, adapter_name)
        
        return TrainingArguments(
            output_dir=save_path,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8, # Simulates a batch size of 16 (2 * 8)
            learning_rate=2e-4,
            lr_scheduler_type="cosine",
            num_train_epochs=2,
            logging_steps=10,
            save_strategy="epoch",
            bf16=True, # Critical: Matches the bfloat16 precision of the base model
            optim="adamw_torch",
            max_grad_norm=0.3,
            warmup_ratio=0.03,
            group_by_length=True, # Groups sequences of similar length to minimize padding overhead
            report_to="none" # Disables wandb/tensorboard logging for this local test
        )

    def train(self, adapter_name: str):
        """Initializes the Trainer and executes the training loop."""
        args = self.get_training_args(adapter_name)
        
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=self.train_dataset,
            data_collator=self.data_collator,
        )
        
        print(f"Starting training for {adapter_name} adapter...")
        trainer.train()
        
        # Save only the LoRA adapter weights, not the full 8B model
        final_save_path = os.path.join(self.output_dir, adapter_name, "final")
        trainer.model.save_pretrained(final_save_path)
        self.tokenizer.save_pretrained(final_save_path)
        
        print(f"Training complete. Adapter saved to {final_save_path}")