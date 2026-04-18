import os
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq

class AdapterTrainer:
    def __init__(self, model, tokenizer, train_dataset, eval_dataset=None, output_dir: str = "./models/adapters"):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset# Initialize eval_dataset
        self.output_dir = output_dir

        # Llama 3 specifically lacks a pad token by default. 
        # Assign it to the EOS token to prevent tensor shape errors during batching.
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

    def get_training_args(
        self, adapter_name: str, batch_size: int, grad_accum: int, epochs: int, logging_steps: int, learning_rate: float
    ) -> TrainingArguments:
        """Configures the hardware utilization and learning hyperparameters."""
        save_path = os.path.join(self.output_dir, adapter_name)
        
        return TrainingArguments(
            output_dir=save_path,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=learning_rate,
            lr_scheduler_type="cosine",
            num_train_epochs=epochs,
            logging_steps=logging_steps,
            save_strategy="steps",
            eval_strategy="steps",
            fp16=True,  
            bf16=False, 
            optim="adamw_torch",
            max_grad_norm=0.3,
            warmup_ratio=0.03,
            eval_steps=100,
            save_steps=100,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
            # group_by_length=True,
            report_to="none" 
        )

    def train(
        self, adapter_name: str, batch_size: int = 4, grad_accum: int = 4, epochs: int = 1, logging_steps: int = 10, learning_rate: float = 2e-4
    ):
        """Initializes the Trainer and executes the training loop."""
        args = self.get_training_args(adapter_name, batch_size, grad_accum, epochs, logging_steps, learning_rate)
        
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=self.data_collator,

        )
        
        print(f"Starting training for {adapter_name} adapter...")
        trainer.train()
        
        final_save_path = os.path.join(self.output_dir, adapter_name, "final")
        trainer.model.save_pretrained(final_save_path)
        self.tokenizer.save_pretrained(final_save_path)
        
        print(f"Training complete. Adapter saved to {final_save_path}")