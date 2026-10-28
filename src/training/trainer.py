import os
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq


class AdapterTrainer:
    def __init__(
        self,
        model,
        tokenizer,
        train_dataset,
        eval_dataset=None,
        output_dir: str = "./models/adapters",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.output_dir = output_dir

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
            return_tensors="pt",
            label_pad_token_id=-100, # Explicitly pad custom labels with ignore_index
        )

    def get_training_args(
        self,
        adapter_name: str,
        batch_size: int,
        grad_accum: int,
        epochs: int,
        logging_steps: int,
        learning_rate: float,
        save_steps: int,
        warmup_steps: int,
        eval_steps: int,
        seed: int,
    ) -> TrainingArguments:
        """Configures the hardware utilization and learning hyperparameters."""
        save_path = os.path.join(self.output_dir, adapter_name)

        return TrainingArguments(
            output_dir=save_path,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=1,
            eval_accumulation_steps=1,
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
            warmup_steps=warmup_steps,  # Replaces deprecated warmup_ratio
            eval_steps=eval_steps,      # Decoupled from save_steps
            save_steps=save_steps,
            seed=seed,                  # Ensures reproducibility
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
            report_to="none",
            remove_unused_columns=False, # Protects the manually created labels tensor
        )

    def train(
        self,
        adapter_name: str,
        batch_size: int = 4,
        grad_accum: int = 4,
        epochs: int = 1,
        logging_steps: int = 10,
        learning_rate: float = 2e-4,
        save_steps: int = 100,
        resume_from_checkpoint: str = None,
        warmup_steps: int = 50,
        eval_steps: int = 100,
        seed: int = 42,
    ):
        """Initializes the Trainer and executes the training loop."""
        args = self.get_training_args(
            adapter_name,
            batch_size,
            grad_accum,
            epochs,
            logging_steps,
            learning_rate,
            save_steps,
            warmup_steps,
            eval_steps,
            seed,
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=self.data_collator,
        )

        if resume_from_checkpoint:
            print(f"Resuming training from checkpoint: {resume_from_checkpoint}")
        else:
            print(f"Starting fresh training for {adapter_name} adapter...")

        # Pass the resume_from_checkpoint flag to the HF trainer
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        final_save_path = os.path.join(self.output_dir, adapter_name, "final")
        trainer.model.save_pretrained(final_save_path)
        self.tokenizer.save_pretrained(final_save_path)

        print(f"Training complete. Adapter saved to {final_save_path}")