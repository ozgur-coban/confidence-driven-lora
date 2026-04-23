"""
Copyright (C) 2026 Özgür ÇOBAN

This file is part of the confidence-driven-lora.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

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
            warmup_ratio=0.03,
            eval_steps=save_steps,  # Sync eval with saving
            save_steps=save_steps,  # Dynamic save steps
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
            report_to="none",
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
