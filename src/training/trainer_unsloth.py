import os
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported
class CustomCompletionCollator(DataCollatorForSeq2Seq):
    """
    A lightweight, dependency-free replacement for TRL's DataCollatorForCompletionOnlyLM.
    Inherits padding logic from Seq2Seq, then manually applies -100 masking to prompts.
    """
    def __init__(self, response_template_ids: list, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.response_template_ids = response_template_ids

    def __call__(self, features, return_tensors=None):
        # 1. Let Seq2Seq collator handle dynamic padding first
        batch = super().__call__(features, return_tensors)
        
        # 2. FIX: Causal LM requires labels to be a copy of input_ids. 
        # If they don't exist, create them.
        if "labels" not in batch or batch["labels"] is None:
            batch["labels"] = batch["input_ids"].clone()
            
        labels = batch["labels"].clone()
        template_len = len(self.response_template_ids)
        
        # 3. FIX: Ensure any padding tokens added by the collator are ignored in the loss calculation
        if self.tokenizer is not None and self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
        
        # 4. Search for the template and mask the prompt
        for i in range(labels.shape[0]):
            label_seq = labels[i].tolist()
            match_idx = -1
            for j in range(len(label_seq) - template_len + 1):
                if label_seq[j : j + template_len] == self.response_template_ids:
                    match_idx = j + template_len
                    break
            
            if match_idx != -1:
                labels[i, :match_idx] = -100
            else:
                pass
                
        batch["labels"] = labels
        return batch
class AdapterTrainer:
    def __init__(self, model, tokenizer, train_dataset, eval_dataset=None, output_dir: str = "./models/adapters"):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.output_dir = output_dir

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        response_template="Answer:"
        template_ids=self.tokenizer.encode(response_template, add_special_tokens=False)


        self.data_collator = CustomCompletionCollator(
            response_template_ids=template_ids,
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
            return_tensors="pt"
        )

    def get_training_args(
        self, adapter_name: str, batch_size: int, grad_accum: int, epochs: int, logging_steps: int, learning_rate: float, eval_steps: int
    ) -> TrainingArguments:
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
            # UNSLOTH HARDWARE DETECTION
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            optim="adamw_8bit", # Recommended for Unsloth to save VRAM
            max_grad_norm=0.3,
            warmup_ratio=0.03,
            eval_steps=eval_steps,
            save_steps=eval_steps,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
            dataloader_num_workers=2,
            report_to="none" 
        )

    def train(
        self, adapter_name: str, batch_size: int = 4, grad_accum: int = 4, epochs: int = 1, logging_steps: int = 10, learning_rate: float = 2e-4, eval_steps:int=100
    ):
        
        args = self.get_training_args(adapter_name, batch_size, grad_accum, epochs, logging_steps, learning_rate, eval_steps)
        
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
        
        # UNSLOTH SPECIFIC SAVE COMMAND
        trainer.model.save_pretrained_merged(final_save_path, self.tokenizer, save_method="lora")
        
        print(f"Training complete. Adapter saved to {final_save_path}")