"""LoRA fine-tuning, the same for every condition and seed.

Run one training per process: TRL fails when a second trainer starts in the same one.
"""
import torch
from datasets import Dataset

from . import config as C


def train(model_name: str, records: list[dict], out_dir, seed: int, epochs: int,
          max_steps=None):
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    lora = LoraConfig(r=C.LORA_R, lora_alpha=C.LORA_ALPHA, lora_dropout=C.LORA_DROPOUT,
                      target_modules=C.LORA_TARGETS, bias="none", task_type="CAUSAL_LM")
    kwargs = dict(
        output_dir=str(out_dir / "_trainer"), seed=seed, data_seed=seed,
        learning_rate=C.LEARNING_RATE, num_train_epochs=epochs,
        per_device_train_batch_size=C.MICRO_BATCH, gradient_accumulation_steps=C.GRAD_ACCUM,
        gradient_checkpointing=C.GRADIENT_CHECKPOINTING, max_length=C.MAX_LEN,
        completion_only_loss=True, packing=False, bf16=True,
        # TRL loads the base in float32 unless told otherwise.
        model_init_kwargs={"dtype": C.DTYPE},
        logging_steps=25, save_strategy="no", report_to="none")
    if max_steps:
        kwargs["max_steps"] = max_steps
    ds = Dataset.from_list([{"prompt": r["prompt"], "completion": r["completion"]}
                            for r in records])
    trainer = SFTTrainer(model=model_name, args=SFTConfig(**kwargs),
                         train_dataset=ds, peft_config=lora)
    largest = max(trainer.model.parameters(), key=lambda p: p.numel())
    assert largest.dtype == getattr(torch, C.DTYPE), f"base weights are {largest.dtype}"
    trainer.train()
    trainer.save_model(str(out_dir))
    return trainer.state.log_history
