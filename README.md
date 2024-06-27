### HomeWork 7
This homework need to use PEFT(Parameter Efficient Fine-Tuning) to fine-tune a
LLM(Large Language Model) and try to use different LoRA configs finally compare the result.

## Setup
!pip install transformers
!pip install datasets
!pip install accelerate
!pip install peft

## Result

Without LoRA 
trainable model parameters: 65783042 all model parameters: 65783042 percentage of trainable model parameters: 100.00%
Using LoRA
trainable model parameters: 813314 all model parameters: 66596356 percentage of trainable model parameters: 1.22%

Config 1

    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_lin", "k_lin","v_lin"],
    bias='none',
    task_type=TaskType.SEQ_CLS

Config 2

    r=8,
    lora_alpha=16,
    lora_dropout=0.01,
    target_modules=["q_lin", "k_lin","v_lin"],
    bias='none',
    task_type=TaskType.SEQ_CLS

Config 3

    r=16,
    lora_alpha=8,
    lora_dropout=0.002,
    target_modules=["q_lin", "k_lin","v_lin"],
    bias='none',
    task_type=TaskType.SEQ_CLS
