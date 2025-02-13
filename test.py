from datasets import load_dataset, Dataset
from transformers import LlamaTokenizer, DataCollatorForLanguageModeling, AutoModelForCausalLM
from transformers.pytorch_utils import Conv1D
from torch.utils.data import DataLoader
import torch
from functools import partial
from peft import initialize_sva_weights
from peft.tuners.sva.config import SvaConfig
from peft.tuners.sva.layer import SvaLayer, Linear
from peft.tuners.tuners_utils import check_target_module_exists
from peft.tuners._buffer_dict import BufferDict
from peft import get_peft_model

def target_module_check_fn_default(name, module, peft_config):
    "check if a module is an adapter module via target_modules"
    is_target_module = True
    if peft_config.target_modules is not None:
        is_target_module = check_target_module_exists(peft_config, name)
    # Conv1D for GPT2 support
    return isinstance(module, (torch.nn.Linear, Conv1D)) and is_target_module


# Load the Hellaswag dataset
hellaswag = load_dataset("Rowan/hellaswag", split="train[:1%]")  # Using only 1% for dummy dataset

# Initialize the Llama 2 tokenizer
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token

# Tokenization function
def tokenize_function(examples):
    texts = [ctx + " " + ending for ctx, endings in zip(examples["ctx"], examples["endings"]) for ending in endings]
    return tokenizer(texts, truncation=True, padding="max_length", max_length=128)

# Apply the tokenization
hellaswag_tokenized = hellaswag.map(tokenize_function, batched=True, remove_columns=hellaswag.column_names)

# Data collator for causal language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Causal language modeling doesn't use masked language modeling
)

# Create DataLoader
dataloader = DataLoader(hellaswag_tokenized, batch_size=4, shuffle=True, collate_fn=data_collator)

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model = model.to("cuda")

peft_config = SvaConfig(
    target_modules=["q_proj", "v_proj", "k_proj"],
    r=16,
)

target_module_check_fn = partial(target_module_check_fn_default, peft_config=peft_config)

#sva_state_dict = get_sva_state_dict(
#    model=model,
#    dataloader=dataloader,
#    peft_config=peft_config
#)



model = get_peft_model(model, peft_config)
initialize_sva_weights(model, dataloader=dataloader)

dummy_input = torch.randint(0, 100, (4, 128)).to("cuda")

import IPython; IPython.embed(); exit(1)

base_layer = model.model.layers[0].self_attn.q_proj
k = "model.layers.0.self_attn.q_proj"

sva_A = BufferDict({}, persistent=True)
sva_B = BufferDict({}, persistent=True)
sva_A["default"] = eva_state_dict[f"{k}.default.lora_A.weight"]
sva_B["default"] = eva_state_dict[f"{k}.default.lora_B.weight"]

# import IPython; IPython.embed(); exit(1)

layer = Linear(
    base_layer=base_layer,
    sva_A=sva_A,
    sva_B=sva_B,
    adapter_name="default",
    r=16,
    sva_dropout=0.0,
    fan_in_fan_out=False,
    is_target_conv_1d_layer=False,
    init_weights=True,
)

x = torch.randn(1, 128, 4096).to("cuda")

layer(x)

import IPython; IPython.embed(); exit(1)

model = get_peft_model(model, peft_config)


for n, m in model.named_modules():
    if isinstance(m, Linear):
        m.sva_A.default