import argparse
import os
import argparse
import time

import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.optim import AdamW
from torch.utils.data import DataLoader
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    PeftType,
    PrefixTuningConfig,
    PromptEncoderConfig,
    LoraConfig
)
from datasets import Dataset, DatasetDict
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed, \
    InputExample
from tqdm import tqdm

# ------------------------init parameters----------------------------
parser = argparse.ArgumentParser(description='P-tuning v2')
parser.add_argument('--plm', type=str, default="bert-large", help='choose plm: bert or roberta or ernie')
parser.add_argument('--type', type=str, default="ptuningv2", help='choose model: ptuningv1, ptuningv2, lora')
parser.add_argument('--rank', type=int, default=4, help='LORA rank')
parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
parser.add_argument('--soft_tokens', type=int, default=10, help='soft tokens')
parser.add_argument('--train_file', type=str, default='datasets/preprocessed/train.json', help='train set')
parser.add_argument('--valid_file', type=str, default='datasets/preprocessed/dev.json', help='validation set')
parser.add_argument('--test_file', type=str, default='datasets/preprocessed/test.json', help='test set')
parser.add_argument('--train', action="store_true", help='If train or not')
parser.add_argument('--eval', action="store_true", help='If evaluate or not')
args = parser.parse_args()

if args.plm == 'bert':
    model_name_or_path = "bert-base-chinese"
elif args.plm == 'roberta':
    model_name_or_path = "uer/chinese_roberta_L-12_H-768"
elif args.plm == 'ernie':
    model_name_or_path = "nghuyong/ernie-3.0-base-zh"
elif args.plm == 'bert-large':
    model_name_or_path = "yechen/bert-large-chinese"
elif args.plm == 'bert-large':
    model_name_or_path = "yechen/bert-large-chinese"
elif args.plm == 'roberta-large':
    model_name_or_path = "hfl/chinese-roberta-wwm-ext-large"

device = "cuda"
batch_size = 4
num_epochs = 8
lr = args.lr

if args.type=='ptuningv2':
    peft_type = PeftType.PREFIX_TUNING
    peft_config = PrefixTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=args.soft_tokens)
elif args.type=='lora':
    peft_type = PeftType.LORA
    peft_config = LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=args.rank, 
                             lora_alpha=16, lora_dropout=0.1, target_modules=['query', 'key', 'value', 'output.dense'])
elif args.type=='ptuningv1':
    peft_type = PeftType.P_TUNING
    peft_config = PromptEncoderConfig(task_type="SEQ_CLS", num_virtual_tokens=args.soft_tokens)

# if any(k in model_name_or_path for k in ("gpt", "opt", "bloom")):
#     padding_side = "left"
# else:
#     padding_side = "right"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
if getattr(tokenizer, "pad_token_id") is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Load Dataset
train_path = args.train_file # 'datasets/preprocessed/train.json'
dev_path = args.valid_file # 'datasets/preprocessed/dev.json'
test_path = args.test_file # 'datasets/preprocessed/test.json'
datasets = DatasetDict.from_json({'train': train_path, 'dev': dev_path, 'test': test_path})

def tokenize_function(examples):
    # max_length=None => use the model max length (it's actually the default
    outputs = tokenizer(examples["claim"], examples["evidences"], truncation=True, max_length=256)
    return outputs

tokenized_datasets = datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=["claimId", "claim", "evidences"],
)
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

def collate_fn(examples):
    return tokenizer.pad(examples, return_tensors="pt")

# Instantiate dataloaders.
train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, collate_fn=collate_fn, batch_size=batch_size)
dev_dataloader = DataLoader(
    tokenized_datasets["dev"], shuffle=False, collate_fn=collate_fn, batch_size=batch_size
)
test_dataloader = DataLoader(
    tokenized_datasets["test"], shuffle=False, collate_fn=collate_fn, batch_size=batch_size
)

model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, return_dict=True, num_labels=3)
# model_sd = model.state_dict()
# for k in list(model_sd.keys()):
#       print('module:', k)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
model.to(device)

optimizer = AdamW(params=model.parameters(), lr=lr)
# Instantiate scheduler
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(train_dataloader) * num_epochs),
)

total_time = 0
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    best_microf1 = 0
    best_macrof1 = 0
    best_recall = 0
    best_precision = 0
    start_time = time.time()
    # ========================================
    #               Training
    # ========================================
    if args.train:
        for step, batch in enumerate(tqdm(train_dataloader)):
            batch.to(device)
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.sum()
            loss.sum().backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        end_time = time.time()
        training_time = end_time - start_time
        total_time += training_time

        # ========================================
        #               Validation
        # ========================================
        model.eval()
        valid_y_pred = []
        valid_y_true = []
        for step, batch in enumerate(tqdm(dev_dataloader)):
            batch.to(device)
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.softmax(dim=-1)
            predictions = torch.argmax(predictions, dim=-1)
            labels = batch['labels']
            valid_y_true.extend(labels.cpu().tolist())
            valid_y_pred.extend(predictions.cpu().tolist())
        pre, recall, f1, _ = precision_recall_fscore_support(valid_y_true, valid_y_pred, average='micro')
        microf1 = f1
        pre, recall, f1, _ = precision_recall_fscore_support(valid_y_true, valid_y_pred, average='macro')
        if f1 > best_macrof1:
            best_microf1 = microf1
            best_macrof1 = f1
            torch.save(model.state_dict(), f"./checkpoint/lora_model.ckpt")
        print("Epoch {}, f1 {}".format(epoch, f1), flush=True)
    print('Training complete in {:.0f}m {:.0f}s'.format(total_time // 60, total_time % 60))

# ========================================
#               Test
# ========================================
if args.eval:
    model.load_state_dict(torch.load(f"./checkpoint/lora_model.ckpt"))
    model = model.to(device)
    model.eval()
    test_y_pred = []
    test_y_true = []
    for step, batch in enumerate(tqdm(test_dataloader)):
        batch.to(device)
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.softmax(dim=-1)
        # predictions = torch.argmax(predictions, dim=-1)
        predictions = torch.argmax(predictions)
        class_labels = ['SUP', 'REF', 'NEI']
        predicted_label = class_labels[predictions]
        print("預測的 label 為:", predicted_label)
    #     labels = batch['labels']
    #     test_y_true.extend(labels.cpu().tolist())
    #     test_y_pred.extend(predictions.cpu().tolist())
    # pre, recall, f1, _ = precision_recall_fscore_support(test_y_true, test_y_pred, average='micro')
    # print("       F1 (micro): {:.2%}".format(f1))
    # pre, recall, f1, _ = precision_recall_fscore_support(test_y_true, test_y_pred, average='macro')
    # print("Precision (macro): {:.2%}".format(pre))
    # print("   Recall (macro): {:.2%}".format(recall))
    # print("       F1 (macro): {:.2%}".format(f1))