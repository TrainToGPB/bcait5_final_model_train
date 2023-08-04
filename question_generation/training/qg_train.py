import torch
import random
import torch.nn as nn
import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
from transformers import default_data_collator
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from nltk.translate.bleu_score import corpus_bleu

import os
import wandb


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("DEVICE:", DEVICE)

BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, '../data')
OUTPUT_DIR = os.path.join(BASE_DIR, '../qg_models/with_answer')

model_name = "google/mt5-large"

# 데이터셋 클래스 정의
class QuestionGenerationDataset(Dataset):
    def __init__(self, data_df, tokenizer):
        self.data = data_df
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        context = self.data['context'][idx]
        question = self.data['question'][idx]
        answer = self.data['answer_text'][idx]
        
        context_inputs = self.tokenizer.encode_plus(
            context,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        question_answer_labels = self.tokenizer.encode_plus(
            tokenizer.cls_token + question + tokenizer.sep_token + answer,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        
        return {
            "input_ids": context_inputs["input_ids"].squeeze(),
            "attention_mask": context_inputs["attention_mask"].squeeze(),
            "label_ids": question_answer_labels["input_ids"].squeeze()
        }

# 모델 초기화
tokenizer = MT5Tokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({'cls_token': '<cls>', 'sep_token': '<sep>'})
model = MT5ForConditionalGeneration.from_pretrained(model_name)
model.to(DEVICE)

# 데이터 불러오기
train_data = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
eval_data = pd.read_csv(os.path.join(DATA_DIR, 'eval.csv'))

# 데이터셋과 데이터로더 초기화
train_dataset = QuestionGenerationDataset(train_data, tokenizer)
eval_dataset = QuestionGenerationDataset(eval_data, tokenizer)
# print(train_dataset[0]['label_ids'])
# print(tokenizer.decode(train_dataset[0]['label_ids'], skip_special_tokens=False))
# exit(0)

# TrainingArguments 설정
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=1e-5,
    warmup_steps = 3090,
    lr_scheduler_type='linear',
    weight_decay=0.01,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=2,
    save_total_limit=1,
    logging_dir="./logs",
    logging_strategy='steps',
    evaluation_strategy='steps',
    save_strategy='steps',
    logging_steps=100,
    eval_steps=1000,
    save_steps=1000,
    seed=SEED,
    report_to="wandb",
)

# 손실 함수 및 옵티마이저 설정
criterion = nn.CrossEntropyLoss()

# 학습 및 평가 함수 정의
def compute_metrics(pred):
    pred_ids = pred.predictions[0]
    label_ids = pred.label_ids

    # Calculate BLEU score
    bleu_score = corpus_bleu([[label] for label in label_ids], pred_ids)

    return {"BLEU": bleu_score}

def preprocess_logits_for_metrics(logits, labels):
    pred_ids = torch.argmax(logits[0], dim=-1)
    return pred_ids, labels

# WandB 초기화
wandb.init(project="final_question_generation", name='mt5_large_with_answer')

# Trainer 초기화
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=default_data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
)

# 모델 훈련
trainer.train()

# 학습 완료 후 저장
best_model_path = OUTPUT_DIR
trainer.save_model(best_model_path)