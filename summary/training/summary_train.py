import torch
import random
import torch.nn as nn
import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from transformers import T5TokenizerFast, T5ForConditionalGeneration, AutoModelForSeq2SeqLM
from transformers import default_data_collator
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
# from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer

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
OUTPUT_DIR = os.path.join(BASE_DIR, '../summary_models')

model_name = "paust/pko-t5-large"

# 데이터셋 클래스 정의
class PassageSummaryDataset(Dataset):
    def __init__(self, data_df, tokenizer):
        self.data = data_df
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        passage = self.data['passage'][idx]
        summary = self.data['summary'][idx]
        
        passage_inputs = self.tokenizer.encode_plus(
            passage,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        summary_labels = self.tokenizer.encode_plus(
            summary,
            padding="max_length",
            truncation=True,
            max_length=200,
            return_tensors="pt",
        )
        
        return {
            "input_ids": passage_inputs["input_ids"].squeeze(),
            "attention_mask": passage_inputs["attention_mask"].squeeze(),
            "label_ids": summary_labels["input_ids"].squeeze()
        }

# 모델 초기화
# tokenizer = MT5Tokenizer.from_pretrained(model_name) # google/mt5
tokenizer = T5TokenizerFast.from_pretrained(model_name) # polyglot-ko

# model = MT5ForConditionalGeneration.from_pretrained(model_name) # google/mt5
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.to(DEVICE)

# 데이터 불러오기
train_data = pd.read_csv(os.path.join(DATA_DIR, 'v2_short/train_short.csv'))
eval_data = pd.read_csv(os.path.join(DATA_DIR, 'v2_short/eval_short.csv'))

# 데이터셋과 데이터로더 초기화
train_dataset = PassageSummaryDataset(train_data, tokenizer)
eval_dataset = PassageSummaryDataset(eval_data, tokenizer)

# TrainingArguments 설정
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=1e-5,
    warmup_steps = 1050,
    lr_scheduler_type='linear',
    weight_decay=0.01,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,
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
    fp16=True,
    # predict_with_generate=True,
    # generation_max_length=200,
    load_best_model_at_end=True,
    metric_for_best_model="rougeL"
)

# 손실 함수 및 옵티마이저 설정
criterion = nn.CrossEntropyLoss()

# 학습 및 평가 함수 정의
def compute_metrics_rouge(pred):
    pred_ids = pred.predictions[0]
    label_ids = pred.label_ids

    # Convert the token IDs back to text
    pred_sentences = tokenizer.batch_decode(pred_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    label_sentences = tokenizer.batch_decode(label_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    # Initialize the ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    for pred_sent, label_sent in zip(pred_sentences, label_sentences):
        scores = scorer.score(pred_sent, label_sent)
        for rouge_type in ['rouge1', 'rouge2', 'rougeL']:
            rouge_scores[rouge_type].append(scores[rouge_type].fmeasure)

    avg_rouge_scores = {rouge_type: sum(scores) / len(scores) for rouge_type, scores in rouge_scores.items()}
    
    return avg_rouge_scores


def preprocess_logits_for_metrics(logits, labels):
    pred_ids = torch.argmax(logits[0], dim=-1)
    return pred_ids, labels

# WandB 초기화
wandb.init(project="final_summary", name='pkot5_baseline')

# Trainer 초기화
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=default_data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics_rouge,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
)

max_output_length = 200  # 원하는 최대 출력 토큰 길이
trainer.model.config.max_length = max_output_length

# 모델 훈련
trainer.train()

# 학습 완료 후 저장
best_model_path = OUTPUT_DIR
trainer.save_model(best_model_path)
