import torch
import random
import torch.nn as nn
import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from transformers import AutoConfig, AutoTokenizer, AutoModelForTokenClassification
from transformers import Trainer, TrainingArguments, EvalPrediction
from transformers import default_data_collator
from sklearn.metrics import accuracy_score

import os
import wandb


def set_seed(SEED=42):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def basic_settings(DATA_PATH, OUTPUT_PATH):
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("DEVICE:", DEVICE)

    BASE_DIR = os.getcwd()
    DATA_DIR = os.path.join(BASE_DIR, DATA_PATH)
    OUTPUT_DIR = os.path.join(BASE_DIR, OUTPUT_PATH)

    return DEVICE, DATA_DIR, OUTPUT_DIR


# 데이터셋 클래스 정의
class PunctCorrectionDataset(Dataset):
    def __init__(self, data_df, tokenizer):
        self.data = data_df
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        context = self.data['removed'][idx]
        infos = eval(self.data['position'][idx])

        inputs = self.tokenizer(
            context,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_token_type_ids=False
        )

        labels = [-100] * len(inputs["input_ids"].squeeze())
        for label_info in infos:
            labels[label_info[1]] = label_info[0]
        labels = torch.tensor(labels, dtype=torch.long)
        inputs["labels"] = labels

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": labels
        }


# 모델 초기화
def train_settings(model_name, DEVICE, TRAIN_DIR, EVAL_DIR, OUTPUT_DIR, SEED=42):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_config = AutoConfig.from_pretrained(model_name)
    model_config.num_labels = 4
    model = AutoModelForTokenClassification.from_pretrained(model_name, config=model_config)
    model.to(DEVICE)

    # 데이터 불러오기
    train_data = pd.read_csv(TRAIN_DIR)
    eval_data = pd.read_csv(EVAL_DIR)[:80]

    # 데이터셋과 데이터로더 초기화
    train_dataset = PunctCorrectionDataset(train_data, tokenizer)
    eval_dataset = PunctCorrectionDataset(eval_data, tokenizer)

    # TrainingArguments 설정
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=1e-5,
        warmup_steps = 251,
        lr_scheduler_type='linear',
        weight_decay=0.01,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        save_total_limit=1,
        logging_dir="./logs",
        logging_strategy='steps',
        evaluation_strategy='steps',
        save_strategy='steps',
        logging_steps=10,
        eval_steps=100,
        save_steps=100,
        seed=SEED,
        report_to="wandb",
        # fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy"
    )

    return model, training_args, train_dataset, eval_dataset, tokenizer

# 학습 및 평가 함수 정의
def compute_metrics(p: EvalPrediction) -> dict:
    preds = p.predictions.argmax(axis=2)
    labels = p.label_ids
    
    target_preds = preds[labels != -100]
    target_labels = labels[labels != -100]
    accuracy = accuracy_score(target_labels, target_preds)
    
    return {
        "accuracy": accuracy,
    }


def train(model, training_args, train_dataset, eval_dataset, tokenizer, OUTPUT_DIR):
    # WandB 초기화
    wandb.init(project="final_punct", name='roberta_baseline')

    # Trainer 초기화
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
    )

    # 모델 훈련
    trainer.train()

    # 학습 완료 후 저장
    best_model_path = OUTPUT_DIR
    trainer.save_model(best_model_path)


def main():
    set_seed()
    DATA_PATH = '../data/processed'
    OUTPUT_PATH = '../pc_models'
    DEVICE, DATA_DIR, OUTPUT_DIR = basic_settings(DATA_PATH, OUTPUT_PATH)

    train_filename = 'train.csv'
    TRAIN_DIR = os.path.join(DATA_DIR, train_filename)
    eval_filename = 'eval.csv'
    EVAL_DIR = os.path.join(DATA_DIR, eval_filename)
    model_name = 'klue/roberta-large'
    model, training_args, train_dataset, eval_dataset, tokenizer = train_settings(
        model_name, 
        DEVICE, 
        TRAIN_DIR, 
        EVAL_DIR, 
        OUTPUT_DIR
    )
    train(
        model, 
        training_args, 
        train_dataset, 
        eval_dataset, 
        tokenizer, 
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()
