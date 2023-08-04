import torch
import pandas as pd
import os
from pprint import pprint
from tqdm import tqdm
from collections import defaultdict
from transformers import MT5ForConditionalGeneration, MT5Tokenizer

# Set the device
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("DEVICE:", DEVICE)

BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, '../data')
MODEL_DIR = os.path.join(BASE_DIR, '../qg_models/with_answer')

# Set the model and tokenizer paths
model_name = "google/mt5-large"

# Load the model and tokenizer
tokenizer = MT5Tokenizer.from_pretrained(model_name)
model = MT5ForConditionalGeneration.from_pretrained(MODEL_DIR)
model.to(DEVICE)

# Define the inference function
def generate_questions(context, n_beams=3):
    input_ids = tokenizer.encode_plus(
        context,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).input_ids.to(DEVICE)

    generated_ids = model.generate(
        input_ids,
        num_beams=n_beams,
        max_length=128,
        early_stopping=True,
        num_return_sequences=n_beams,
    )

    generated_qas = []
    for generated_id in generated_ids:
        qa = tokenizer.decode(generated_id, skip_special_tokens=True)
        generated_qas.append(qa)
    return generated_qas

# Example usage
test_data = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
n_beams, n_qa = 10, 3
questions, answers = defaultdict(list), defaultdict(list)
for context_idx, context in tqdm(test_data.context.items(), total=len(test_data)):
    generated_qas = generate_questions(context, n_beams=n_beams)
    total_answers = []
    for qa_idx, qa in enumerate(generated_qas):
        try:
            question, answer = qa.split('<extra_id_-2>')
            question = question.replace('<extra_id_-1>', '').strip()
            answer = answer.strip()
        except ValueError:
            question = '<no_question>'
            answer = '<no_answer>'
        if answer != '<no_answer>':
            if answer in question:
                question = '<no_question>'
                answer = '<no_answer>'
            if answer in total_answers:
                question = '<no_question>'
                answer = '<no_answer>'
            for ta in total_answers:
                if ta in answer or answer in ta:
                    question = '<no_question>'
                    answer = '<no_answer>'
                    break
        if answer != '<no_answer>':
            questions['question' + str(len(total_answers) + 1)].append(question)
            answers['answer' + str(len(total_answers) + 1)].append(answer)
            total_answers.append(answer)
        if len(total_answers) == n_qa:
            break
    if len(total_answers) < n_qa:
        for i in range(len(total_answers), n_qa):
            questions['question' + str(i + 1)].append('<no_question>')
            answers['answer' + str(i + 1)].append('<no_answer>')
            total_answers.append('<no_answer>')

test_data_with_qa = test_data.copy()
qa_order = []
for (q_idx, q_list), (a_idx, a_list) in zip(sorted(questions.items()), sorted(answers.items())):
    test_data_with_qa[q_idx] = q_list
    test_data_with_qa[a_idx] = a_list
    qa_order.extend([q_idx, a_idx])

column_order = ['id', 'title'] + qa_order + ['context']
test_data_with_qa = test_data_with_qa[column_order]
test_data_with_qa.to_csv('../results/test_with_answer.csv', index=False)

# # context = "다이아몬드 또는 금강석은 천연광물중 가장 굳기가 우수하며 광채가 뛰어난 보석이다. 주성분은 탄소이며 분자구조상의 차이로 인해 동일한 원자로 구성된 자연 산물인 흑연과는 매우 다른 특성을 가진 보석이다. 뛰어난 경도로 인해 공업용으로도 많이 쓰이나, 대부분의 공업용 다이아몬드는 인간이 만든 인조 다이아몬드를 쓴다. 현재에 이르러서는 질이 나쁜 자연 다이아몬드와 질이 좋은 공업용 다이아몬드를 식별하기 어렵다. 흔히 다이아몬드의 질량의 단위에는 대부분의 보석의 질량의 단위로 쓰이는 캐럿을 쓴다. 0.25캐럿 이하로 세공된, 크기가 작은 다이아몬드는 멜레라고 부른다. 다이아몬드는 강도에 따라 화씨 1400도부터 1607도 사이에서 완전히 연소된다. 실제로 다이아몬드가 형성되는 곳은 땅속 깊이 130킬로미터 아래에서이다. 땅 위에서 발견되는 것은 화산이 분출할 때 함께 땅 위로 솟아오른 것이다. 보통 색깔이 없는 것을 귀하게 생각하며, 간혹 매우 특별한 색깔의 것이 가치를 인정받는 경우도 있다. 다이아몬드는 두 번째로 단단한 강옥보다 90배나 더 단단하다."
# context = "다이아몬드 또는 금강석은 천연광물중 가장 굳기가 우수하며 광채가 뛰어난 보석이다. 주성분은 탄소이며 분자구조상의 차이로 인해 동일한 원자로 구성된 자연 산물인 흑연과는 매우 다른 특성을 가진 보석이다. 뛰어난 경도로 인해 공업용으로도 많이 쓰이나, 대부분의 공업용 다이아몬드는 인간이 만든 인조 다이아몬드를 쓴다."
# pprint(context)
# generated_qas = generate_questions(context, n_beams=3)
# print("[Generated Questions]")
# for qa in generated_qas:
#     print(qa)
