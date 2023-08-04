import torch
import pandas as pd
from tqdm import tqdm
from transformers import T5TokenizerFast, AutoModelForSeq2SeqLM
from pprint import pprint


def set_inference():
    # Set the device
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Model and Tokenizer paths
    MODEL_PATH = "../summary_models/checkpoint-5000"  # Change this to the path where you saved your trained model

    # Load the trained model and tokenizer
    tokenizer = T5TokenizerFast.from_pretrained(MODEL_PATH)

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    model.config.max_length = 200

    return model, tokenizer, DEVICE

# Inference function
def generate_summary(input_text, model, tokenizer, DEVICE):
    inputs = tokenizer.encode_plus(
        input_text,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)

    # Generate summary
    with torch.no_grad():
        output = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=200)
    summary = tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
    return summary


# 파일 inference
def file_inference(TEST_DATA_PATH, OUTPUT_CSV_PATH):
    # Path to your test CSV file containing passages
    # Load the test data
    test_data = pd.read_csv(TEST_DATA_PATH)

    # Generate summaries for each passage in the test data
    generated_summaries = []
    for passage in tqdm(test_data['passage']):
        summary = generate_summary(passage, model, tokenizer, DEVICE)
        generated_summaries.append(summary)
        print("[Passage]")
        pprint(passage)
        print("[Summary]")
        pprint(summary)

    # Add the generated summaries to the test_data DataFrame
    test_data['generated_summary'] = generated_summaries

    test_data.to_csv(OUTPUT_CSV_PATH, index=False)

    print("Inference completed and results saved to:", OUTPUT_CSV_PATH)


# 개별 inference
def single_inference(test_passage):
    generated_summary = generate_summary(test_passage, model, tokenizer, DEVICE)
    pprint(generated_summary)


if __name__ == "__main__":
    model, tokenizer, DEVICE = set_inference()

    # test_passage = "휴먼다큐 순례가 미국 퍼시픽 크레스트 트레일 구간 야생의 순간들을 프레임에 담았습니다. 명품 다큐멘터리의 산실 KBS가 2017년 새롭게 \
    # 선보이는 4부 작 UHD 다큐멘터리 순례의 제작진은 퍼시픽 크레스트 트레일을 배경으로 하는 4편 4,30km 한 걸음 나에게로의 스페셜 스틸컷을 공개했습니다. 순례 \
    # 4편 4,30km 한 걸음 나에게 오는 미국 남부, 멕시코 국경에서 출발해 캐나다 국경에서 끝나는 미국 서부 종단 트레일에 도전한 사람들의 이야기다. 먼저 눈길을 \
    # 사로잡는 것은 대자연이 지닌 야생의 아름다움입니다. 석양이 지는 사막과 드넓은 초원, 맑은 호수 등 일상에서는 보기 힘든 그림 같은 풍경입니다. 하지만 여행자들은 \
    # 뜨겁게 달궈진 사막에서 타는 듯한 목마름을 견디고, 거친 눈보라 속에서 꽁꽁 얼어붙는 추위와 싸우며 극한의 상황을 지나야 비경을 감상할 수 있습니다. 특히 뾰족한 \
    # 암석이 많은 가파르고 험난한 등산로는 부상의 위험을 동반한 난코스다. 또한 생명을 위협하는 곰의 습격이나 산불도 큰 위험요소다. 그야말로 인간이 경험할 수 있는 \
    # 모든 자연환경을 거치고서야 완수할 수 있는 극한의 도보여행입니다. 퍼시픽 크레스트 트레일을 완주하기 위해 걷는 길은 장장 4,30km. 하루도 쉬지 않고 40km씩을 \
    # 걷는다 해도 110일이 걸리는 험난한 여정입니다. 황량한 사막에서 출발해 눈 덮인 고원지대인 종착점에 닿으면, 이미 봄에 시작했던 일정은 가을을 지납니다. 일정이 \
    # 긴 만큼 챙겨야 할 짐이 많아 배낭의 무게도 무거울 수밖에 없습니다. 이 길은 여행자들이 반년 동안 꾸준히 걸어야 완주할 수 있습니다. 그들이 육체의 한계를 뛰어넘는 \
    # 도전을 통해 얻고자 하는 것이 과연 무엇일지는 4편 4,30km 한 걸음 나에게 록에서 확인할 수 있습니다. 순례 제작진은 6개월 동안 퍼시픽 크레스트 트레일의 전 구간 \
    # 4,30km 동행 취재하며 야생의 아름다움을 오롯이 담아낼 수 있었습니다. UHD로 구현되는 실사와 같은 절경은 도시의 삶에 지친 현대인들의 안식처가 되는 것은 물론 \
    # 색다른 볼거리를 제공할 것이라고 자신했습니다."
    # passage_tokens = tokenizer(test_passage, return_tensors='pt')
    # print(f"Passage token length: {len(passage_tokens['input_ids'][0])}")
    # single_inference(test_passage)

    TEST_DATA_PATH = "../data/v2_short/eval_short.csv"  # Change this to the path of your test CSV file
    OUTPUT_CSV_PATH = "../results/baseline.csv"  # Change this to the desired output CSV path
    file_inference(TEST_DATA_PATH, OUTPUT_CSV_PATH)
