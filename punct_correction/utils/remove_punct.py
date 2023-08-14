import os
import random
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer


class PunctDicts:
    punct2num = {
        ',': 0,
        '.': 1,
        '!': 2,
        '?': 3
    }
    num2punct = {
        0: ',',
        1: '.',
        2: '!',
        3: '?'
    }


def remove_punctuation(text, tokenizer):
    tokens = tokenizer.tokenize(text)
    removed_position = []
    for idx, token in enumerate(tokens):
        if token in ',.!?' and random.random() < 0.7:
            punct_dicts = PunctDicts()
            punct2num = punct_dicts.punct2num
            del tokens[idx]
            removed_position.append([punct2num[token], idx])

    removed_text = tokenizer.convert_tokens_to_string(tokens)
    return removed_text, removed_position


def make_missing_punct_csv(folder_path, tokenizer):
    original, removed, position = [], [], []
    for txt_file in tqdm(os.listdir(folder_path)):
        """TODO
        1. txt 파일들을 돌면서, 각 파일의 text를 tokenizing하고 512토큰 단위로 분할 및 저장(str) -> original에 append
        2. 분할된 텍스트에 remove_punctuation 함수를 적용하고 removed_text를 저장(str) -> removed에 append
        3. 2번의 remove_punctuation 적용 결과 중 하나인 removed_position을 저장(list) -> position에 append
        """
        with open(os.path.join(folder_path, txt_file), 'r', encoding='utf-8') as f:
            text = f.read()

        tokens = tokenizer.tokenize(text)
        chunked_tokens = [tokens[i:i+512] for i in range(0, len(tokens), 512)]

        for chunk in chunked_tokens:
            chunk_text = tokenizer.convert_tokens_to_string(chunk)
            removed_text, removed_pos = remove_punctuation(chunk_text, tokenizer)
            removed_text = removed_text.replace("# # ", "##")

            original.append(chunk_text)
            removed.append(removed_text)
            position.append(removed_pos)
    """TODO
    4. 모든 txt 파일을 다 돌았다면 original, removed, position을 각각의 이름을 칼럼으로 가지는 하나의 pd.DataFrame으로 저장
    5. 해당 DataFrame을 CSV로 저장
    """
    df_path = '../data/processed'
    df = pd.DataFrame({'original': original, 'removed': removed, 'position': position})
    df.to_csv(os.path.join(df_path, 'total.csv'), index=False)


def main():
    model_name = 'klue/roberta-large'
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    folder_path = '../data/raw_txt'
    make_missing_punct_csv(folder_path, tokenizer)


def sample_test():
    from pprint import pprint
    model_name = 'klue/roberta-large'
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    sample_text = """
        저 같은 경우는 이제 그, 원래 평론가죠. 평론가들이 보통 이제 어디를 주로 다니냐면, \
        전시회를 보러 다니면서, 뭐 전시회 평을 쓰기도 하고 또 감상하기도 하는데, 저도 전시회는 가는데 \
        제가 즐겨가는 전시회는 이런, 산업기계 박람회 같은 거 있죠? 코엑스나 킨텍스 가면 이런 거 많이 하거든요? \
        거기서 봤던 장면인데요. 소위 그 CNC 레이저 커팅 머신이죠. 아마 여기 그, 공대 학생들은 잘 아실 텐데, \
        CNC라고 하는 거는, computerized numerical control의 약자입니다. 즉 수치, 방식의 어떤 그 기계를 \
        가공하는 방식이거든요? 그런데 이거를 제가, 영상을 찍으면서, 뭐 너무 당연, 사실 너무 당연한 \
        얘기인데, 기계라는 게 사람보다 훨씬 능력이 낫구나. 훨씬 빠르고, 지치지 않고, 또 같은 \
        동작을 얼마든지 반복할 수 있고 그리고, 또한 한편으로는, 놀라운 스펙터클을 만들어내죠. 정말 이걸 보면서, \
        어떻게 한 치의 오차도 없이 저렇게 깎아 낼까? 그래서 제가 생각한 주제는 뭐냐면, 기계는 애시당초 옛날에 처음 \
        나왔을 때부터, 인간을 훨씬 뛰어넘었다. 즉 초월해 있는 존재다. 인간을 초월한 존재 중 가장 절대적인 존재는, \
        하늘에 있는 신이죠. 근데 이제, 요즘 사람들은 신의 존재만으로 만족하지 않는 것 같아요. 수많은 기계들을 사용한단 \
        말이죠. 지금 이 강의만 해도, 컴퓨터가 있고, 프로젝트가 있고, 포인터가 있고, 마이크 같은 기계들이 있는데, 이거의 \
        도움 없이는 꼼짝 못하거든요? 이런 것들이, 당장 마이크는 내 목소리를 훨씬 크게 해서, 공간에 넓게 퍼지게 해준단 \
        말이죠. 그래서 저는 기계의 특성은, 초월적이다. 종교적 초월하고 당연히 다르죠. 일단 초월이라는 말은, 아주 \
        간단하게 그렇게 정의하기로 해요. 사람이 어떤 상태에서 살고 있거든요? 하루 세끼 밥 먹어야 되고, 너무 많이 일하면 \
        힘들고, 졸리면 자야 되잖아요? 그 상태를 넘어설 수 있게 해주는 것. 사실은 초월의 단계가 되게 많죠. 예를 들어서 \
        저는, 훌륭한 운동 선수가, 놀라운 플레이들 많이 보여주잖아요. 100m 세계기록 보유자인, 우사인 볼트가 달리는 모습을 \
        보면 저건 인간이 아니다 싶어요. 그 사람은 정말, 육체적인 능력과, 엄청난 노력으로, 보통 사람의 상태를 초월한 거죠.
    """
    removed_text, removed_position = remove_punctuation(sample_text, tokenizer)
    pprint(removed_text)
    
    tokens = tokenizer.tokenize(removed_text)
    added_punct = 0
    punct_dicts = PunctDicts
    num2punct = punct_dicts.num2punct
    for punct, position in removed_position:
        tokens.insert(position + added_punct, num2punct[punct])
        added_punct += 1
    restored_text = tokenizer.convert_tokens_to_string(tokens)
    pprint(restored_text)


def data_split_and_save(TOTAL_DIR, TRAIN_DIR, EVAL_DIR, TEST_DIR):
    total = pd.read_csv(TOTAL_DIR)
    train_temp, test = train_test_split(total, test_size=1/20, random_state=42)
    train, valid = train_test_split(train_temp, test_size=3/19, random_state=42)
    train.to_csv(TRAIN_DIR, index=False)
    valid.to_csv(EVAL_DIR, index=False)
    test.to_csv(TEST_DIR, index=False)

    print("Train size:", len(train))
    print("Validation size:", len(valid))
    print("Test size:", len(test))


if __name__ == "__main__":
    main()

    # TOTAL_DIR = '../data/processed/total.csv'
    # TRAIN_DIR = '../data/processed/train.csv'
    # EVAL_DIR = '../data/processed/eval.csv'
    # TEST_DIR = '../data/processed/test.csv'
    # data_split_and_save(
    #     TOTAL_DIR,
    #     TRAIN_DIR,
    #     EVAL_DIR,
    #     TEST_DIR
    # )
