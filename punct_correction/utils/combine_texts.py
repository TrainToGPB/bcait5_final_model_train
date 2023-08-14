import os
import re
import zipfile
import shutil
from tqdm import tqdm

def preprocess_text(text):
    pattern = r'\((.*?)\)/\((.*?)\)'
    matches = re.findall(pattern, text)
    if matches:
        for first_match, second_match in matches:
            special_characters = r'[!@#$%^&*()_+{}\[\]:;<>,.?~\\/\-|=]'
            only_numbers = r'^\d+\s*$'
            first_has_special = re.search(special_characters, first_match) is not None
            second_has_special = re.search(special_characters, second_match) is not None
            first_has_numbers = re.search(only_numbers, first_match) is not None
            second_has_numbers = re.search(only_numbers, second_match) is not None
            if first_has_special and not second_has_special:
                text = text.replace(f'({first_match})/({second_match})', f'{second_match}')
            elif not first_has_special and second_has_special:
                text = text.replace(f'({first_match})/({second_match})', f'{first_match}')
            elif first_has_numbers and not second_has_numbers:
                text = text.replace(f'({first_match})/({second_match})', f'{second_match}')
            elif not first_has_numbers and second_has_numbers:
                text = text.replace(f'({first_match})/({second_match})', f'{first_match}')
            else:
                text = text.replace(f'({first_match})/({second_match})', f'{first_match}')
    
    plus_pattern = r"\w+\+"
    slash_pattern = r"[a-z]+/"
    text = re.sub(plus_pattern, '', text)
    text = re.sub(slash_pattern, '', text)
    text = text.replace('\n', '')
    text = text.replace('/', '')

    return text

def combine_text_files(folder_path):
    contents = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.json'):
            continue
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
            content = preprocess_text(content)
            contents.append(content)

    combined_text = " ".join(contents)
    combined_text = re.sub(r"\s+", ' ', combined_text)
    output_path = '../data/raw_txt'
    output_filename = os.path.join(output_path, "/".join(folder_path.split('/')[-1:]) + ".txt")
    with open(output_filename, "w", encoding="utf-8") as output_file:
        output_file.write(combined_text)

def top_folder_combine(data_folder):
    """TODO
    data_folder는 여러 ZIP 파일이 들어있음(ZIP 말고 다른 파일은 없음)
    모든 각각의 ZIP 파일은 {최상위 경로}/{중간 경로들}/{txt파일들}로 구성되어 있음
    모든 '중간 경로들'에 대해 combine_text_files 함수를 적용하도록 함수를 구성

    1. data_folder 내의 ZIP 파일들을 돌아다니면서
    1-1. ZIP 파일의 압축을 해제
    1-2. 압축을 해제한 폴더 내부를 돌아다니며 {중간 경로}들에 대해 combine_text_files 함수 적용
    2. 모든 {중간 경로}들에 대해 처리가 끝난 경우 압축을 해제한 폴더를 삭제
    """
    for zip_filename in tqdm(os.listdir(data_folder)):
        zip_filepath = os.path.join(data_folder, zip_filename)
        with zipfile.ZipFile(zip_filepath, "r") as zip_ref:
            temp_dir = './temp'
            zip_ref.extractall(temp_dir)
            for upper_name in os.listdir(temp_dir):
                upper_path = os.path.join(temp_dir, upper_name)
                for middle_name in os.listdir(upper_path):
                    middle_path = os.path.join(upper_path, middle_name)
                    for target_name in tqdm(os.listdir(middle_path)):
                        target_path = os.path.join(middle_path, target_name)
                        combine_text_files(target_path)
                        shutil.rmtree(target_path)
                    shutil.rmtree(middle_path)
                shutil.rmtree(upper_path)
            shutil.rmtree(temp_dir)
        # os.remove(zip_filepath)
    print("All files are combined")

def main():
    top_folder_combine('../data/raw_zip')

if __name__ == "__main__":
    main()  
