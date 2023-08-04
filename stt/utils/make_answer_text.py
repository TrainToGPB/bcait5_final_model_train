import os
import re

def concat_wav_files(wav_dir, output_dir):
    file_paths = [
        os.path.join(wav_dir, f) for f in os.listdir(wav_dir) if f.endswith('.wav')
    ]
    file_paths = sorted(file_paths)

    wav_data_list = []
    for file_path in file_paths:
        audio_segment = AudioSegment.from_file(file_path, format="wav")
        wav_data_list.append(audio_segment)

    combined_wav = wav_data_list[0]
    for wav_data in wav_data_list[1:]:
        combined_wav += wav_data

    return combined_wav

def concat_txt_files(txt_dir, output_dir):
    file_paths = [
        os.path.join(txt_dir, f) for f in os.listdir(txt_dir) if f.endswith('.txt')
    ]
    file_paths = sorted(file_paths)

    combined_txt = ''

# 최종 정답 텍스트를 저장할 변수 초기화
final_text = ''

# 파일 수에 따라 반복
num_files = 100  # 예시에서는 10개의 파일이 있다고 가정
for i in range(num_files):
    # segment 파일 이름 생성 (6자리 숫자)
    file_name = f'./data/txt/{i:06d}.txt'
    print(file_name)
    
    # segment 파일 읽기
    try:
        with open(file_name, 'r') as file:
            segment_text = file.read()
    except:
        continue
    print(segment_text)
    
    # 괄호 내부의 텍스트 처리 (정규식으로 "/(문자)" 형태의 글 제거)
    segment_text = re.sub(r'\/\([^)]*\)', '', segment_text)
    print(segment_text)
    
    # 정규식을 사용하여 문자 제거
    segment_text = re.sub(r'\b[nb]/', '', segment_text)
    segment_text = re.sub(r'[^\s가-힣a-zA-Z0-9?!\.]', '', segment_text)
    print(segment_text)
    
    # 최종 정답 텍스트에 추가
    final_text += segment_text.strip() + ' '  # segment 텍스트 추가 후 공백 추가

# 최종 정답 텍스트 출력
# print(final_text)

# 저장할 파일 이름
output_file = './final_text.txt'

# final_text를 txt 파일로 저장
with open(output_file, 'w') as file:
    file.write(final_text)
    
print(f'{output_file}에 저장되었습니다.')
