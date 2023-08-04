import os
import zipfile
from pydub import AudioSegment
from tqdm import tqdm

def merge_wav_files(wav_folder_path):
    wav_segments = []
    for filename in sorted(os.listdir(wav_folder_path)):
        if filename.endswith('.wav'):
            wav_file_path = os.path.join(wav_folder_path, filename)
            wav_segments.append(AudioSegment.from_wav(wav_file_path))

    merged_wav = AudioSegment.empty()
    for wav_segment in wav_segments:
        merged_wav += wav_segment

    output_dir = wav_folder_path.split('/')
    output_dir[-4] = 'wav_test'
    output_filename = output_dir.pop() + '.wav'
    output_dir = '/'.join(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    merged_wav.export(os.path.join(output_dir, output_filename), format='wav')

def mwf_parent(wav_parent_folder_path):
    for wav_folder_path in os.listdir(wav_parent_folder_path):
        wav_folder_path = os.path.join(wav_parent_folder_path, wav_folder_path)
        if os.path.isdir(wav_folder_path):
            merge_wav_files(wav_folder_path)

def mwf_grand_parent(wav_grand_parent_folder_path):
    for wav_parent_folder_path in tqdm(os.listdir(wav_grand_parent_folder_path)):
        print(f"\nFolder {wav_parent_folder_path} is being processed...\n")
        wav_parent_folder_path = os.path.join(wav_grand_parent_folder_path, wav_parent_folder_path)
        if os.path.isdir(wav_parent_folder_path):
            mwf_parent(wav_parent_folder_path)

def mwf_top_unzip(wav_top_folder_path, remove_zip=False):
    for zipfile_name in os.listdir(wav_top_folder_path):
        if zipfile_name.endswith('.zip'):
            zipfile_path = os.path.join(wav_top_folder_path, zipfile_name)
            with zipfile.ZipFile(zipfile_path, 'r') as zip_ref:
                temp_folder_path = os.path.splitext(zipfile_path)[0]  # 압축 해제를 위한 임시 폴더 경로
                temp_folder_path = os.path.join(temp_folder_path, '..')
                zip_ref.extractall(temp_folder_path)
        if remove_zip:
            os.remove(zipfile_path)

def mwf_top(wav_top_folder_path):
    for wav_grand_parent_folder_path in tqdm(os.listdir(wav_top_folder_path)):
        wav_grand_parent_folder_path = os.path.join(wav_top_folder_path, wav_grand_parent_folder_path)
        if os.path.isdir(wav_grand_parent_folder_path):
            print(f"\nParent Folder {wav_grand_parent_folder_path.split('/')[-1]} is being processed...")
            mwf_grand_parent(wav_grand_parent_folder_path)

if __name__ == '__main__':
    # mwf_top('../data/wav')
    merge_wav_files('../data/wav_part/D11/G02/S001025')