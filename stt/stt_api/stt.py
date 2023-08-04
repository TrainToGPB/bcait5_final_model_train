import os
import time
import openai, whisper
from pydub import AudioSegment
import speech_recognition as sr
from threading import Thread


def split_audio_by_time(audio_segment, chunk_duration=3):
    chunk_length = chunk_duration * 60 * 1000  # Convert chunk duration from minutes to milliseconds
    chunks = []
    for i in range(0, len(audio_segment), chunk_length):
        chunk = audio_segment[i:i+chunk_length]
        chunks.append(chunk)
    return chunks

def milliseconds_to_time(milliseconds):
    seconds = milliseconds / 1000
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    formatted_time = f"{hours}h {minutes}m {seconds:.2f}s" if hours > 0 else f"{minutes}m {seconds:.2f}s"
    return formatted_time

def whisper_transcribe(audio_path, whisper_version, wav_path=None):
    audio_format = audio_path.split('.')[-1]
    audio_segment = AudioSegment.from_file(audio_path, format=audio_format)
    
    if wav_path:
        wav_dir = '/'.join(wav_path.split('/')[:-1])
        os.makedirs(wav_dir, exist_ok=True)
        audio_segment.export(wav_path, format='wav')
    
    print(f"[Whisper Transcription ({whisper_version})]")
    print(f"Audio Length (Time): {milliseconds_to_time(len(audio_segment))} ({len(audio_segment)} ms)")
    print(f"Audio Frame Rate: {audio_segment.frame_rate / 1000} kHz")
    print(f"Audio Sample Width: {2**(audio_segment.sample_width+1)}-bit")

    assert "HOSTED" in whisper_version or whisper_version == "API", \
        "Please write appropriate Whisper's version (HOSTED_LARGE/MEDIUM/SMALL or API)"
    
    if "HOSTED" in whisper_version:
        stt_result = whisper_transcribe_hosted(audio_path)

    elif whisper_version == "API":
        chunk_duration = 3
        wav_chunks = split_audio_by_time(audio_segment, chunk_duration)

        stt_parts, threads = [], []
        for idx, chunk in enumerate(wav_chunks):
            print(f"Processing chunk {idx+1}...")
            thread = Thread(target=multi_worker, args=(chunk, idx, stt_parts))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        stt_parts = [stt_part for _, stt_part in sorted(stt_parts)]
        stt_result = "\n\n".join(stt_parts)

    return stt_result

def whisper_transcribe_hosted(audio_path):   
    version_mapping = {
        "LARGE": "large-v2",
        "MEDIUM": "medium",
        "SMALL": "small"
    }
    version = version_mapping[whisper_version.split('_')[-1]]
    model = whisper.load_model(version)
    stt_result = model.transcribe(audio_path)["text"]
    
    return stt_result

def whisper_transcribe_api(audio_segment):   
    with open('../credentials/openai/openai_api_key.txt', 'r') as file:
        OPENAI_API_KEY = file.read().strip()
    openai.api_key = OPENAI_API_KEY

    recognizer = sr.Recognizer()
    audio_data = sr.AudioData(audio_segment.raw_data, audio_segment.frame_rate, audio_segment.sample_width)
    stt_result = recognizer.recognize_whisper_api(audio_data, api_key=OPENAI_API_KEY)
    
    return stt_result

def multi_worker(chunk, idx, results_list):
    result = whisper_transcribe_api(chunk)
    if result:
        results_list.append((idx, result))

if __name__ == "__main__":
    """
    STT Models (whisper_version)
    - WHI_HOS_SMALL: OpenAI Whisper hosted small (price=free, limit=??) [O]
    - WHI_HOS_MEDIUM: OpenAI Whisper hosted medium (price=free, limit=??) [O]
    - WHI_HOS_LARGE: OpenAI Whisper hosted large-v2 (price=free, limit=??) [O]
    - WHI_API: OpenAI Whisper API (price=0.006$/min, limit=25MB) [O]
    """
    whisper_version = "API"
    output_dir = f"../results/WHISPER_{whisper_version}/sample"
    os.makedirs(output_dir, exist_ok=True)

    audio_path = "../data/wav_test/format_sample/sample-3.aif"
    audio_format = audio_path.split('.')[-1]
    audio_name = audio_path.split('/')[-1][:-(len(audio_format)+1)]

    audio_segment = AudioSegment.from_file(audio_path, format=audio_format)
    result_name = audio_name.replace('audio', 'result', 1) + '.txt'

    start_time = time.time()
    stt_result = whisper_transcribe(audio_path, whisper_version, wav_path=None)
    end_time = time.time()
    stt_time = (end_time - start_time) * 1000 # in millisecond
    print(f"Audio STT Time: {milliseconds_to_time(stt_time)}")

    with open(os.path.join(output_dir, result_name), "w", encoding="utf-8") as f:
        f.write(stt_result)
    