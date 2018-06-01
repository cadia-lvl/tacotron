import os 
from pydub import AudioSegment
import argparse

def detect_leading_silence(wav, silence_threshold = -50.0, chunk_size = 10):
    trim_ms = 0
    
    assert chunk_size > 0
    while wav[trim_ms: trim_ms + chunk_size].dBFS < silence_threshold and trim_ms < len(wav):
        trim_ms += chunk_size
    
    return trim_ms

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default=os.path.expanduser('~/tacotron_data/unsilenced_icelandic/ismData/wavs'))
    parser.add_argument('--output_dir', default='trimmed')
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)
    for file in os.listdir(input_dir):
        if file[-4:] != '.wav':
            # not a wav file
            continue
        current_file_path = os.path.join(input_dir, file)
        print(current_file_path)
        output_file_path = os.path.join(output_dir, file)

        file_stats = os.stat(current_file_path)

        if file_stats.st_size is 0:
            continue

        wav = AudioSegment.from_file(current_file_path, format='wav')
        start_trim = detect_leading_silence(wav)
        end_trim = detect_leading_silence(wav.reverse())
        duration = len(wav)
        trimmed_sound = wav[start_trim:duration - end_trim]        
        trimmed_sound.export(output_file_path, format='wav')
