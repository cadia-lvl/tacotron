import argparse
import os
from multiprocessing import cpu_count

from data import data_load
from hparams import hparams


def preprocess_ljspeech(args):
    '''
        Create the training directory that contains:
        * Linear scaled spectrograms for all files in the dataset
        * Mel scaled spectrograms for all files in the dataset
        * The metadata for this dataset

        Assumes that the dataset is organized in the following way
        at the base directory path:
        LJSpeech-1.1/
            wavs/
                LJ{id#1}.wav
                LJ{id#2}.wav
                ...
            metadata.csv
    '''
    in_dir = args.input_dir
    out_dir = os.path.join(args.output_dir, 'ljspeech')
    os.makedirs(out_dir, exist_ok=True)
    metadata = data_load.prep_ljspeech(in_dir, out_dir)
    write_metadata(metadata, args.output_dir)

def preprocess_icelandic(args):
    '''
        Create the training directory that contains:
        * Linear scaled spectrograms for all files in the dataset
        * Mel scaled spectrograms for all files in the dataset
        * The metadata for this dataset

        Assumes that the dataset is organized in the following way
        at the base directory path:
        TTS_icelandic_Google_m/
            ismData/
                tokens/
                    ism_{id#1}.token
                    ism_{id#2}.token
                    ...
                wavs/
                    ism_{id#1}.wav
                    ism_{id#2}.wav
                    ...
                line_index.tsv
    '''
    in_dir = os.path.join(args.input_dir, 'ismData')
    out_dir = os.path.join(args.output_dir, 'icelandic')
    os.makedirs(out_dir, exist_ok=True)
    metadata = data_load.prep_icelandic(in_dir, out_dir)
    write_metadata(metadata, out_dir)

def preprocess_unsilenced_icelandic(args):
    '''
        Create the training directory that contains:
        * Linear scaled spectrograms for all files in the dataset
        * Mel scaled spectrograms for all files in the dataset
        * The metadata for this dataset

        Assumes that the dataset is organized in the following way
        at the base directory path:
        TTS_icelandic_Google_m/
            ismData/
                tokens/
                    ism_{id#1}.token
                    ism_{id#2}.token
                    ...
                wavs/
                    ism_{id#1}.wav
                    ism_{id#2}.wav
                    ...
                line_index.tsv
    '''
    in_dir = os.path.join(args.input_dir, 'ismData')
    out_dir = os.path.join(args.output_dir, 'icelandic_silenced')
    os.makedirs(out_dir, exist_ok=True)
    metadata = data_load.prep_icelandic(in_dir, out_dir)
    write_metadata(metadata, out_dir)

def preprocess_ivona(args):
    '''
        Create the training directory that contains:
        * Linear scaled spectrograms for all files in the dataset
        * Mel scaled spectrograms for all files in the dataset
        * The metadata for this dataset

        Assumes that the dataset is organized in the following way
        at the base directory path:
        ivona_speech_data/
            ismData/
                ivona_txt/
                    ivona_{id#1}.txt
                    ivona_{id#2}.txt
                    ...
                Krisjan_export/
                    ivona_{id#1}.wav
                    ivona_{id#2}.wav
                    ...
                line_index.tsv

        The line index can be generated via metawave
    '''
    out_dir = os.path.join(args.output_dir, 'ivona')
    os.makedirs(out_dir, exist_ok=True)
    metadata = data_load.prep_ivona(args.input_dir, out_dir, trim_silence=args.trim_silence, 
        outlier_index_path=args.outlier_idx_path, index_path=args.index_path)
    write_metadata(metadata, out_dir)


def write_metadata(metadata, output_dir):
    '''
        Writes dataset metadata to train.txt into the given output
        directory that contains the following information for all files:
        "{lin spec file name} | {mel spec file name} | {num frames} | {text}"
    '''
    with open(os.path.join(output_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        for m in metadata:
            f.write('|'.join([str(x) for x in m]) + '\n')
        frames = sum([m[2] for m in metadata])
        hours = frames * hparams.get('frame_shift_ms') / (3600 * 1000)
        print('Wrote %d utterances, %d frames (%.2f hours)' % (len(metadata), frames, hours))
        print('Max input length:  %d' % max(len(m[3]) for m in metadata))
        print('Max output length: %d' % max(m[2] for m in metadata))   


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, help='Full path to the base directory of the dataset')
    parser.add_argument('--output_dir', required=True, help='Relative path from /home/<user> to the base output directory')
    parser.add_argument('--dataset_name', required=True, choices=['ljspeech', 'icelandic', 'unsilenced_icelandic', 'ivona'])
    parser.add_argument('--trim_silence', required=False, default=False)
    parser.add_argument('--index_path', required=False)
    parser.add_argument('--outlier_idx_path', required=False,  default=None)    
    args = parser.parse_args()
    if args.dataset_name == 'ljspeech':
        preprocess_ljspeech(args)
    elif args.dataset_name == 'icelandic':
        preprocess_icelandic(args)
    elif args.dataset_name == 'unsilenced_icelandic':
        preprocess_unsilenced_icelandic(args)
    elif args.dataset_name == 'ivona':
        preprocess_ivona(args)
    print('Data has been preprocessed and is now available at ', os.path.join(args.output_dir, args.dataset_name))
