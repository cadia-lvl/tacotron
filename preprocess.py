from multiprocessing import cpu_count
from datasets import lj_speech
import argparse
import os
from hparams import hparams

def preprocess_ljspeech(args):
    '''
    '''
    os.makedirs(args.output_dir, exist_ok=True)
    metadata = lj_speech.load_data(args.output_dir)
    write_metadata(metadata, args.output_dir)


def write_metadata(metadata, output_dir):
    with open(os.path.join(output_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        for m in metadata:
            f.write('|'.join([str(x) for x in m]) + '\n')
        frames = sum([m[2] for m in metadata])
        hours = frames * hparams.frame_shift_ms / (3600 * 1000)
        print('Wrote %d utterances, %d frames (%.2f hours)' % (len(metadata), frames, hours))
        print('Max input length:  %d' % max(len(m[3]) for m in metadata))
        print('Max output length: %d' % max(m[2] for m in metadata))   

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='training')
    parser.add_argument('--dataset', required=True, choices=['ljspeech'])
    args = parser.parse_args()
    if args.dataset == 'ljspeech':
        preprocess_ljspeech(args)
