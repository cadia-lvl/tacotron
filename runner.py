import argparse
import os
import tensorflow as tf
from datetime import datetime
from hparams import hparams
from model.tacotron import Tacotron


'''
    Assumed directory structure:

    The input directory houses the pre-processed
    data and is considered temporary or at least expandable.
    Under the input directory is a list of sub-directories
    identified by the dataset-name. 

    /home/<user>/<input_base_dir>/
        <dataset_1>/
            *.npy
            training.txt
        ...
    
    The output directory houses model meta, checkpoints and logs
    and is not temporary since it stores valuable output. Again,
    the ouput directory has a list of sub-directories, here each one
    represents a model.
    
    /home/<user>/<output_base_dir>/
        <model_1>
            logs/
                ...
            model/
                checkpoints
                model meta
            samples/
                Synthesized training samples
                alignment graphs
        ...
'''

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', help='relative path from /home/<user> to base preprocessed data directory'),
    parser.add_argument('--output_dir', help='relative path from /home/<user> to the base output directory')
    parser.add_argument('--dataset_name', help='The given dataset has to exist in the given input directory')
    parser.add_argument('--model_name', 
        help='name of model to be trained on, defaults to main. This is just used for file-keeping.', 
        default='main')
    parser.add_argument('--restore_step', type=int, help='Global step to restore from checkpoint.')
    parser.add_argument('--summary_interval', type=int, default=100,
        help='Steps between running summary ops.')
    parser.add_argument('--checkpoint_interval', type=int, default=1000,
        help='Steps between writing checkpoints.')
    # used for broadcasting training updates to slack.
    parser.add_argument('--slack_url', help='Slack webhook URL to get periodic reports.')
    parser.add_argument('--tf_log_level', type=int, default=1, help='Tensorflow C++ log level.')
    args = parser.parse_args()
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(args.tf_log_level)
    args.in_dir = os.path.expanduser('~/'+os.path.join(args.input_dir, args.dataset_name))
    args.out_dir = os.path.expanduser('~/'+os.path.join(args.output_dir, args.model_name))
    args.log_dir = os.path.join(args.out_dir, 'logs')
    args.meta_dir = os.path.join(args.out_dir, 'meta')
    args.sample_dir = os.path.join(args.out_dir, 'samples')    
    # create the output directories if needed
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.meta_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)

    model = Tacotron(hparams)
    model.train(args)

main()
