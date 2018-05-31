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

    /home/<user>/<input_dir>/
        <dataset_1>/
            *.npy
            training.txt
        ...
    
    The output directory houses model meta, checkpoints and logs
    and is not temporary since it stores valuable output. Again,
    the ouput directory has a list of sub-directories, here each one
    represents a model.
    
    /home/<user>/<output_dir>/
        <model_1>
            logs, checkpoints, model meta
        ...


'''
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default=os.path.expanduser('~/tacotron_data'), 
        help='Defaults to /home/<user>/tacotron_data')
    # name of the subdirectory in base_dir which holds the training input
    parser.add_argument('--input', required=True, help='Name of subdir in base_dir holding the training input')
    # name of the model. This is used for creating a logging-directory
    # which will be at base_dir/logs-<name>
    parser.add_argument('--name', help='Name of the run. Defaults to tacotron', 
        default='tacotron')
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
    log_dir = os.path.join(args.base_dir, 'logs-%s' % args.name)
    os.makedirs(log_dir, exist_ok=True)
    model = Tacotron(hparams)
    # add stats before training
    stats = model.add_stats()
    model.train(log_dir,args)

main()
