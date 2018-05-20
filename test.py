import argparse
import os
import tensorflow as tf
from datetime import datetime
from hparams import hparams
from model.tacotron import Tacotron

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default=os.path.expanduser('~/tacotron_data'))
    parser.add_argument('--input', default='training_demo')
    parser.add_argument('--model', default='tacotron')
    parser.add_argument('--name', help='Name of the run. Used for logging. Defaults to model name.')
    #parser.add_argument('--hparams', default=hparams,
    #    help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    parser.add_argument('--restore_step', type=int, help='Global step to restore from checkpoint.')
    parser.add_argument('--summary_interval', type=int, default=100,
        help='Steps between running summary ops.')
    parser.add_argument('--checkpoint_interval', type=int, default=1000,
        help='Steps between writing checkpoints.')
    parser.add_argument('--slack_url', help='Slack webhook URL to get periodic reports.')
    parser.add_argument('--tf_log_level', type=int, default=1, help='Tensorflow C++ log level.')
    parser.add_argument('--git', action='store_true', help='If set, verify that the client is clean.')
    args = parser.parse_args()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(args.tf_log_level)
    run_name = args.name or args.model
    log_dir = os.path.join(args.base_dir, 'logs-%s' % run_name)
    os.makedirs(log_dir, exist_ok=True)
    with tf.variable_scope('model') as scope:
        model = Tacotron(hparams)
        # stats = add_stats(model)
        model.train(log_dir,args)

main()
