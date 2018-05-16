import os
import argparse
from data.data_feed import DataFeeder
from text.text_tools import onehot_to_text


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default=os.path.expanduser('~/tacotron_data'))
    parser.add_argument('--input_dir', default='training')
    args = parser.parse_args()
    d = DataFeeder(os.path.join(args.base_dir, args.input_dir))
    a = d._get_next_superbatch()
    for b in a:
        print(onehot_to_text(b[0][0]))