import argparse
import os
from data.data_feed import SimpleDataFeeder
from datetime import datetime

def test_feeder(in_dir):
    feeder = SimpleDataFeeder(in_dir)
    feeder.get_next_superbatch()


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--base_dir', default=os.path.expanduser('~/tacotron_data'))
  parser.add_argument('--input', default='training_demo')
  parser.add_argument('--name', default='tacotron_simple')
  args = parser.parse_args()
  run_name = args.name or args.model
  log_dir = os.path.join(args.base_dir, 'logs-%s' % run_name)
  os.makedirs(log_dir, exist_ok=True)
  test_feeder(os.path.join(args.base_dir, args.input))

if __name__ == '__main__':
  main()
