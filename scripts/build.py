#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys

from bootstrap import ROOT_DIR, current_os

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('targets', nargs='*', default=[ 'all' ],
                      help='The targets to build')
  parser.add_argument('-C', dest='out_dir', default='out/Release',
                      help='Which config to build')
  args, unknown_args = parser.parse_known_args()

  ninja = os.path.join(ROOT_DIR, 'third_party/build-gn/ninja')
  if current_os() == 'win':
    ninja += '.exe'
  ninja_args = [ ninja,  '-C', args.out_dir ]

  try:
    subprocess.run(ninja_args + unknown_args + args.targets, check=True)
  except KeyboardInterrupt:
    sys.exit(1)
  except subprocess.CalledProcessError as e:
    sys.exit(e.returncode)

if __name__ == '__main__':
  main()
