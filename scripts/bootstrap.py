#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
from urllib import request
from zipfile import ZipFile, ZipInfo

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def current_os():
  if sys.platform.startswith('linux'):
    return 'linux'
  elif sys.platform.startswith('win'):
    return 'win'
  elif sys.platform == 'darwin':
    return 'mac'
  else:
    raise ValueError(f'Unsupported platform: {sys.platform}')

def download_and_extract(url, dest_dir):
  # https://stackoverflow.com/questions/39296101
  class ZipFileWithPermissions(ZipFile):
    def _extract_member(self, member, targetpath, pwd):
      if not isinstance(member, ZipInfo):
        member = self.getinfo(member)
      targetpath = super()._extract_member(member, targetpath, pwd)
      attr = member.external_attr >> 16
      if attr != 0:
        os.chmod(targetpath, attr)
      return targetpath

  zip_file = 'temp.zip'
  request.urlretrieve(url, zip_file)
  with ZipFileWithPermissions(zip_file, 'r') as zip:
    zip.extractall(dest_dir)
  os.remove(zip_file)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--weights', default=None, help='Path to model weights')
  args = parser.parse_args()

  # Download weights if not specified.
  if args.weights:
    weights = args.weights
  else:
    weights = 'stories15M.bin'
    if not os.path.exists(weights):
      url = f'https://huggingface.co/karpathy/tinyllamas/resolve/main/{weights}'
      request.urlretrieve(url, weights)

  # Download and bootstrap GN.
  gn_url = 'https://github.com/yue/build-gn'
  gn_version = 'v0.10.0'
  gn_dir = os.path.join(ROOT_DIR, 'third_party/build-gn')
  if not os.path.isdir(gn_dir):
    download_and_extract(f'{gn_url}/releases/download/{gn_version}/gn_{gn_version}_{current_os()}_x64.zip',
                         gn_dir)
    subprocess.run([ sys.executable,
                     os.path.join(gn_dir, 'tools/clang/scripts/update.py') ])

  # Generate ninja files.
  gn_args = [
    f'llama2_c_weigets="//{weights}"',
    'is_component_build=false',
    'is_debug=false',
    'is_official_build=true',
  ]
  gn = os.path.join(gn_dir, 'gn')
  if current_os() == 'win':
    gn += '.exe'
  subprocess.run([ gn, 'gen', os.path.join(ROOT_DIR, 'out/Release'),
                   f'--args={" ".join(gn_args)}' ])

if __name__ == '__main__':
  exit(main())
