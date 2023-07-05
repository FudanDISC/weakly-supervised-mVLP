# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import os
import time
import queue
import logging
import datetime
import warnings
import argparse
import threading
from io import BytesIO

import requests
from PIL import Image
import tqdm
from multiprocessing import Pool

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

retries = 0
save_dir = None

headers = {
    #'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36',
    'User-Agent':'Googlebot-Image/1.0', # Pretend to be googlebot
    'X-Forwarded-For': '64.18.15.200'
}

def request_image(img_info):
    url, img_index = img_info
    target = os.path.join(save_dir, '{:07d}.jpg'.format(img_index))
    for t in range(retries + 1):
        try:
            i = requests.get(url, stream=False, timeout=10, allow_redirects=True, headers=headers)
            img = Image.open(BytesIO(i.content))
            img = img.conert('RGB')
        except:
            continue
        with open(target, 'wb') as wf:
            wf.write(i.content)
            break


def load_tsv(tsv_fn, url_index=0):
    data = []
    i = 0
    with open(tsv_fn, 'r') as rf:
        for line in tqdm.tqdm(rf, desc='loading the data'):
            url = line.strip().split('\t')[url_index]
            data.append((url, i))
            i += 1
    return data


def main():
    parser = argparse.ArgumentParser(description='downloader for wukong dataset')
    parser.add_argument('--tsv_file', help='dir path for csv file', required=True)
    parser.add_argument('--img_dir', help='path to save images to', required=True)
    parser.add_argument('--start_id', help='start id for csv file', type=int, default=0)
    parser.add_argument('--end_id', help='end id for csv file', type=int, default=-1)
    parser.add_argument('--thread_num', help='download thread count', type=int, default=4)
    parser.add_argument('--retries', type=int, default=0, help='the number of retries')
    parser.add_argument('--url_index', type=int, default=0, help='the column index for url in the tsv file')

    args = parser.parse_args()
    tsv_file = args.tsv_file
    img_dir = args.img_dir
    start_id = args.start_id
    end_id = args.end_id
    thread_num = args.thread_num

    global retries
    retries = args.retries
    global save_dir
    save_dir = args.img_dir

    if not os.path.exists(tsv_file):
        logger.error('csv dir %s not exists', tsv_file)
        raise FileNotFoundError
    if not os.path.isdir(img_dir):
        logger.error('img dir %s not exists', img_dir)
        raise FileNotFoundError

    raw_data = load_tsv(tsv_file, args.url_index)
    raw_data = raw_data[start_id:end_id]
    # with Pool(thread_num) as p:
    for res in tqdm.tqdm(map(request_image, raw_data), desc='downloading the images'):
        continue



if __name__ == '__main__':
    main()
