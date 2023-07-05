from concurrent.futures import ThreadPoolExecutor
from functools import partial
import io
import urllib

import PIL.Image
import argparse

from datasets import load_dataset
from datasets.utils.file_utils import get_datasets_user_agent
import os


USER_AGENT = get_datasets_user_agent()


def fetch_single_image(image_url, timeout=None, retries=0):
    image = None
    for _ in range(retries + 1):
        try:
            request = urllib.request.Request(
                image_url,
                data=None,
                headers={"user-agent": USER_AGENT},
            )
            with urllib.request.urlopen(request, timeout=timeout) as req:
                image = PIL.Image.open(io.BytesIO(req.read()))
            break
        except Exception:
            image = None
    return image


def fetch_images_save(batch, num_threads, timeout=None, retries=0, save_dir='./'):
    fetch_single_image_with_args = partial(fetch_single_image, timeout=timeout, retries=retries)
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for i, res in enumerate(list(executor.map(fetch_single_image_with_args, batch["image_url"]))):
            if res is not None:
                res.save(os.path.join(save_dir, '{:07}.jpg'.format(batch['index'][i])))
    return batch


def fetch_images(batch, num_threads, timeout=None, retries=0):
    fetch_single_image_with_args = partial(fetch_single_image, timeout=timeout, retries=retries)
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        batch["image"] = list(executor.map(fetch_single_image_with_args, batch["image_url"]))
    return batch


def add_index(dset, split):
    return dset[split].add_column('index', list(range(len(dset[split]))))


def main():
    parser = argparse.ArgumentParser('download image caption dataset from huggingface dataset')
    parser.add_argument('--num_threads', type=int, default=10)
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=-1)
    parser.add_argument('--dataset_name', type=str, required=True, default=None)
    parser.add_argument('--split', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--retries', type=int, default=0)
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()
    num_threads = args.num_threads
    print('loading the dataset')
    dset = load_dataset(args.dataset_name)
    print('adding index column')
    target_dset = add_index(dset, args.split)
    print('filtering row numbers')
    target_dset = target_dset.filter(lambda x: x['index']<args.end_idx and x['index']>=args.start_idx)
    print('downloading the images')
    if args.save:
        target_dset = target_dset.map(fetch_images_save, batched=True, batch_size=100, fn_kwargs={'retries': args.retries, "num_threads": num_threads, 'save_dir': args.save_dir})
    else:
        target_dset = target_dset.map(fetch_images, batched=True, batch_size=100, fn_kwargs={"num_threads": num_threads, 'retries': args.retries})

if __name__=='__main__':
    main()