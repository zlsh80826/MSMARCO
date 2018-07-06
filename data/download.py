import argparse
import wget
import shutil
import os
import zipfile
import hashlib

GLOVE_ZIP_CHECKSUM = '313d432da12af70755766ba6dc47f491'

v1_url = ['https://msmarco.blob.core.windows.net/msmarco/train_v1.1.json.gz',
          'https://msmarco.blob.core.windows.net/msmarco/dev_v1.1.json.gz',
          'https://msmarco.blob.core.windows.net/msmarco/test_public_v1.1.json.gz']
        

v2_url = ['https://msmarco.blob.core.windows.net/msmarco/train_v2.1.json.gz',
          'https://msmarco.blob.core.windows.net/msmarco/dev_v2.1.json.gz',
          'https://msmarco.blob.core.windows.net/msmarco/eval_v2.1_public.json.gz']

def md5check(file_to_check, checksum):
    with open(file_to_check, 'rb') as f:
        cs = hashlib.md5(f.read()).hexdigest()
        if cs != checksum:
            return False
        else:
            return True

def download(version):
    if (version == 'v1'):
        try:
            os.makedirs('v1')
        except OSError as e:
            if e.errno != errno.EEIST:
                raise
            
        for url in v1_url:
            wget.download(url, out='v1')
        os.rename('v1/train_v1.1.json.gz', 'v1/train.json.gz')
        os.rename('v1/dev_v1.1.json.gz', 'v1/dev.json.gz')
        os.rename('v1/test_public_v1.1.json.gz', 'v1/test_public.json.gz')
        shutil.copy('v1/dev.json.gz', 'v1/test.json.gz')
    else:
        try:
            os.makedirs('v2')
        except OSError as e:
            if e.errno != errno.EEIST:
                raise

        for url in v2_url:
            wget.download(url, out='v2')

        os.rename('v2/train_v2.1.json.gz', 'v2/train.json.gz')
        os.rename('v2/dev_v2.1.json.gz', 'v2/dev.json.gz')
        os.rename('v2/eval_v2.1_public.json.gz', 'v2/test_public.json.gz')
        shutil.copy('v2/dev.json.gz', 'v2/test.json.gz')
    
    download_glove = True

    if os.path.exists('glove.840B.300d.zip'):
        download_glove = not md5check('glove.840B.300d.zip', GLOVE_ZIP_CHECKSUM)

    if download_glove:
        wget.download('http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip')

    with zipfile.ZipFile('glove.840B.300d.zip', "r") as zip_ref:
        zip_ref.extractall(".")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download MSMARCO dataset')
    parser.add_argument('version', choices=['v1', 'v2'])
    args = parser.parse_args()
    download(args.version)
