import argparse
import wget
import shutil
import os
import zipfile

v1_url = ['https://msmarco.blob.core.windows.net/msmarco/train_v1.1.json.gz',
		  'https://msmarco.blob.core.windows.net/msmarco/dev_v1.1.json.gz',
		  'https://msmarco.blob.core.windows.net/msmarco/test_public_v1.1.json.gz']
		

v2_url = ['https://msmarco.blob.core.windows.net/msmarco/train_v2.1.json.gz',
		  'https://msmarco.blob.core.windows.net/msmarco/dev_v2.1.json.gz',
		  'https://msmarco.blob.core.windows.net/msmarco/eval_v2.1_public.json.gz']

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
		shutil.copy('v1/dev.json.gz', 'v1/test.json.gz')
		wget.download('http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip')
		with zipfile.ZipFile("glove.840B.300d.zip", "r") as zip_ref:
			zip_ref.extractall(".")

	else:
		raise NotImplementedError

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Download MSMARCO dataset')
	parser.add_argument('version', choices=['v1', 'v2'])
	args = parser.parse_args()
	download(args.version)
