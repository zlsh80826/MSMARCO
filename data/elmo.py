import h5py 
import pickle
from tqdm import tqdm

known, vocab, chars = pickle.load(open('vocabs.pkl', 'rb'))

with open('vocabs.txt', 'w') as file:
    for v in vocab:
        if "/" not in v:
            file.write(v + '\n')



