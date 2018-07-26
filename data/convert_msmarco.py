from tqdm import tqdm
import gzip
import json
import nltk
import re
import bisect
import numpy as np
import multiprocessing
import time
import argparse
import random
from utils import smith_waterman

def preprocess(s):
    return s.replace("''", '" ').replace("``", '" ')

def tokenize(s, context_mode=False ):
    nltk_tokens=[t.replace("''", '"').replace("``", '"') for t in nltk.word_tokenize(s)]
    additional_separators = (
        "-", "\u2212", "\u2014", "\u2013", "/", "~", '"', "'", "\u201C", "\u2019", "\u201D", "\u2018", "\u00B0")
    tokens = []
    for token in nltk_tokens:
        tokens.extend([t for t in (re.split("([{}])".format("".join(additional_separators)), token)
                                   if context_mode else [token])])
    assert(not any([t=='<NULL>' for t in tokens]))
    assert(not any([' ' in t for t in tokens]))
    assert (not any(['\t' in t for t in tokens]))
    return tokens

def trim_empty(tokens):
    return [t for t in tokens if t != '']

def process(i, j, is_test):
    p = j['passages']
    outputs = []

    YesNoHead = ['is', 'are', 'was', 'were', 'do', 'does', 'did',
                 'can', 'could', 'should', 'has', 'have', 'may', 
                 'might', 'am', 'will', 'would']

    query   = preprocess(j['query'])
    qtokens =  trim_empty(tokenize(query))
    yesno = False
    if qtokens[0].lower() in YesNoHead:
        yesno = True

    temp = ' '.join(pp['passage_text'] for pp in p)
    if yesno:
        temp += ' Yes No Answer Present.'
    else:
        if random.random() > 0.7:
            temp += ' No Answer Present.'
    context = preprocess(temp)

    ctokens = trim_empty(tokenize(context, context_mode=True))
    normalized_context = ' '.join(ctokens)
    nctokens = normalized_context.split()

    if not is_test:
        for a in j['answers']:
            bad = False
            answer = preprocess(a)
            if answer == '':
                bad = True

            if not bad:
                atokens = trim_empty(tokenize(answer, context_mode=True))
                normalized_answer = ' '.join(atokens).lower()
                normalized_context_lower = normalized_context.lower()
                pos = normalized_context_lower.find(normalized_answer)
                if pos >= 0:
                    start = bisect.bisect(np.cumsum([1 + len(t) for t in nctokens]), pos)
                    end = start + len(atokens)
                    if len(nctokens) < end:
                        bad = True
                else:
                    natokens = normalized_answer.split()
                    try:
                        (start, end), (_, _), score = smith_waterman(normalized_context_lower.split(), natokens)
                        start -= 1
                        ratio = 0.5 * score / min(len(nctokens), len(natokens))
                        if ratio < 0.8:
                            bad = True
                    except:
                        bad = True
                if not bad:
                    output = [str(j['query_id']), j['query_type'], 
                          ' '.join(nctokens), ' '.join(qtokens),
                          ' '.join(nctokens[start:end]), str(start), str(end)]
                    outputs.append(output)
    else:
        output = [str(j['query_id']), j['query_type'], ' '.join(nctokens),' '.join(qtokens)]
        outputs.append(output)
        
    return outputs
                
def convert(file_name, outfile, is_test, num_threads, version, ratio):
    print('Generating', outfile, '...')
    start = time.perf_counter()

    if version == 'v1':
        ##### v1 ####
        data = []
        with gzip.open(file_name, 'rb') as f:
            for line in f:
                data.append(json.loads(line))
    elif version == 'v2':
        #### v2 ####
        data = []
        with gzip.open(file_name, 'rb') as f:
            raw_data = json.load(f)

        for id_ in raw_data['query_id']:
            select = random.random() <= ratio

            if file_name.find('dev') < 0 or select:
                dict_ = dict()
                dict_['passages'] = raw_data['passages'][id_]
                dict_['query'] = raw_data['query'][id_]
                dict_['query_id'] = raw_data['query_id'][id_]
                dict_['query_type'] = raw_data['query_type'][id_]
                if not is_test:
                    dict_['answers'] = raw_data['answers'][id_]
                data.append(dict_)
    else:
        raise NotImplementedError

    with multiprocessing.Pool(num_threads) as pool:
        results = []
        for idx, d in enumerate(tqdm(data)):
            result = pool.apply_async(process, args=(idx, d, is_test))
            results.append(result)
        pool.close()
        pool.join()
        
    outputs = []

    for result in results:
        for output in result.get():
            outputs.append(output)            

    with open(outfile, 'w', encoding='utf-8') as out:
        for output in outputs:
            out.write("%s\n"%'\t'.join(output))
    end = time.perf_counter()
    print('Take', end - start, 'second.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert MSMARCO raw data to tsv format')
    parser.add_argument('--threads', help='Number of threads to multi-preprocessing', default=1, type=int)
    parser.add_argument('--ratio', help='Ratio of dev data', default=1, type=float)
    parser.add_argument('version', choices=['v1', 'v2'])
    args = parser.parse_args()

    if args.version == 'v1':
        convert('v1/train.json.gz', 'train.tsv', False, args.threads, args.version, args.ratio)
        convert('v1/dev.json.gz', 'dev.tsv', False, args.threads, args.version, args.ratio)
        convert('v1/test.json.gz', 'test.tsv', True, args.threads, args.version, args.ratio)
        convert('v1/test_public.json.gz', 'test_public.tsv', True, args.threads, args.version, args.ratio)

    else:
        convert('v2/train.json.gz', 'train.tsv', False, args.threads, args.version, args.ratio)
        convert('v2/dev.json.gz', 'dev.tsv', False, args.threads, args.version, args.ratio)
        convert('v2/test.json.gz', 'test.tsv', True, args.threads, args.version, args.ratio)
        convert('v2/test_public.json.gz', 'test_public.tsv', True, args.threads, args.version, args.ratio)

