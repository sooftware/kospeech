"""
Copyright 2019-present NAVER Corp.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
     http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import sys
import argparse
import Levenshtein as Lev 

def edit_distance(ref, hyp):
    ref = ref.replace(' ', '') 
    hyp = hyp.replace(' ', '') 

    dist = Lev.distance(hyp, ref)
    length = len(ref.replace(' ', ''))

    return dist, length 

def load_ref(path):
    ref_dict = dict()
    with open(path, 'r') as f:
        for no, line in enumerate(f):
            key, target = line.strip().split(',')
            ref_dict[key] = target

    return ref_dict

def load_hyp(path):
    hyp_dict = dict()
    with open(path, 'r') as f:
        for no, line in enumerate(f):
            key, target = line.strip().split(',')
            key = key.split('.')[0] # remove file-extention 'wav'
            hyp_dict[key] = target

    return hyp_dict

def evaluation_metrics(hyp_path, ref_path):
    ref_dict = load_ref(ref_path)
    hyp_dict = load_hyp(hyp_path)

    dist_sum = 0
    leng_sum = 0

    for k, hyp in hyp_dict.items():

        k = k.split('/')[-1]

        hyp = hyp.replace(' ', '')
        ref = ref_dict[k].replace(' ', '')
        
        dist, leng = edit_distance(ref, hyp)

        dist_sum += dist
        leng_sum += leng

    return dist_sum / leng_sum

if __name__ == '__main__':

    args = argparse.ArgumentParser()
    args.add_argument('--prediction', type=str, default='pred.txt')
    config = args.parse_args()
   
    test_label_path = '/data/sr-hack-2019-dataset/test/test_label'

    CER = evaluation_metrics(config.prediction, test_label_path)
    CRR = (1.0 - CER) * 100.0
    print('%0.4f' % (CRR))
