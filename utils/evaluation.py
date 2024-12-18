# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import string
from collections import Counter
from typing import Callable

import numpy as np
import regex
# from rouge import Rouge
import evaluate

from rouge_chinese import Rouge
import jieba
rouge = Rouge()
# bertscore = evaluate.load("evaluate-main/metrics/bertscore") #, module_type="metric")

# logger = logging.getLogger(__name__)

# Normalization and score functions from SQuAD evaluation script https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return regex.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em(prediction, ground_truth, normalize_fn):
    return float(normalize_fn(prediction) == normalize_fn(ground_truth))


def f1(prediction, ground_truth, normalize_fn):
    prediction_tokens = normalize_fn(prediction).split()
    ground_truth_tokens = normalize_fn(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    # print(f'====> Inner f1,prediction = {prediction}, ground_truth={ground_truth} score = {f1}')
    return f1

def f1_zh(prediction, ground_truth, normalize_fn):
    prediction_tokens = ' '.join(jieba.cut(normalize_fn(prediction)))
    ground_truth_tokens = ' '.join(jieba.cut(normalize_fn(ground_truth)))
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def rouge_wrapper(prediction, ground_truth):
    # try:
    result = rouge.get_scores(prediction, ground_truth, avg=True)
    # print(f'====> Inner rouge_wrapper prediction = {prediction}, ground_truth={ground_truth} score = {result}')
    return result["rouge-1"]["f"], result["rouge-2"]["f"], result["rouge-l"]["f"]
    # except:
    #     return 0.0, 0.0, 0.0

def rouge_wrapper_zh(prediction, ground_truth):
    try:
        # print('In rouge_wrapper_zh jieba.cut(prediction) = ', ' '.join(jieba.cut(prediction)))
        # print('In rouge_wrapper_zh jieba.cut(ground_truth) = ', ' '.join(jieba.cut(ground_truth)))
        result = rouge.get_scores(' '.join(jieba.cut(prediction)), ' '.join(jieba.cut(ground_truth)), avg=True)
        # print('In rouge_wrapper_zh rouge = ', result)
        return result["rouge-1"]["f"], result["rouge-2"]["f"], result["rouge-l"]["f"]
    except:
        return 0.0, 0.0, 0.0


def f1_score(prediction, ground_truths, normalize_fn: Callable[[str], str] = lambda x: x):
    return max([f1_zh(prediction, gt, normalize_fn) for gt in ground_truths])


def exact_match_score(prediction, ground_truths, normalize_fn: Callable[[str], str] = lambda x: x):
    return np.mean([em(prediction, gt, normalize_fn) for gt in ground_truths])


def rouge_score(prediction, ground_truths, normalize_fn: Callable[[str], str] = lambda x: x):
    ground_truths = [x for x in ground_truths if len(x) > 0]
    if (
        len(prediction) == 0 or len(ground_truths) == 0
    ):  # check if empty prediction or if there is no hypothesis with len > 0
        return 0.0, 0.0, 0.0
    scores = [rouge_wrapper_zh(prediction, gt) for gt in ground_truths]
    # print(f'====> Inner rouge_score, scores = {scores}')
    rouge1 = max(s[0] for s in scores)
    rouge2 = max(s[1] for s in scores)
    rougel = max(s[2] for s in scores)
    return rouge1, rouge2, rougel


def get_bert_score(bertscore, predictions, references):
    # print(f'references = {references[:3]}')
    # print(f'predictions = {predictions[:3]}')
    res = []
    for ref in references:
        results = bertscore.compute(predictions=predictions, references=[ref], lang="en")["f1"]
        res.append(results)
    # results = bertscore.compute(predictions=predictions, references=references, lang="en")
    return np.mean(res)

def bert_score(prediction, ground_truths, normalize_fn: Callable[[str], str] = lambda x: x):
    
    return get_bert_score(bertscore, [prediction], ground_truths) 

        