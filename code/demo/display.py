# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT-joint baseline for NQ v1.0."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import gzip
import json
import os
import random
import re
import itertools

import enum
from bert import modeling
from bert import optimization
from bert import tokenization
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.contrib import cluster_resolver as contrib_cluster_resolver
from tensorflow.contrib import tpu as contrib_tpu


start_token = -10000
end_token = -10000
long_start_token = []
long_end_token = []
short_start_token = []
short_end_token = []

#with gzip.GzipFile(fileobj=tf.gfile.Open(input_path, "rb")) as input_file

f = open("predictions.json",)
 #input_paths = tf.gfile.Glob(input_pattern)

#f = gzip.GzipFile(fileobj=tf.gfile.Open("predictions.json",))
g = json.load(f)
ls = g["predictions"]

print(ls)

#with open("predictions.json", "rb") as input_file:
 #   e = json.loads(input_file)

idx = 0

ans = open("answers.txt", "a")

with gzip.GzipFile(fileobj=tf.gfile.Open("tiny-dev/nq-dev-sample.no-annot.jsonl.gz", "rb")) as input_file:
    for line in input_file:
        e = json.loads(line)

        l = []
        s = []
        long_ans = ''
        short_ans = ''

        long_start = ls[idx]["long_answer"]["start_token"]
        long_end = ls[idx]["long_answer"]["end_token"]
        short_start = ls[idx]["short_answers"][0]["start_token"]
        short_end = ls[idx]["short_answers"][0]["end_token"]

        i = 0
        for i in range(long_start, long_end):
            try:
                if(e["document_tokens"][i]["html_token"] == False):
                    l.append(e["document_tokens"][i]["token"])
            except:
                pass

        i = 0
        for i in range(short_start, short_end):
            try:
                if(e["document_tokens"][i]["html_token"] == False):
                    s.append(e["document_tokens"][i]["token"])
            except:
                pass


        for j in l:
            long_ans = long_ans + j + ' '
        for j in s:
            short_ans = short_ans + j + ' '



        #print(e["question_text"])
        #print("Long answer:")
        #print(long_ans)
        #print("Short answer:")
        #print(short_ans)
        #print("")

        ans.write(e["question_text"].encode("utf-8"))
        ans.write("\n")
        ans.write(e["document_url"].encode("utf-8"))
        ans.write("\n")
        ans.write("Long answer:".encode("utf-8"))
        ans.write("\n")
        ans.write(long_ans.encode("utf-8"))
        ans.write("\n")
        ans.write("Short answer:".encode("utf-8"))
        ans.write("\n")
        ans.write(short_ans.encode("utf-8"))
        ans.write("\n")
        ans.write("\n")



        idx = idx + 1

f.close()