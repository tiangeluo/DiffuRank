# ==============================================================================
# Copyright (c) 2023 Tiange Luo, tiange.cs@gmail.com
# Last modified: May 28, 2024
#
# This code is licensed under the MIT License.
# ==============================================================================

import os
from IPython import embed
import csv
import argparse
import pandas as pd
import tqdm
import spacy
from spacy.matcher import Matcher

parser = argparse.ArgumentParser()
parser.add_argument('--input_csv', type=str, default='path_to_csv')
parser.add_argument('--output_csv', type=str, default='path_to_output_csv')
args = parser.parse_args()

def remove_3d_phrases(text):
    text=text.replace('\'','\"')
    nlp = spacy.load('en_core_web_sm')
    
    # Initialize the matcher with the shared vocab
    matcher = Matcher(nlp.vocab)

    # Define the patterns
    pattern1 = [{"LOWER": "\"", "OP": "?"}, {"LOWER": "a", "OP": "?"}, {"LOWER": "white", "OP": "?"}, {"LOWER": "3d"}, {"ORTH": "-", "OP": "?"}, {"LOWER": "white", "OP": "?"}, {"LOWER": {"IN": ["rendering", "model", "object", "models", "scene", "printed", "rendered"]}, "OP": "?"}, {"LOWER": {"IN": ["of", "featuring", "resembling"]}, "OP": "?"}, {"LOWER": ":", "OP": "?"}, {"IS_PUNCT": True, "OP": "?"}]
    pattern2 = [{"LOWER": "with", "OP": "?"}, {"LOWER": "a", "OP": "?"}, {"LOWER": "3d"}, {"ORTH": "-", "OP": "?"}, {"LOWER": {"IN": ["models", "model", "rendered", "object", "models", "scene", "printed", "rendering", "modeled", "modeling"]}, "OP": "?"}, {"LOWER": {"IN": ["of", "featuring", "and"]}, "OP": "?"}]
    pattern3 = [{"LOWER": {"IN": ["in", "for", "featuring"]}, "OP": "?"}, {"LOWER": "a", "OP": "?"}, {"LOWER": "(", "OP": "?"},{"LOWER": "3d"}, {"ORTH": "-", "OP": "?"}, {"LOWER": {"IN": ["rendering", "model", "setting", "object", "models", "rendered", "printing", "printer"]}, "OP": "?"}, {"LOWER": ")", "OP": "?"}, {"LOWER": "inside", "OP": "?"}, {"IS_PUNCT": True, "OP": "?"}]

    # Add patterns to the matcher
    matcher.add("PRE_PATTERN1", [pattern1])
    matcher.add("MID_PATTERN", [pattern2])
    matcher.add("END_PATTERN", [pattern3])


    doc = nlp(text)
    matches = matcher(doc)
    spans_pre=[]
    spans_mid=[]
    spans_end=[]
    for match_id, start, end in matches:
        span = doc[start:end]  # The matched span
        if (start == 0 and (match_id == nlp.vocab.strings["PRE_PATTERN1"])):
            spans_pre.append((start, end, match_id))
        elif (end == len(doc) and match_id == nlp.vocab.strings["END_PATTERN"]):
            spans_end.append((start, end, match_id))
        elif start != 0 and end != len(doc) and match_id == nlp.vocab.strings["MID_PATTERN"]:
            spans_mid.append((start, end, match_id))
    spans_pre.sort(key=lambda x: x[1] - x[0], reverse=True)
    spans_end.sort(key=lambda x: x[1] - x[0], reverse=True)
    if spans_pre:
        longest_span_pre = spans_pre[0]
        span = doc[longest_span_pre[0]:longest_span_pre[1]]
        if text.replace(span.text+' ', "") == text:
            text = text.replace(span.text, "")
        else:
            text = text.replace(span.text+' ', "")
        if 'white' in span.text or 'White' in span.text:
            try:
                if text[0] == ('a'):
                    a_index = text.index('a ')
                    text = text[:a_index+2] + 'white' + ' ' + text[a_index+2:]
                elif text[0] == 'A':
                    a_index = text.index('A ')
                    text = text[:a_index+2] + 'white' + ' ' + text[a_index+2:]
                else:
                    text = 'white ' + text
            except:
                text = 'white ' + text
        if '\"' in span.text:
            text = '\"' + text

    if spans_end:
        longest_span_end = spans_end[0]
        span = doc[longest_span_end[0]:longest_span_end[1]]
        if text.replace(' '+ span.text, "") == text:
            text = text.replace( span.text, "")
        else:
            text = text.replace(' '+ span.text, "")
    
    spans = []
    for i in spans_mid:
        spans.append((i[0],i[1]))
    highest_spans = {}
    for span in spans:
        # If the first element of the span is not yet in the dictionary
        # or if the second element of the span is higher than the current highest
        if span[0] not in highest_spans or span[1] > highest_spans[span[0]]:
            # Set the second element of the span as the new highest
            highest_spans[span[0]] = span[1]
    highest_spans = [(k, v) for k, v in highest_spans.items()]

    for longest_span_mid in highest_spans:
        if spans_pre:
            if not (longest_span_mid[0] >= longest_span_pre[0] and longest_span_mid[1] <= longest_span_pre[1]):
                span = doc[longest_span_mid[0]:longest_span_mid[1]]
                if text.replace(' '+span.text, "") == text:
                    text = text.replace(span.text, "")
                else:
                    text = text.replace(' '+span.text, "")
        if spans_end:
            if not (longest_span_mid[0] >= longest_span_end[0] and longest_span_mid[1] <= longest_span_end[1]):
                span = doc[longest_span_mid[0]:longest_span_mid[1]]
                if text.replace(' '+span.text, "") == text:
                    text = text.replace(span.text, "")
                else:
                    text = text.replace(' '+span.text, "")
        if not spans_pre and not spans_end:
            span = doc[longest_span_mid[0]:longest_span_mid[1]]
            if text.replace(' '+span.text, "") == text:
                text = text.replace(span.text, "")
            else:
                text = text.replace(' '+span.text, "")
    
    text=text.replace('\"','\'')
    if text[-7:] == " in 3D.":
        text=text[:-7]
    if text[:7] == 'white a':
        text = text.replace('white a','a white')
    if text[:8] == 'white an':
        text = text.replace('white an','an white')

    return text

cur_caption_csv = pd.read_csv(args.input_csv, header=None)
uids = list(set(cur_caption_csv[0].values))

n2idx = {}
for i in range(len(cur_caption_csv)):
    n2idx[cur_caption_csv[0][i]] = i

f = open(args.output_csv, 'a')
writer = csv.writer(f)
output = []
print('############begin remove 3D-related words############')
for cur_uid in tqdm.tqdm(uids):
    out_idx = cur_caption_csv[0][n2idx[cur_uid]]
    cur_final_caption = cur_caption_csv[1][n2idx[cur_uid]]

    if '3D' not in cur_final_caption and '3d' not in cur_final_caption:
        summary = cur_final_caption
    else:
        summary = remove_3d_phrases(cur_final_caption)

    writer.writerow([out_idx, summary.replace('"', '')])

f.flush()
os.fsync(f.fileno())
f.close()


