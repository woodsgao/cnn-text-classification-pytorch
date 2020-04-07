import argparse
import os
import torch
import torchtext.data as data
from model import CNN_Text
from mydatasets import MR
from train import predict
from flask import Flask


class Args(object):
    pass


app = Flask(__name__)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
text_field = data.Field(lower=True)
label_field = data.Field(sequential=False)
train_data, dev_data = MR.splits(text_field, label_field)
text_field.build_vocab(train_data, dev_data)
label_field.build_vocab(train_data, dev_data)

args = Args()
args.dropout = 0.5
args.max_norm = 3.0

args.embed_dim = 128
args.kernel_num = 100
args.kernel_sizes = '3,4,5'
args.static = False
args.snapshot = 'snapshot/best.pt'
args.embed_num = len(text_field.vocab)
args.class_num = len(label_field.vocab) - 1
args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]

model = CNN_Text(args)
model.load_state_dict(torch.load(args.snapshot, map_location='cpu'))
model = model.to(device)


@app.route('/cls/<text>')
def classify_text(text):
    app.logger.warning(text)
    result, conf = predict(text, model,
                           text_field, label_field, device)
    app.logger.warning(conf)
    return result
