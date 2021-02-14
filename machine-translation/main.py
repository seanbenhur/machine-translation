
import torch
import torch.nn as nn
import torch.optim as optim
import spacy
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
from model import Transformer
from utils import *
from argparse import Namespace
from train import Trainer
from pathlib import Path
from config import *
# Specify arguments
args = Namespace(
    # Training hyperparameters
    num_epochs = num_epochs,
    learning_rate = learning_rate,
    batch_size = batch_size,
    patience = patience,
    src_vocab_size = 7855,
    trg_vocab_size = 5893,  
    embedding_size = embedding_size,
    num_heads = num_heads,
    num_encoder_layers = num_encoder_layers,
    num_decoder_layers = num_decoder_layers,
    dropout = dropout,
    max_len = max_len,
    forward_expansion = forward_expansion,
)



spacy_ger = spacy.load("de_core_news_sm")
spacy_eng = spacy.load("en")

def tokenize_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]


def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]


german = Field(tokenize=tokenize_ger, lower=True, init_token="<sos>", eos_token="<eos>")

english = Field(
    tokenize=tokenize_eng, lower=True, init_token="<sos>", eos_token="<eos>"
)

train_data, valid_data, test_data = Multi30k.splits(
    exts=(".de", ".en"), fields=(german, english)
)

german.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=10000, min_freq=2)

# We're ready to define everything we need for training our Seq2Seq model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

src_pad_idx = english.vocab.stoi["<pad>"]



train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=args.batch_size,
    sort_within_batch=True,
    sort_key=lambda x: len(x.src),
    device=device,
)

model = Transformer(
    args.embedding_size,
    args.src_vocab_size,
    args.trg_vocab_size,
    src_pad_idx,
    args.num_heads,
    args.num_encoder_layers,
    args.num_decoder_layers,
    args.forward_expansion,
    args.dropout,
    args.max_len,
    device,
).to(device)

optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.1, patience=10, verbose=True
)

pad_idx = english.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)


# Trainer module
trainer = Trainer(
    model=model, device=device, loss_fn=criterion,
    optimizer=optimizer,scheduler=None)

best_model, best_val_loss = trainer.train(
    args.num_epochs, args.patience, train_iterator, valid_iterator)