import torch
import torch.nn as nn
import torch.nn.functional as F
import spacy
from torchtext.data.metrics import bleu_score

def save_checkpoint(state, filename="/content/drive/MyDrive/Models/MLT/my_checkpoint.pth.tar"):
  print("---->Saving checkpoint")
  torch.save(state,filename)

def load_checkpoint(checkpoint, model, optimizer):
  print("----->Loading checkpoint")
  model.load_state_dict(checkpoint["state_dict"])
  optimizer.load_state_dict(checkpoint["optimizer"])


def translate_sentence(model, sentence, german, english, device, max_length=50):
  #load german tokenizer
  spacy_ger = spacy.load("de_core_news_sm")
  #create tokens in spacy and convert everything into lower case
  if type(sentence) == str:
    tokens = [token.text.lower() for token in spacy_ger(sentence)]
  else:
    tokens = [token.lower() for token in sentence]

  #Add <SOS> and <EOS> token in beginning and end
  tokens.insert(0, german.init_token)
  tokens.append(german.eos_token)

  #convert text to indices---->Numericalize them
  text_to_indices = [german.vocab.stoi[tok] for tok in tokens]
  #convert to tensors
  sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)
  outputs = [english.vocab.stoi["<sos>"]]
  for i in range(max_length):
    trg_tensor = torch.LongTensor(outputs).unsqueeze(1).to(device)

    with torch.no_grad():
      output = model(sentence_tensor, trg_tensor)

    best_pred = output.argmax(2)[-1, :].item()
    outputs.append(best_pred)

    if best_pred == english.vocab.stoi["<eos>"]:
      break

  translated_sentence =  [english.vocab.itos[idx] for idx in outputs]
  #remove start token
  return translated_sentence[1:]

def bleu(data, model, german, english, device):
  target = []
  outputs = []

  for example in data:
    src = vars(example)["src"]
    trg = vars(example)["trg"]

    prediction = translate_sentence(model,src,german,english,device)
    prediction = prediction[:-1] #remove <eos> token

    target.append([trg])
    outputs.append(prediction)

  return bleu_score(outputs, targets)