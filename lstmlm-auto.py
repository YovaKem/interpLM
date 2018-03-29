from __future__ import print_function

from collections import defaultdict
import math
import random
import time

import dynet as dy

'''from https://github.com/clab/dynet/blob/master/examples/rnnlm/'''

# path to Mikolov PTB train.txt and valid.txt
FLAGS_train = 'data/train.en'
FLAGS_valid = 'data/valid.en'
FLAGS_layers = 2
FLAGS_hidden_dim = 512
FLAGS_batch_size = 30
FLAGS_word_dim = 64

def shuffled_infinite_list(lst):
  order = range(len(lst))
  while True:
    random.shuffle(order)
    for i in order:
      yield lst[i]


def read(filename, w2i, train):
  stop_symbol = w2i["</s>"]
  unk_symbol = w2i['_UNK']
  with open(filename, "r") as fh:
    for line in fh:
      sent = [w2i[x] if (train or x in w2i.keys()) else w2i['_UNK'] for x in line.strip()]
      sent.append(stop_symbol)
      yield sent


class LSTMLM:
  def __init__(self, model, vocab_size, start):
    self.start = start
    self.embeddings = model.add_lookup_parameters((vocab_size, FLAGS_word_dim))
    self.rnn = dy.VanillaLSTMBuilder(FLAGS_layers,
                                     FLAGS_word_dim,
                                     FLAGS_hidden_dim,
                                     model)

    self.h2l = model.add_parameters((vocab_size, FLAGS_hidden_dim))
    self.lb = model.add_parameters(vocab_size)

  # Compute the LM loss for a single sentence.
  def sent_lm_loss(self, sent):
    rnn_cur = self.rnn.initial_state()
    losses = []
    prev_word = self.start
    for word in sent:
      x_t = self.embeddings[prev_word]
      rnn_cur = rnn_cur.add_input(x_t)
      logits = dy.affine_transform([self.lb,
                                    self.h2l,
                                    rnn_cur.output()])
      losses.append(dy.pickneglogsoftmax(logits, word))
      prev_word = word
    return dy.esum(losses)

  # "Naively" computed Loss for a minibatch of sentences
  def minibatch_lm_loss(self, sents):
    sent_losses = [self.sent_lm_loss(sent) for sent in sents]
    minibatch_loss = dy.esum(sent_losses)
    total_words = sum(len(sent) for sent in sents)
    return minibatch_loss, total_words

print("RUN WITH AND WITHOUT --dynet_autobatch=1")
start = time.time()

w2i = defaultdict(lambda: len(w2i))
start_symbol = w2i["<s>"]

train = list(read(FLAGS_train, w2i, train=1))
vocab_size = len(w2i)
valid = list(read(FLAGS_valid, w2i, train=0))
assert vocab_size == len(w2i)  # Assert that vocab didn't grow.
pickle.dump(dict(w2i),open('data/w2i.en','wb'))

model = dy.Model()
trainer = dy.AdamTrainer(model)

lm = LSTMLM(model, vocab_size, start_symbol)

print("startup time: %r" % (time.time() - start))
start = time.time()
epoch = all_sents = dev_time = all_words = this_words = this_loss = 0
random_training_instance = shuffled_infinite_list(train)
while True:
  if updates % int(5000 / FLAGS_batch_size) == 0:
    trainer.status()
    train_time = time.time() - start - dev_time
    all_words += this_words
    print("loss=%.4f, words per second=%.4f" %
          (this_loss / this_words, all_words / train_time))
    this_loss = this_words = 0
  if updates % int(10000 / FLAGS_batch_size) == 0:
    dev_start = time.time()
    dev_loss = dev_words = 0
    for i in xrange(0, len(valid), FLAGS_batch_size):
      valid_minibatch = valid[i:i + FLAGS_batch_size]
      dy.renew_cg()  # Clear existing computation graph.
      loss_exp, mb_words = lm.minibatch_lm_loss(valid_minibatch)
      dev_loss += loss_exp.scalar_value()
      dev_words += mb_words
    print("nll=%.4f, ppl=%.4f, words=%r, time=%.4f, word_per_sec=%.4f" % (
        dev_loss / dev_words, math.exp(dev_loss / dev_words), dev_words,
        train_time, all_words / train_time))
    if dev_loss / dev_words >= prev_loss: no_improv+=1
    else:
        no_improv = 0
        model.save("models/bestmodel_epoch{}".format(epoch))
    if no_improv > 5:
        print('Ended trainign after 5 epochs of no improvement')
        break
    prev_loss = dev_loss / dev_words

    # Compute loss for one training minibatch.
  minibatch = [next(random_training_instance)
               for _ in xrange(FLAGS_batch_size)]
  dy.renew_cg()  # Clear existing computation graph.
  loss_exp, mb_words = lm.minibatch_lm_loss(minibatch)
  this_loss += loss_exp.scalar_value()
  this_words += mb_words
  all_sents += FLAGS_batch_size
  avg_minibatch_loss = loss_exp / len(minibatch)
  avg_minibatch_loss.forward()
  avg_minibatch_loss.backward()
  trainer.update()
  cur_epoch = int(all_sents / len(train))
  if cur_epoch != epoch:
    print("epoch %r finished" % cur_epoch)
    epoch = cur_epoch
