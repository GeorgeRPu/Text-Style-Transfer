import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from InferSent.models import InferSent
from sklearn.neighbors import KNeighborsClassifier


# some hyperparameters
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VERSION = 2
WEIGHTS = './InferSent/encoder/infersent{}.pkl'.format(VERSION)
PARAMS = {'bsize': 1, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
          'pool_type': 'max', 'dpout_model': 0.0, 'version': VERSION}
if VERSION == 1:
    W2V = 'C:/users/georg/Desktop/GloVe/glove.840B.300d.txt'
else:
    W2V = 'C:/Users/georg/Desktop/fastText/crawl-300d-2M.vec'
VOCAB_SIZE = 100000
NUM_STEPS = 300


# set up model
model = InferSent(PARAMS).to(DEVICE)
model.load_state_dict(torch.load(WEIGHTS))
model.set_w2v_path(W2V)
word2vec = model.build_vocab_k_words(K=VOCAB_SIZE)


# setup the NN-classifer
vec2word = KNeighborsClassifier(n_neighbors=1)
vecs = []
words = []
for key, val in word2vec.items():
    if val.shape == (300,):
        vecs.append(val)
        words.append(key)
X = np.vstack(vecs)
y = np.array(words)
vec2word.fit(X, y)


# NOTE: stacked word vectors are len x 1 x vec dim
def getStack(sentence):
    vecs = []
    for word in sentence.split(' '):
        vec = torch.from_numpy(word2vec[word])
        vecs.append(vec.clone())
    return torch.stack(vecs)


def transfer(c_sent, s_sent):
    # NOTE: replace "," with " , " so split works
    c_stack = getStack(c_sent.replace(',', ' ,'))
    c_stack = c_stack.to(dtype=torch.float, device=DEVICE)
    c_in = c_stack.unsqueeze(1)
    c_len = c_stack.size(0)

    s_stack = getStack(s_sent.replace(',', ' ,'))
    s_stack = s_stack.to(dtype=torch.float, device=DEVICE)
    s_len = s_stack.size(0)

    f_stack = torch.randn(s_len, PARAMS['word_emb_dim'], device=DEVICE)
    f_stack.requires_grad = True
    f_in = f_stack.unsqueeze(1)
    f_len = f_stack.size(0)

    mse = nn.MSELoss()
    # LBGFS needs iterable of Tensors
    optimizer = optim.LBFGS(params=[f_stack])

    for i in range(NUM_STEPS):
        print('step', i)

        def closure():
            # NOTE: use stacked word vecs in place of feature maps
            with torch.no_grad():
                c_embed = model.embed([c_len], c_in)
                s_gram = torch.mm(s_stack, s_stack.t())
            f_embed = model.embed([f_len], f_in)
            f_gram = torch.mm(f_stack, f_stack.t())

            c_loss = mse(f_embed, c_embed)
            # NOTE: mse divides by num of elements, must divide again
            MN4 = f_gram.size(0) * f_gram.size(1) * 4
            s_loss = mse(f_gram, s_gram) / MN4
            reg = f_gram.mean()
            # print(c_loss, s_loss, reg)
            # NOTE: weights emprically determined
            loss = 10 * c_loss + s_loss + 1e-2 * reg

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(f_stack, 1)
            return loss

        optimizer.step(closure)

    # get words
    f_np = f_stack.squeeze().cpu().detach().numpy()
    f_sent = vec2word.predict(f_np)
    return ' '.join(f_sent)


if __name__ == '__main__':
    c_sent = "I love you"
    s_sent = "And God said let there be light"
    print(transfer(c_sent, s_sent))
