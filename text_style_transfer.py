import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from InferSent.models import InferSent
from sklearn.neighbors import KNeighborsClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set up model
VERSION = int(os.environ['VERSION'])
params = {
    'bsize': 1,
    'word_emb_dim': 300,
    'enc_lstm_dim': 2048,
    'pool_type': 'max',
    'dpout_model': 0.0,
    'version': VERSION
}
model = InferSent(params).to(device)
model.load_state_dict(torch.load(f'./InferSent/encoder/infersent{VERSION}.pkl'))

if VERSION == 1:
    w2v = 'InferSent/dataset/GloVe/glove.840B.300d.txt'
else:
    w2v = 'InferSent/dataset/fastText/crawl-300d-2M.vec.gz'
model.set_w2v_path(w2v)

VOCAB_SIZE = int(os.environ['VOCAB_SIZE'])
word2vec = model.build_vocab_k_words(VOCAB_SIZE)

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


def transfer(c_sent, s_sent, num_steps):
    # NOTE: replace "," with " , " so split works
    c_stack = getStack(c_sent.replace(',', ' ,'))
    c_stack = c_stack.to(dtype=torch.float, device=device)
    c_in = c_stack.unsqueeze(1)
    c_len = c_stack.size(0)

    s_stack = getStack(s_sent.replace(',', ' ,'))
    s_stack = s_stack.to(dtype=torch.float, device=device)
    s_len = s_stack.size(0)

    f_stack = torch.randn(s_len, params['word_emb_dim'], device=device)
    f_stack.requires_grad = True
    f_in = f_stack.unsqueeze(1)
    f_len = f_stack.size(0)

    mse = nn.MSELoss()
    # LBGFS needs iterable of Tensors
    optimizer = optim.LBFGS(params=[f_stack])

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
        loss = 10 * c_loss + s_loss + 1e-3 * reg

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(f_stack, 1)
        return loss

    for _ in tqdm.trange(num_steps):
        optimizer.step(closure)

    # get words
    f_np = f_stack.squeeze().cpu().detach().numpy()
    f_sent = vec2word.predict(f_np)
    return ' '.join(f_sent)


if __name__ == '__main__':
    print(transfer('To be or not to be', 'that is the question', 20))
