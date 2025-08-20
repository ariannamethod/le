"""
Train and sample from character-level language models using a single script.
The program reads a text file with one item per line and learns to generate
similar text through an autoregressive Transformer-based neural network.
"""

import os
import sys
import time
import math
import argparse
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from memory import Memory
import metrics
import response_log

# force CPU execution
DEVICE = torch.device('cpu')
torch.set_num_threads(4)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

QUIET = False


def qprint(*args, **kwargs):
    if not QUIET:
        print(*args, **kwargs)

# -----------------------------------------------------------------------------

@dataclass
class ModelConfig:
    block_size: int = None # length of the input sequences of integers
    vocab_size: int = None # the input integers are in range [0 .. vocab_size -1]
    # parameters below control the sizes of each model slightly differently
    n_layer: int = 4
    n_embd: int = 64
    n_embd2: int = 64
    n_head: int = 4

# -----------------------------------------------------------------------------
# Transformer Language Model (*exactly* as used in GPT-2)

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
            act     = NewGELU(),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.c_proj(m.act(m.c_fc(x))) # MLP forward

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x

class Transformer(nn.Module):
    """ Transformer Language Model, exactly as seen in GPT-2 """

    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        qprint("number of parameters: %.2fM" % (n_params/1e6,))

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

# -----------------------------------------------------------------------------
# Bag of Words (BoW) language model

class CausalBoW(nn.Module):
    """
    Causal bag of words. Averages the preceding elements and looks suspiciously like
    a CausalAttention module you'd find in a transformer, for no apparent reason at all ;)
    """
    def __init__(self, config):
        super().__init__()

        # used to mask out vectors and preserve autoregressive property
        self.block_size = config.block_size
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                            .view(1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, n_embd

        # do the weighted average of all preceeding token features
        att = torch.zeros((B, T, T), device=x.device)
        att = att.masked_fill(self.bias[:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ x # (B, T, T) x (B, T, C) -> (B, T, C)

        return y

class BoWBlock(nn.Module):
    """ collects BoW features and adds an MLP """

    def __init__(self, config):
        super().__init__()

        # Causal BoW module
        self.cbow = CausalBoW(config)
        # MLP assembler
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, config.n_embd2),
            c_proj  = nn.Linear(config.n_embd2, config.n_embd),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.c_proj(F.tanh(m.c_fc(x))) # MLP forward

    def forward(self, x):
        x = x + self.cbow(x)
        x = x + self.mlpf(x)
        return x

class BoW(nn.Module):
    """
    takes the previous block_size tokens, encodes them with a lookup table,
    also encodes their positions with lookup table, then averages all of those
    embeddings up and uses that to predict the next token.
    """

    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size
        # token embedding
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        # position embedding
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        # context block
        self.context_block = BoWBlock(config)
        # language model head decoder layer
        self.lm_head = nn.Linear(config.n_embd, self.vocab_size)

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):

        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the token and position embedding layers
        tok_emb = self.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.wpe(pos) # position embeddings of shape (1, t, n_embd)
        # add and run through the decoder MLP
        x = tok_emb + pos_emb
        # run the bag of words context module
        x = self.context_block(x)
        # decode to next token probability
        logits = self.lm_head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

# -----------------------------------------------------------------------------
"""
Recurrent Neural Net language model: either a vanilla RNN recurrence or a GRU.
Did not implement an LSTM because its API is a bit more annoying as it has
both a hidden state and a cell state, but it's very similar to GRU and in
practice works just as well.
"""

class RNNCell(nn.Module):
    """
    the job of a 'Cell' is to:
    take input at current time step x_{t} and the hidden state at the
    previous time step h_{t-1} and return the resulting hidden state
    h_{t} at the current timestep
    """
    def __init__(self, config):
        super().__init__()
        self.xh_to_h = nn.Linear(config.n_embd + config.n_embd2, config.n_embd2)

    def forward(self, xt, hprev):
        xh = torch.cat([xt, hprev], dim=1)
        ht = F.tanh(self.xh_to_h(xh))
        return ht

class GRUCell(nn.Module):
    """
    same job as RNN cell, but a bit more complicated recurrence formula
    that makes the GRU more expressive and easier to optimize.
    """
    def __init__(self, config):
        super().__init__()
        # input, forget, output, gate
        self.xh_to_z = nn.Linear(config.n_embd + config.n_embd2, config.n_embd2)
        self.xh_to_r = nn.Linear(config.n_embd + config.n_embd2, config.n_embd2)
        self.xh_to_hbar = nn.Linear(config.n_embd + config.n_embd2, config.n_embd2)

    def forward(self, xt, hprev):
        # first use the reset gate to wipe some channels of the hidden state to zero
        xh = torch.cat([xt, hprev], dim=1)
        r = F.sigmoid(self.xh_to_r(xh))
        hprev_reset = r * hprev
        # calculate the candidate new hidden state hbar
        xhr = torch.cat([xt, hprev_reset], dim=1)
        hbar = F.tanh(self.xh_to_hbar(xhr))
        # calculate the switch gate that determines if each channel should be updated at all
        z = F.sigmoid(self.xh_to_z(xh))
        # blend the previous hidden state and the new candidate hidden state
        ht = (1 - z) * hprev + z * hbar
        return ht

class RNN(nn.Module):

    def __init__(self, config, cell_type):
        super().__init__()
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size
        self.start = nn.Parameter(torch.zeros(1, config.n_embd2)) # the starting hidden state
        self.wte = nn.Embedding(config.vocab_size, config.n_embd) # token embeddings table
        if cell_type == 'rnn':
            self.cell = RNNCell(config)
        elif cell_type == 'gru':
            self.cell = GRUCell(config)
        self.lm_head = nn.Linear(config.n_embd2, self.vocab_size)

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        b, t = idx.size()

        # embed all the integers up front and all at once for efficiency
        emb = self.wte(idx) # (b, t, n_embd)

        # sequentially iterate over the inputs and update the RNN state each tick
        hprev = self.start.expand((b, -1)) # expand out the batch dimension
        hiddens = []
        for i in range(t):
            xt = emb[:, i, :] # (b, n_embd)
            ht = self.cell(xt, hprev) # (b, n_embd2)
            hprev = ht
            hiddens.append(ht)

        # decode the outputs
        hidden = torch.stack(hiddens, 1) # (b, t, n_embd2)
        logits = self.lm_head(hidden)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss
        # -----------------------------------------------------------------------------
# MLP language model

class MLP(nn.Module):
    """
    takes the previous block_size tokens, encodes them with a lookup table,
    concatenates the vectors and predicts the next token with an MLP.

    Reference:
    Bengio et al. 2003 https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf
    """

    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size
        self.wte = nn.Embedding(config.vocab_size + 1, config.n_embd) # token embeddings table
        # +1 in the line above for a special <BLANK> token that gets inserted if encoding a token
        # before the beginning of the input sequence
        self.mlp = nn.Sequential(
            nn.Linear(self.block_size * config.n_embd, config.n_embd2),
            nn.Tanh(),
            nn.Linear(config.n_embd2, self.vocab_size)
        )

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):

        # gather the word embeddings of the previous 3 words
        embs = []
        for k in range(self.block_size):
            tok_emb = self.wte(idx) # token embeddings of shape (b, t, n_embd)
            idx = torch.roll(idx, 1, 1)
            idx[:, 0] = self.vocab_size # special <BLANK> token
            embs.append(tok_emb)

        # concat all of the embeddings together and pass through an MLP
        x = torch.cat(embs, -1) # (b, t, n_embd * block_size)
        logits = self.mlp(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

# -----------------------------------------------------------------------------
# Bigram language model

class Bigram(nn.Module):
    """
    Bigram Language Model 'neural net', simply a lookup table of logits for the
    next character given a previous character.
    """

    def __init__(self, config):
        super().__init__()
        n = config.vocab_size
        self.logits = nn.Parameter(torch.zeros((n, n)))

    def get_block_size(self):
        return 1 # this model only needs one previous character to predict the next

    def forward(self, idx, targets=None):

         # 'forward pass', lol
        logits = self.logits[idx]

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

# -----------------------------------------------------------------------------
# helper functions for evaluating and sampling from the model

@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None, top_p=None):
    """
    Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    """
    block_size = model.get_block_size()
    for _ in range(max_new_tokens):
        # if the sequence context is growing too long we must crop it at block_size
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        # forward the model to get the logits for the index in the sequence
        logits, _ = model(idx_cond)
        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop the logits to only the top k or nucleus top_p options
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('Inf')
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            for i in range(logits.size(0)):
                indices = sorted_indices[i][sorted_indices_to_remove[i]]
                logits[i, indices] = -float('Inf')
        # apply softmax to convert logits to (normalized) probabilities
        probs = F.softmax(logits, dim=-1)
        # either sample from the distribution or take the most likely element
        if do_sample:
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            _, idx_next = torch.topk(probs, k=1, dim=-1)
        # append sampled index to the running sequence and continue
        idx = torch.cat((idx, idx_next), dim=1)

    return idx

def sample_prompt(prompt: str, model, dataset, memory: Memory, *, max_new_tokens: int = 15, temperature: float = 0.8, top_k: int | None = 40, top_p: float | None = 0.9) -> str:
    """Generate text conditioned on a prompt and conversation history.

    The ``prompt`` is tokenized, the token with the highest information gain
    (lowest probability given preceding context) is selected as the "charged"
    word, and generation is seeded with this token. Previous conversation
    history retrieved from ``memory`` is prepended to provide additional
    context. Nucleus (``top_p``) or top-k sampling is used during generation to
    avoid verbatim dataset quotes.

    The returned string always begins with a capital letter and ends with a
    period.
    """

    def _encode(text: str) -> torch.Tensor:
        return torch.tensor([dataset.stoi[ch] for ch in text if ch in dataset.stoi], dtype=torch.long)
    
    def _encode_word(word: str) -> torch.Tensor:
        """Encode a single word as character sequence"""
        return torch.tensor([dataset.stoi[ch] for ch in word if ch in dataset.stoi], dtype=torch.long)

    # Получаем историю сообщений из памяти
    try:
        messages = memory.get_messages()
        memory_tokens = _encode(" ".join(messages)) if messages else _encode("")
    except Exception:
        memory_tokens = _encode("")
    # Токенизируем текущий промпт
    prompt_tokens = _encode(prompt)
    start_tok = torch.tensor([0], dtype=torch.long)
    block_size = model.get_block_size()
    max_memory = block_size - len(prompt_tokens) - 1
    if max_memory > 0 and len(memory_tokens) > max_memory:
        memory_tokens = memory_tokens[-max_memory:]
    context_for_charge = torch.cat((start_tok, memory_tokens, prompt_tokens), dim=0)
    
    # По умолчанию используем первый токен промпта или 0, если промпт пустой
    charged_token = prompt_tokens[0] if len(prompt_tokens) > 0 else torch.tensor(0)

    # Находим самое "заряженное" СЛОВО (не символ!)
    words = prompt.strip().split()
    charged_word = ""
    
    if words and context_for_charge.numel() > 1:
        word_scores = []
        
        # Для каждого слова вычисляем его "заряженность"
        for word in words:
            word_tokens = _encode_word(word.lower())
            if len(word_tokens) == 0:
                word_scores.append((word, 0.0))
                continue
                
            # Создаем контекст для этого слова
            word_context = torch.cat((context_for_charge, word_tokens), dim=0)
            if word_context.size(0) > block_size:
                word_context = word_context[-block_size:]
            
            # Вычисляем вероятность слова в контексте
            try:
                logits, _ = model(word_context[:-len(word_tokens)].unsqueeze(0).to(DEVICE))
                probs = F.softmax(logits, dim=-1)[0]
                
                # Средняя вероятность символов слова
                word_prob = 1.0
                for i, token in enumerate(word_tokens):
                    if i < probs.size(0):
                        word_prob *= probs[i, token].item()
                
                # Чем меньше вероятность, тем больше "заряженность"
                word_scores.append((word, -word_prob))  # отрицательная для сортировки
            except:
                word_scores.append((word, 0.0))
        
        # Выбираем самое заряженное слово
        if word_scores:
            charged_word = max(word_scores, key=lambda x: x[1])[0]
            print(f"DEBUG: Заряженное слово: '{charged_word}' из {words}")
    
    # Если не нашли заряженное слово, берем первое
    if not charged_word and words:
        charged_word = words[0]

    # Теперь генерируем текст, начиная с заряженного слова
    def _generate_once() -> str:
        # Кодируем заряженное слово как последовательность символов
        if charged_word:
            charged_word_tokens = _encode_word(charged_word.lower())
        else:
            charged_word_tokens = torch.tensor([charged_token.item()], dtype=torch.long)
        
        # Начинаем с контекста + заряженного слова
        idx_context = torch.cat((start_tok, memory_tokens, charged_word_tokens), dim=0)
        if idx_context.size(0) > block_size:
            idx_context = idx_context[-block_size:]
        idx = idx_context.unsqueeze(0).to(DEVICE)
        
        # Генерируем продолжение
        out = generate(
            model,
            idx,
            max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_k=top_k,
            top_p=top_p,
        )
        
        # Извлекаем сгенерированные токены (продолжение после заряженного слова)
        gen_tokens = out[0, idx.size(1):].tolist()
        if 0 in gen_tokens:
            gen_tokens = gen_tokens[:gen_tokens.index(0)]
        gen_tokens = [t for t in gen_tokens if t != 0]
        
        # Декодируем продолжение
        continuation = dataset.decode(gen_tokens) if gen_tokens else ""
        
        # Формируем финальный текст: заряженное слово + продолжение
        if charged_word:
            text = charged_word + continuation
        else:
            # Если нет заряженного слова, используем только продолжение
            text = continuation
        
        text = text.strip()
        
        # Обеспечиваем, что предложение начинается с заглавной буквы и заканчивается точкой
        if text:
            text = text[0].upper() + text[1:]
        if not text.endswith('.'):
            text += '.'
        
        return text

    # ВСЕГДА генерируем что-то, НИКАКИХ ШАБЛОНОВ!
    for _ in range(5):
        text = _generate_once()
        if response_log.check_and_log(text):
            return text
    
    # Если все повторы - возвращаем хотя бы последний сгенерированный
    return _generate_once()

def print_samples(num=20, return_samples=False):
    """Samples from the model and optionally returns the decoded samples.

    When ``return_samples`` is True the function returns a list of samples and
    suppresses all printing. Otherwise it behaves as before and prints a nice
    summary.
    """
    X_init = torch.zeros(num, 1, dtype=torch.long).to(DEVICE)
    top_k = args.top_k if args.top_k != -1 else None
    steps = train_dataset.get_output_length() - 1  # -1 because we already start with <START> token (index 0)
    X_samp = generate(
        model,
        X_init,
        steps,
        temperature=args.temperature,
        top_k=top_k,
        do_sample=True,
    ).to('cpu')
    train_samples, test_samples, new_samples = [], [], []
    samples = []
    for i in range(X_samp.size(0)):
        # get the i'th row of sampled integers, as python list
        row = X_samp[i, 1:].tolist()  # note: we need to crop out the first <START> token
        # token 0 is the <STOP> token, so we crop the output sequence at that point
        crop_index = row.index(0) if 0 in row else len(row)
        row = row[:crop_index]
        word_samp = train_dataset.decode(row)
        metrics.log_response_metrics(word_samp, i)
        samples.append(word_samp)
        # separately track samples that we have and have not seen before
        if train_dataset.contains(word_samp):
            train_samples.append(word_samp)
        elif test_dataset.contains(word_samp):
            test_samples.append(word_samp)
        else:
            new_samples.append(word_samp)
    if return_samples:
        return samples
    qprint('-'*80)
    for lst, desc in [(train_samples, 'in train'), (test_samples, 'in test'), (new_samples, 'new')]:
        qprint(f"{len(lst)} samples that are {desc}:")
        for word in lst:
            qprint(word)
    qprint('-'*80)

@torch.inference_mode()
def evaluate(model, dataset, batch_size=50, max_batches=None):
    model.eval()
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=0)
    losses = []
    for i, batch in enumerate(loader):
        batch = [t.to(DEVICE) for t in batch]
        X, Y = batch
        logits, loss = model(X, Y)
        losses.append(loss.item())
        if max_batches is not None and i >= max_batches:
            break
    mean_loss = torch.tensor(losses).mean().item()
    model.train() # reset model back to training mode
    return mean_loss

def chat(model, data_path, memory):
    """interactive loop that fine-tunes on the dataset and answers the user"""
    os.makedirs('logs', exist_ok=True)
    log_path = os.path.join('logs', 'leconvo.txt')
    res_step = 0
    while True:
        train_dataset, _ = create_datasets(data_path)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        loader = InfiniteDataLoader(train_dataset, batch_size=32, num_workers=0)
        try:
            user = input('you: ')
        except EOFError:
            break
        if user is None or user.strip() == '':
            break
        for _ in range(20):
            X, Y = [t.to(DEVICE) for t in loader.next()]
            logits, loss = model(X, Y)
            model.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        # Используем sample_prompt для поиска заряженного слова и генерации ответа
        response = sample_prompt(user, model, train_dataset, memory)
        print(f'le: {response}')
        metrics.log_response_metrics(response, res_step)
        memory.save_conversation(user, response)
        resonance = metrics.compute_resonance(model, train_dataset, user, response)
        metrics.log_resonance(resonance, res_step)
        res_step += 1
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f'User: {user}\nLE: {response}\n')

# -----------------------------------------------------------------------------
# helper functions for creating the training and test Datasets that emit words

class CharDataset(Dataset):

    def __init__(self, words, chars, max_word_length):
        self.words = words
        self.chars = chars
        self.max_word_length = max_word_length
        self.stoi = {ch:i+1 for i,ch in enumerate(chars)}
        self.itos = {i:s for s,i in self.stoi.items()} # inverse mapping

    def __len__(self):
        return len(self.words)

    def contains(self, word):
        return word in self.words

    def get_vocab_size(self):
        return len(self.chars) + 1 # all the possible characters and special 0 token

    def get_output_length(self):
        return self.max_word_length + 1 # <START> token followed by words

    def encode(self, word):
        ix = torch.tensor([self.stoi[w] for w in word], dtype=torch.long)
        return ix

    def decode(self, ix):
        word = ''.join(self.itos[i] for i in ix)
        return word

    def __getitem__(self, idx):
        word = self.words[idx]
        ix = self.encode(word)
        x = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        y = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        x[1:1+len(ix)] = ix
        y[:len(ix)] = ix
        y[len(ix)+1:] = -1 # index -1 will mask the loss at the inactive locations
        return x, y

def create_datasets(input_path):

    # gather all lines from every file in the directory of input_path
    if os.path.isdir(input_path):
        files = [os.path.join(input_path, f) for f in sorted(os.listdir(input_path))]
    else:
        directory = os.path.dirname(input_path) or '.'
        files = [os.path.join(directory, f) for f in sorted(os.listdir(directory))]
    words = []
    for fname in files:
        if os.path.isfile(fname):
            with open(fname, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        # Разбиваем строку на слова, но сохраняем некоторые символы для творческого хаоса
                        line_words = line.split()
                        for word in line_words:
                            # Очищаем от большинства символов, но оставляем некоторые для "безумия"
                            cleaned_word = ''.join(c for c in word if c.isalnum() or c in ".,!?'-")
                            if cleaned_word and len(cleaned_word) > 1:  # Только слова длиннее 1 символа
                                words.append(cleaned_word)
    chars = sorted(list(set(''.join(words)))) # all the possible characters
    max_word_length = max(len(w) for w in words) if words else 0
    qprint(f"number of examples in the dataset: {len(words)}")
    qprint(f"max word length: {max_word_length}")
    qprint(f"number of unique characters in the vocabulary: {len(chars)}")
    qprint("vocabulary:")
    qprint(''.join(chars))

    # partition the input data into a training and the test set
    test_set_size = min(1000, int(len(words) * 0.1)) # 10% of the training set, or up to 1000 examples
    rp = torch.randperm(len(words)).tolist()
    train_words = [words[i] for i in rp[:-test_set_size]]
    test_words = [words[i] for i in rp[-test_set_size:]]
    qprint(f"split up the dataset into {len(train_words)} training examples and {len(test_words)} test examples")

    # wrap in dataset objects
    train_dataset = CharDataset(train_words, chars, max_word_length)
    test_dataset = CharDataset(test_words, chars, max_word_length)

    return train_dataset, test_dataset

class InfiniteDataLoader:
    """
    this is really hacky and I'm not proud of it, but there doesn't seem to be
    a better way in PyTorch to just create an infinite dataloader?
    """

    def __init__(self, dataset, **kwargs):
        train_sampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=int(1e10))
        self.train_loader = DataLoader(dataset, sampler=train_sampler, **kwargs)
        self.data_iter = iter(self.train_loader)

    def next(self):
        try:
            batch = next(self.data_iter)
        except StopIteration: # this will technically only happen after 1e10 samples... (i.e. basically never)
            self.data_iter = iter(self.train_loader)
            batch = next(self.data_iter)
        return batch

# -----------------------------------------------------------------------------
if __name__ == '__main__':

    # parse command line args
    parser = argparse.ArgumentParser(description="LE")
    # system/input/output
    parser.add_argument('--input-file', '-i', type=str, default='blood/lines01.txt', help="seed data file inside the blood directory")
    parser.add_argument('--work-dir', '-o', type=str, default='out', help="output working directory")
    parser.add_argument('--resume', action='store_true', help="when this flag is used, we will resume optimization from existing model in the workdir")
    parser.add_argument('--sample-only', action='store_true', help="just sample from the model and quit, don't train")
    parser.add_argument('--num-workers', '-n', type=int, default=4, help="number of data workers for both train/test")
    parser.add_argument('--max-steps', type=int, default=200, help="max number of optimization steps to run for, or -1 for infinite.")
    parser.add_argument('--seed', type=int, default=3407, help="seed")
    # sampling
    parser.add_argument('--num-samples', type=int, default=1, help="number of samples to draw when using --sample-only")
    parser.add_argument('--top-k', type=int, default=-1, help="top-k for sampling, -1 means no top-k")
    parser.add_argument('--temperature', type=float, default=1.0, help="temperature for sampling")
    parser.add_argument('--prompt', type=str, default=None, help="prompt to condition on when sampling")
    # model
    parser.add_argument('--type', type=str, default='transformer', help="model class type to use, bigram|mlp|rnn|gru|bow|transformer")
    parser.add_argument('--n-layer', type=int, default=4, help="number of layers")
    parser.add_argument('--n-head', type=int, default=4, help="number of heads (in a transformer)")
    parser.add_argument('--n-embd', type=int, default=64, help="number of feature channels in the model")
    parser.add_argument('--n-embd2', type=int, default=64, help="number of feature channels elsewhere in the model")
    # optimization
    parser.add_argument('--batch-size', '-b', type=int, default=32, help="batch size during optimization")
    parser.add_argument('--learning-rate', '-l', type=float, default=5e-4, help="learning rate")
    parser.add_argument('--weight-decay', '-w', type=float, default=0.01, help="weight decay")
    parser.add_argument('--quiet', action='store_true', help="suppress non-sample output when used with --sample-only")
    args = parser.parse_args()
    QUIET = args.quiet and args.sample_only
    qprint(vars(args))

    # system inits
    torch.manual_seed(args.seed)
    os.makedirs(args.work_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.work_dir)
    metrics.set_writer(writer)

    # init datasets
    train_dataset, test_dataset = create_datasets(args.input_file)
    vocab_size = train_dataset.get_vocab_size()
    block_size = train_dataset.get_output_length()
    qprint(f"dataset determined that: vocab_size={vocab_size}, block_size={block_size}")

    memory = Memory()
    data_hash = Memory.hash_file(args.input_file)
    stored_hash = memory.get_meta('data_hash')
    skip_training = stored_hash == data_hash

    # init model
    config = ModelConfig(vocab_size=vocab_size, block_size=block_size,
                       n_layer=args.n_layer, n_head=args.n_head,
                       n_embd=args.n_embd, n_embd2=args.n_embd2)
    if args.type == 'transformer':
        model = Transformer(config)
    elif args.type == 'bigram':
        model = Bigram(config)
    elif args.type == 'mlp':
        model = MLP(config)
    elif args.type == 'rnn':
        model = RNN(config, cell_type='rnn')
    elif args.type == 'gru':
        model = RNN(config, cell_type='gru')
    elif args.type == 'bow':
        model = BoW(config)
    else:
        raise ValueError(f'model type {args.type} is not recognized')
    model.to(DEVICE)
    qprint(f"model #params: {sum(p.numel() for p in model.parameters())}")
    model_path = os.path.join(args.work_dir, 'model.pt')
    if args.resume or args.sample_only or skip_training:
        if os.path.exists(model_path):
            qprint("resuming from existing model in the workdir")
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        else:
            qprint("no existing model found in the workdir")
    if args.sample_only:
        top_k = args.top_k if args.top_k != -1 else None
        if args.prompt:
            # Используем sample_prompt для поиска заряженного слова и генерации ответа
            sample = sample_prompt(args.prompt, model, train_dataset, memory, 
                                 temperature=args.temperature, top_k=top_k)
            print(sample)
        else:
            for sample in print_samples(num=args.num_samples, return_samples=True):
                print(sample)
        sys.exit()

    if skip_training:
        qprint('model already trained on this data hash; skipping training')
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.99),
            eps=1e-8,
        )

        batch_loader = InfiniteDataLoader(
            train_dataset,
            batch_size=args.batch_size,
            pin_memory=True,
            num_workers=args.num_workers,
        )

        best_loss = None
        step = 0
        epoch = 0
        while True:

            t0 = time.time()

            # get the next batch, ship to device, and unpack it to input and target
            batch = batch_loader.next()
            batch = [t.to(DEVICE) for t in batch]
            X, Y = batch

            # feed into the model
            logits, loss = model(X, Y)

            # calculate the gradient, update the weights
            model.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            t1 = time.time()

            # logging
            if step % 10 == 0:
                print(
                    f"step {step} | loss {loss.item():.4f} | step time {(t1-t0)*1000:.2f}ms"
                )

            # evaluate the model
            if step > 0 and step % 100 == 0:
                train_loss = evaluate(
                    model, train_dataset, batch_size=100, max_batches=10
                )
                test_loss = evaluate(
                    model, test_dataset, batch_size=100, max_batches=10
                )
                metrics.log_loss("train", train_loss, epoch)
                metrics.log_loss("test", test_loss, epoch)
                print(
                    f"step {step} train loss: {train_loss} test loss: {test_loss}"
                )
                # save the model to disk if it has improved
                if best_loss is None or test_loss < best_loss:
                    out_path = os.path.join(args.work_dir, "model.pt")
                    print(
                        f"test loss {test_loss} is the best so far, saving model to {out_path}"
                    )
                    torch.save(model.state_dict(), out_path)
                    best_loss = test_loss
                epoch += 1

            # sample from the model
            if step > 0 and step % 200 == 0:
                print_samples()

            step += 1
            # termination conditions
            if args.max_steps >= 0 and step >= args.max_steps:
                break

        memory.set_meta('data_hash', data_hash)

    chat(model, args.input_file, memory)
