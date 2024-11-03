import math
import torch
from torch import nn

def get_slopes(n_heads: int):
    n = 2 ** math.floor(math.log2(n_heads))
    m_0 = 2.0 ** (-8.0 / n) # 开始的斜率
    m = torch.pow(m_0, torch.arange(1, 1 + n))
    if n < n_heads:
        m_hat_0 = 2.0 ** (-4.0 / n)
        m_hat = torch.pow(m_hat_0, torch.arange(1, 1 + 2 * (n_heads - n), 2))
        m = torch.cat([m, m_hat])
    return m

seq_len = 10
n_heads = 8
m = get_slopes(n_heads)
print(m)

alibi_biases = torch.zeros(seq_len,seq_len)
for j in range(1,seq_len):
    for i in range(j, seq_len):
        alibi_biases[i, i - j] = -j
print(alibi_biases)

print(alibi_biases[:, :, None].shape, m[None, None, :].shape)