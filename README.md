# YamRNN

YamRNN (Yamada modified RNN) is a compact recurrent neural network layer for PyTorch.

## Usage

```python
import torch
from yamrnn import YamRNN

inputs = torch.randn(32, 100, 64)

rnn = YamRNN(input_size=64, hidden_size=128, batch_first=True)
outputs, hidden = rnn(inputs)
```

`YamRNN` follows the basic input and output convention of a single-layer, unidirectional `torch.nn.GRU`. With `batch_first=True`, `inputs` has shape `(batch, sequence, input_size)`, `outputs` has shape `(batch, sequence, hidden_size)`, and `hidden` has shape `(1, batch, hidden_size)`.

## Installation

```bash
python -m pip install "git+https://github.com/caramelazuki/YamRNN.git"
```

## Reference

Yamada KD, Lin F, Nakamura T, [Developing a Novel Recurrent Neural Network Architecture with Fewer Parameters and Good Learning Performance](https://doi.org/10.4036/iis.2020.R.01), Interdisciplinary Information Sciences, 27(1), 25–40, 2021
