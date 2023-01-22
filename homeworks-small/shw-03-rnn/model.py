import torch
from typing import Type
from torch import nn
from dataset import TextDataset
from torch.distributions.categorical import Categorical
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LanguageModel(nn.Module):
    def __init__(self, dataset: TextDataset, embed_size: int = 256, hidden_size: int = 256,
                 rnn_type: Type = nn.RNN, rnn_layers: int = 1):
        """
        Model for text generation
        :param dataset: text data dataset (to extract vocab_size and max_length)
        :param embed_size: dimensionality of embeddings
        :param hidden_size: dimensionality of hidden state
        :param rnn_type: type of RNN layer (nn.RNN or nn.LSTM)
        :param rnn_layers: number of layers in RNN
        """
        super(LanguageModel, self).__init__()
        self.dataset = dataset  # required for decoding during inference
        self.vocab_size = dataset.vocab_size
        self.max_length = dataset.max_length

        """
        YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        Create necessary layers
        """

        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=embed_size,
            padding_idx=dataset.pad_id
        )
        self.rnn = rnn_type(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=rnn_layers,
            batch_first=True
        )
        self.linear = nn.Linear(
            in_features=hidden_size,
            out_features=self.vocab_size
        )

    def forward(self, indices: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Compute forward pass through the model and
        return logits for the next token probabilities
        :param indices: LongTensor of encoded tokens of size (batch_size, length)
        :param lengths: LongTensor of lengths of size (batch_size, )
        :return: FloatTensor of logits of shape (batch_size, length, vocab_size)
        """
        """
        YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        Convert indices to embeddings, pass them through recurrent layers
        and apply output linear layer to obtain the logits
        """
        # output: (B, L, H), hidden: (B, H) in packed form

        indices[indices == self.dataset.pad_id] = self.embedding.padding_idx
        embeddings = self.embedding(indices)
        packed_embeddings = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
        context_vectors, hn = self.rnn(packed_embeddings)
        context_vectors, hn = pad_packed_sequence(context_vectors, batch_first=True)
        logits = self.linear(context_vectors)

        return logits

    @torch.inference_mode()
    def inference(self, prefix: str = '', temp: float = 1.) -> str:
        """
        Generate new text with an optional prefix
        :param prefix: prefix to start generation
        :param temp: sampling temperature
        :return: generated text
        """
        """
        YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        Encode the prefix (do not forget the BOS token!),
        pass it through the model to accumulate RNN hidden state and
        generate new tokens sequentially, sampling from categorical distribution,
        until EOS token or reaching self.max_length.
        Do not forget to divide predicted logits by temperature before sampling
        """
        # encode prefix
        self.eval()
        tokens = torch.tensor(self.dataset.text2ids(prefix), dtype=torch.int)
        tokens = torch.cat([
            torch.tensor([self.dataset.bos_id]),
            tokens
        ]).unsqueeze(0)

        # generate hidden for prefix
        logits = self.forward(tokens, torch.as_tensor([len(tokens)])) / temp

        # sample new token from logits
        new_tokens = Categorical(logits=logits[:, -1:]).sample()
        tokens = torch.cat([tokens, new_tokens], dim=1)

        # 2 stopping conditions: reaching max len or getting <eos> token
        while tokens.shape[1] < self.max_length:
            if new_tokens.item() == self.dataset.eos_id:
                break

            logits = self.forward(new_tokens, torch.as_tensor([len(new_tokens)])) / temp

            # sample the next token from logits
            new_tokens = Categorical(logits=logits[:, -1:]).sample()
            tokens = torch.cat([tokens, new_tokens], dim=1)

        # decode result to a string

        # tokens = torch.cat([
        #     tokens,
        #     torch.tensor([self.dataset.eos_id])
        # ], dim=1)

        return self.dataset.ids2text(tokens.squeeze())
