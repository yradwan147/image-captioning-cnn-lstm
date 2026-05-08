"""CNN-encoder + LSTM-decoder model for image captioning on MS-COCO.

The encoder uses a pretrained ResNet-18 and projects its 512-dim feature
vector down to `embed_size`. The decoder is an LSTM whose first input is
the image embedding and whose subsequent inputs are the embeddings of the
caption's tokens (teacher forcing during training; greedy step-by-step at
inference time).
"""
import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    """Frozen ResNet-18 → linear → embed_size."""

    def __init__(self, embed_size):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        for p in resnet.parameters():
            p.requires_grad_(False)
        # Drop the final classification layer; keep the 512-dim feature vector.
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed  = nn.Linear(resnet.fc.in_features, embed_size)
        # Match the reference architecture's input distribution.
        self.bn     = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images)             # (B, 512, 1, 1)
        features = features.view(features.size(0), -1)  # (B, 512)
        features = self.embed(features)                 # (B, embed)
        return self.bn(features)


class DecoderRNN(nn.Module):
    """Word-embedding → LSTM → vocab logits."""

    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.embed_size  = embed_size
        self.hidden_size = hidden_size
        self.vocab_size  = vocab_size
        self.num_layers  = num_layers

        self.embed  = nn.Embedding(vocab_size, embed_size)
        self.lstm   = nn.LSTM(input_size=embed_size,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        """`features` (B, embed) and `captions` (B, T) — including <start>."""
        # Drop <end> from caption since the decoder will *predict* it.
        embeddings = self.embed(captions[:, :-1])               # (B, T-1, E)
        # Concatenate the image feature as the first time step.
        inputs = torch.cat(
            (features.unsqueeze(1), embeddings),                # (B, T, E)
            dim=1,
        )
        hiddens, _ = self.lstm(inputs)                           # (B, T, H)
        outputs    = self.linear(hiddens)                        # (B, T, V)
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        """Greedy autoregressive decoding from the image feature."""
        predicted_sentence = []
        for _ in range(max_len):
            hiddens, states = self.lstm(inputs, states)          # (1, 1, H)
            outputs = self.linear(hiddens.squeeze(1))            # (1, V)
            _, predicted = outputs.max(1)                        # (1,)
            predicted_sentence.append(int(predicted.item()))
            inputs = self.embed(predicted).unsqueeze(1)          # (1, 1, E)
        return predicted_sentence
