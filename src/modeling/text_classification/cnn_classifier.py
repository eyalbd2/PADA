from torch import Tensor
from torch.nn import Module, Dropout, Conv1d, AvgPool1d, Linear


class CnnClassifier(Module):
    def __init__(self, num_labels: int, hidden_size: int, max_seq_length: int, filter_size: int = 9,
                 out_channels: int = 16, padding=True, hidden_dropout_prob: float = 0.1) -> Tensor:
        super().__init__()
        self.dropout = Dropout(hidden_dropout_prob)
        padding_size = int((filter_size - 1) / 2) if padding else 0
        self.conv1 = Conv1d(in_channels=hidden_size, out_channels=out_channels, kernel_size=filter_size,
                               padding=padding_size)
        self.max_pool = AvgPool1d(kernel_size=2)
        self.classifier_in_size = int(out_channels * max_seq_length / 2) if padding else \
            int((out_channels * (max_seq_length - filter_size + 1)) / 2)
        self.classifier = Linear(self.classifier_in_size, num_labels)

    def forward(self, enc_sequence):
        enc_sequence = self.dropout(enc_sequence)
        enc_seq_shape = enc_sequence.shape
        enc_sequence = enc_sequence.reshape(enc_seq_shape[0], enc_seq_shape[2], enc_seq_shape[1])
        features = self.conv1(enc_sequence)
        features_shape = features.shape
        features = features.reshape(features_shape[0], features_shape[2], features_shape[1])
        final_features_shape = features.shape
        final_features = features.reshape(final_features_shape[0], final_features_shape[2], final_features_shape[1])
        final_features = self.max_pool(final_features)
        final_features_shape = final_features.shape
        flat = final_features.reshape(-1, final_features_shape[1] * final_features_shape[2])
        logits = self.classifier(flat)

        return logits
