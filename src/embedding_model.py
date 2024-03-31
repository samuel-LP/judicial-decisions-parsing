import torch.nn as nn
from transformers import AutoConfig, AutoModel


class EmbeddingModel(nn.Module):
    def __init__(self,
                 model_name="flaubert/flaubert_base_uncased"
                 ):
        super().__init__()

        self.config = AutoConfig.from_pretrained(model_name,
                                                 output_hidden_states=True)
        self.backbone = AutoModel.from_pretrained(model_name)

    def forward(self, batch):
        x = self.backbone(input_ids=batch["input_ids"],
                          attention_mask=batch["attention_mask"]
                          ).last_hidden_state
        # Taking only CLS output for sentence classification
        x = x[:, 0, :]
        return x
