import torch.nn as nn
from transformers import AutoConfig, AutoModel


class BertModel(nn.Module):
    def __init__(self, model_name="almanach/camembert-base", num_labels=3):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name,
                                                 output_hidden_states=True)
        self.backbone = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(self.config.hidden_size, num_labels)

    def forward(self, batch):
        inputs = {k: v for k, v in batch.items() if k in ['input_ids',
                                                          'attention_mask']}
        outputs = self.backbone(**inputs)
        pooled = outputs[1]
        dropout_output = self.dropout(pooled)
        # x = outputs.last_hidden_state[:, 0, :]
        x = self.fc(dropout_output)
        return x
