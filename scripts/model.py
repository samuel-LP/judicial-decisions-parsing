import torch.nn as nn
from transformers import AutoConfig, AutoModel


class MyBertModel(nn.Module):
    def __init__(self, model_name="almanach/camembert-base", num_labels=2):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name,
                                                 output_hidden_states=True)
        self.backbone = AutoModel.from_pretrained(model_name)
        self.fc = nn.Linear(self.config.hidden_size, num_labels)

    def forward(self, batch):
        inputs = {k: v for k, v in batch.items() if k in ['input_ids',
                                                          'attention_mask']}
        outputs = self.backbone(**inputs)
        x = outputs.last_hidden_state[:, 0, :]
        x = self.fc(x)
        return x
