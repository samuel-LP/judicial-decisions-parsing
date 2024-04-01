import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
import torch.nn.functional as F


class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, intermediate_size,
                 dropout_rate=0.1):
        super(TransformerLayer, self).__init__()
        self.attention = nn.MultiheadAttention(hidden_size,
                                               num_attention_heads,
                                               dropout=dropout_rate)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.linear1 = nn.Linear(hidden_size, intermediate_size)
        self.activation = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(intermediate_size, hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)

    def forward(self, inputs, attention_mask=None):
        attention_output, _ = self.attention(inputs, inputs, inputs,
                                             attn_mask=attention_mask)
        attention_output = self.dropout1(attention_output)
        attention_output = self.layer_norm1(inputs + attention_output)
        intermediate_output = self.linear1(attention_output)
        intermediate_output = self.activation(intermediate_output)
        intermediate_output = self.dropout2(intermediate_output)
        layer_output = self.linear2(intermediate_output)
        layer_output = self.layer_norm2(attention_output + layer_output)
        return layer_output


class MyBertModel(nn.Module):
    def __init__(self, model_name="almanach/camembert-base", num_labels=2):
        super(MyBertModel, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name,
                                                 output_hidden_states=True)
        self.backbone = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(self.config.hidden_size, num_labels)
        self.transformer_layer = \
            TransformerLayer(hidden_size=self.config.hidden_size,
                             num_attention_heads=self.config.num_attention_heads,
                             intermediate_size=self.config.intermediate_size,
                             dropout_rate=self.config.hidden_dropout_prob)
        
        self.dense = nn.Linear(num_labels, 1)

    def forward(self, batch):
        inputs = {k: v for k, v in batch.items() if k in ['input_ids',
                                                          'attention_mask']}
        outputs = self.backbone(**inputs)
        pooled = outputs[1]
        dropout_output = self.dropout(pooled)
        transformer_output = self.transformer_layer(dropout_output)
        x = self.fc(transformer_output)
        x = F.relu(x)
        x = self.dense(x)
        x = torch.sigmoid(x)
        return x
