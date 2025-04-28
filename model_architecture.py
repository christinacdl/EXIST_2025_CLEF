import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from transformers.modeling_outputs import SequenceClassifierOutput
from typing import Optional, Tuple, Union
from transformers.models.xlm_roberta.modeling_xlm_roberta import (
    XLMRobertaModel,
    XLMRobertaPreTrainedModel,
    XLMRobertaLayer)


def apply_pooling(x, attention_mask, strategy="cls"):
    if isinstance(attention_mask, tuple):
        attention_mask = attention_mask[0]

    if attention_mask is not None:
        mask = attention_mask.unsqueeze(-1).to(dtype=x.dtype, device=x.device)
        mean_pool = (x * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        x_masked = x.masked_fill(mask == 0, -1e9)
        max_pool = x_masked.max(dim=1).values
    else:
        mean_pool = x.mean(dim=1)
        max_pool = x.max(dim=1).values

    if strategy == "cls":
        return x[:, 0, :]
    elif strategy == "mean":
        return mean_pool
    elif strategy == "max":
        return max_pool
    else:
        raise ValueError(f"Unknown pooling strategy: {strategy}")


class ClassificationHeadXLMRoberta(nn.Module):
    """
    Custom transformer-based classification head for sentence-level tasks.
    """

    def __init__(self, config, num_labels, aux_dim: int = 0):
        super().__init__()

        self.extra_layer = XLMRobertaLayer(config)
        # self.extra_layer_2 = XLMRobertaLayer(config)
        # self.extra_layer_3 = XLMRobertaLayer(config)
        # self.extra_layer_4 = XLMRobertaLayer(config)
        self.input_dim = config.hidden_size + aux_dim
        self.dense = nn.Linear(self.input_dim, config.hidden_size)
        self.dropout = nn.Dropout(config.classifier_dropout or config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)        

    def forward(self, x, aux_input=None, attention_mask=None, strategy="mean"):
        x = self.extra_layer(x)[0]
        # x = self.extra_layer_2(x)[0]
        # x = self.extra_layer_3(x)[0]
        # x = self.extra_layer_4(x)[0]
        pooled = apply_pooling(x, attention_mask, strategy=strategy)

        if aux_input is not None:
            pooled = torch.cat([pooled, aux_input], dim=-1)

        x = self.dropout(pooled)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return self.out_proj(x)


class MultiTask_MultiHead_XLMRoberta(XLMRobertaPreTrainedModel):
    """
    Multi-task XLM-RoBERTa model with:
    - Language-specific heads for multi-label sexism classification (EN, ES)
    - Shared head for multi-class sentiment classification
    """

    def __init__(
        self,
        config,
        num_sexist_labels: int,
        num_sentiment_labels: int,
        pooling_strategy: str = "cls",):
        super().__init__(config)
        self.num_sexist_labels = num_sexist_labels
        self.num_sentiment_labels = num_sentiment_labels

        self.roberta = XLMRobertaModel(config, add_pooling_layer=True)
        self.dropout = nn.Dropout(config.classifier_dropout or config.hidden_dropout_prob)
        self.pooling_strategy = pooling_strategy

        # Language-specific heads for sexism classification
        self.sexist_heads = nn.ModuleDict({
            "en": ClassificationHeadXLMRoberta(config, num_sexist_labels, aux_dim=num_sentiment_labels),
            "es": ClassificationHeadXLMRoberta(config, num_sexist_labels, aux_dim=num_sentiment_labels),})

        # Language-specific heads for sentiment classification
        self.sentiment_heads = nn.ModuleDict({
            "en": ClassificationHeadXLMRoberta(config, num_sentiment_labels),
            "es": ClassificationHeadXLMRoberta(config, num_sentiment_labels),})

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels_sexist: Optional[torch.FloatTensor] = None,
        labels_sentiment: Optional[torch.LongTensor] = None,
        language: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Union[Tuple, SequenceClassifierOutput]:

        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hidden_state = outputs.last_hidden_state  # shape (batch_size, seq_len, hidden)
        pooled_output = self.dropout(hidden_state)

        id2lang = {0: "en", 1: "es"}
        logits_sexist_list = []
        logits_sentiment_list = []

        for i in range(pooled_output.size(0)):
            lang = id2lang[int(language[i])]
            sentiment_head = self.sentiment_heads[lang]
            sexism_head = self.sexist_heads[lang]

            sent_logits = sentiment_head(pooled_output[i].unsqueeze(0), attention_mask=attention_mask[i].unsqueeze(0), strategy=self.pooling_strategy)
            sexism_logits = sexism_head(pooled_output[i].unsqueeze(0), aux_input=sent_logits, attention_mask=attention_mask[i].unsqueeze(0), strategy=self.pooling_strategy)

            logits_sexist_list.append(sexism_logits)
            logits_sentiment_list.append(sent_logits)

        logits_sexist = torch.cat(logits_sexist_list, dim=0)
        logits_sentiment = torch.cat(logits_sentiment_list, dim=0)

        return SequenceClassifierOutput(
            logits=(logits_sexist, logits_sentiment, language if language is not None else None),
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )
