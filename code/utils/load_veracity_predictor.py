"""loads NLI model also generates predictions for veracity of the claim."""

from transformers import AutoModel, AutoTokenizer
import torch
import os
from torch import Tensor

from torch import nn

dir_path = os.path.dirname(os.path.realpath(os.getcwd()))


class MultiClassClassifier(nn.Module):
    def __init__(self, bert_model_path, labels_count, hidden_dim=768, mlp_dim=500, extras_dim=100, dropout=0.1,state_dict_final=None, freeze_bert=False):
        super().__init__()

        self.roberta = AutoModel.from_pretrained(bert_model_path,output_hidden_states=True,output_attentions=True)
        if(state_dict_final):
            self.roberta.load_state_dict(state_dict_final,strict=False)

        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.ReLU(),
            # nn.Linear(mlp_dim, mlp_dim),
            # # nn.ReLU(),
            # # nn.Linear(mlp_dim, mlp_dim),
            # nn.ReLU(),
            nn.Linear(mlp_dim, labels_count)
        )
        # self.softmax = nn.LogSoftmax(dim=1)
        if freeze_bert:
            print("Freezing layers")
            for param in self.roberta.parameters():
                param.requires_grad = False

    def forward(self, tokens, masks):
        output = self.roberta(tokens, attention_mask=masks)
        dropout_output = self.dropout(output["pooler_output"])
        # concat_output = torch.cat((dropout_output, topic_emb), dim=1)
        # concat_output = self.dropout(concat_output)
        mlp_output = self.mlp(dropout_output)
        # proba = self.sigmoid(mlp_output)
        # proba = self.softmax(mlp_output)

        return mlp_output

import torch



class VeracityClassifier:
    """performs stance detection."""

    def __init__(self, base_model, model_name: str = None,device=None) -> None:
        """initialized the model.

        Args:
        base_model: the backbone model to load from
            model_name (str): name or path to model
        """
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-large-mnli")

        self.model = MultiClassClassifier("FacebookAI/roberta-large-mnli",3, 1024,768,140,dropout=0.2,freeze_bert=False)
        
        print(self.model)
        loaded_dict = torch.load(model_name, map_location="cpu")
        # loaded_dict = {k.lstrip('module.'):v for k,v in loaded_dict.items()}
        # loaded_dict = {('ml'+k if k[:2] =='p.' else k):v for k,v in loaded_dict.items()}
        self.model.load_state_dict(loaded_dict)
        self.model.to(device)

    def predict(self, input: str, max_legnth: int = 256) -> str:
        """predicts the veracity label given claim and evidence.

        Args:
            input (str): claim with evidences
            max_legnth (int, optional): max length of sequence. Defaults to 256.

        Returns:
            str: verdict
        """

        print("claim", input)

        x = self.tokenizer.encode_plus(
            input,
            return_tensors="pt",
            return_attention_mask=True,
            truncation=True,
            max_length=max_legnth,
        )
        with torch.no_grad():
            logits = self.model(x["input_ids"].to(self.device), x["attention_mask"].to(self.device))
            logits = logits.detach().cpu()

        probs = logits.softmax(dim=1)
        print(probs)
        label_index = probs.argmax(dim=1)

        if label_index == 2:
            label = "SUPPORTS"
        elif label_index == 1:
            label = "CONFLICTING"
        elif label_index == 0:
            label = "REFUTES"
        # else:
        #   label = "NONE"
        return label.upper(), probs
