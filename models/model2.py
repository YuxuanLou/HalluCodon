import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import multimolecule
from multimolecule import RnaTokenizer, RnaFmModel
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoConfig


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionPooling, self).__init__()
        self.query_linear = nn.Linear(hidden_size,
                                      hidden_size)
        self.key_linear = nn.Linear(hidden_size,
                                    hidden_size)

    def forward(self, hidden_state,
                attention_mask):
        batch_size, seq_len, _ = hidden_state.size()
        Q = self.query_linear(
            hidden_state[:, 0:1, :])
        K = self.key_linear(hidden_state)
        attention_scores = torch.matmul(Q,
                                        K.transpose(
                                            -2,
                                            -1))
        attention_scores = attention_scores / torch.sqrt(
            torch.tensor(K.size(-1),
                         dtype=torch.float32,
                         device=K.device))
        attention_mask = attention_mask.unsqueeze(
            1).float()
        attention_scores = torch.where(
            attention_mask != 0, attention_scores,
            torch.tensor(-1e9,
                         device=attention_scores.device))
        attention_weights = F.softmax(
            attention_scores, dim=-1)
        pooled_output = torch.matmul(
            attention_weights, hidden_state)
        return pooled_output.squeeze(1)


class CustomPlantRNAModel(nn.Module):
    def __init__(self, config):
        super(CustomPlantRNAModel,
              self).__init__()
        #self.plantrna_config = AutoConfig.from_pretrained("multimolecule/mrnafm")
        #self.esm2_config = AutoConfig.from_pretrained("facebook/esm2_t33_650M_UR50D")
        # Load PlantRNA model
        self.plantrna = RnaFmModel.from_pretrained("multimolecule/mrnafm")

        # Load ESM2 model
        self.esm2 = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
        # Freeze PlantRNA and ESM2 parameters
        #for param in self.plantrna.parameters():
            #param.requires_grad = False
        #for param in self.esm2.parameters():
            #param.requires_grad = False

        self.hidden_size = 1280
        self.pooler = AttentionPooling(self.hidden_size)


        # Normalization and activation
        self.cds_layernorm = nn.LayerNorm(self.hidden_size)
        self.protein_layernorm = nn.LayerNorm(self.hidden_size)
        self.resnet_layernorm = nn.LayerNorm(self.hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.2)

        # Fusion weights parameter
        self.fusion_raw_weights = nn.Parameter(
            torch.tensor([0.0, 0.0]))

        self.fc1 = nn.Linear(self.hidden_size,self.hidden_size // 2)
        self.fc2 = nn.Linear(self.hidden_size // 2,self.hidden_size // 4)
        self.fc3 = nn.Linear(self.hidden_size // 4, 1)

        # 辅助预测层 - 只使用CDS特征
        self.aux_fc1 = nn.Linear(self.hidden_size,self.hidden_size // 2)
        self.aux_fc2 = nn.Linear(self.hidden_size // 2, 1)


        # 损失函数
        self.loss_fn = nn.BCEWithLogitsLoss()

    def get_fusion_weights(self):
        return F.softmax(self.fusion_raw_weights,
                         dim=0)

    def forward(self, cds_input_ids=None,
                cds_attention_mask=None,
                protein_input_ids=None,
                protein_attention_mask=None,
                inputs_embeds=None,
                pre_computed_protein_embeddings=None,
                labels=None, **kwargs):
        # Process protein sequence
        if pre_computed_protein_embeddings is not None:
            pro_last_hidden_state = pre_computed_protein_embeddings
        else:
            protein_outputs = self.esm2(input_ids=protein_input_ids,attention_mask=protein_attention_mask)
            pro_last_hidden_state = protein_outputs.last_hidden_state
            pro_last_hidden_state = self.protein_layernorm(pro_last_hidden_state)
        cds_outputs = self.plantrna(input_ids=cds_input_ids,attention_mask=cds_attention_mask)
        cds_last_hidden_state = cds_outputs.last_hidden_state

        cds_last_hidden_state = self.cds_layernorm(cds_last_hidden_state)
        # 保存原始CDS特征用于辅助损失
        original_cds_features = cds_last_hidden_state.clone()

        # Weighted feature fusion
        weights = self.get_fusion_weights()
        alpha, beta = weights[0], weights[1]


        resnet_out = (alpha * cds_last_hidden_state + beta * pro_last_hidden_state)

        resnet_out = self.resnet_layernorm(resnet_out)
        resnet_out = self.dropout(resnet_out)
        resnet_out = self.activation(resnet_out)

        # Prediction path
        pooled_output_cds = self.pooler(resnet_out,cds_attention_mask)

        # 使用拼接后的特征进行主预测
        x = self.fc1(pooled_output_cds)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.activation(x)
        x = self.dropout(x)

        # Output logits
        logits = self.fc3(x)
        outputs = {"logits": logits}

        # 辅助预测路径 - 仅使用CDS特征
        pooled_cds = self.pooler(original_cds_features,cds_attention_mask)

        aux_x = self.aux_fc1(pooled_cds)
        aux_x = self.activation(aux_x)
        aux_x = self.dropout(aux_x)
        aux_logits = self.aux_fc2(aux_x)

        if labels is not None:
            # Note: BCEWithLogitsLoss expects float labels
            main_loss = self.loss_fn(
                logits.squeeze(-1),
                labels.float())
            # 辅助损失
            aux_loss = self.loss_fn(
                aux_logits.squeeze(-1),
                labels.float())

            total_loss = main_loss + 1 * aux_loss
            outputs["loss"] = total_loss
        return outputs

    def get_learned_parameters(self):
        weights = self.get_fusion_weights()
        return {
            "alpha": weights[0].item(),
            "beta": weights[1].item() }

    def compute_protein_embeddings(self,
                                   protein_input_ids,
                                   protein_attention_mask,
                                   device="cuda"):
        with torch.no_grad():
            protein_outputs = self.esm2(
                input_ids=protein_input_ids,
                attention_mask=protein_attention_mask
            )
            pro_last_hidden_state = protein_outputs.last_hidden_state
            pro_last_hidden_state = self.protein_layernorm(pro_last_hidden_state)
            return pro_last_hidden_state