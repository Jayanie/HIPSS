import torch
import torch.nn as nn

from conch_extended.open_clip_custom import create_model_from_pretrained, tokenize, get_tokenizer
import torch.nn.functional as F

"""
Add the LLM-generated descriptions for each dataset considering the entire WSI and region-level. 
There are 2 classes in Camelyon16 and TCGA-Lung datasets. So, there will be 2 wsi level descriptions and 
2 region level descriptions in these datasets.
In contrast, there are 5 classes in UBC-OCEAN dataset. Therefore, there will be 5 wsi level descriptions and
another 5 region level descriptions.
"""
wsi_level_description_class_1 = ''
wsi_level_description_class_2 = ''

region_level_description_class_1 = ''
region_level_description_class_2 = ''


class TextEncoder(nn.Module):
    def __init__(self, conch_model):
        super().__init__()
        self.transformer = conch_model.text.transformer
        self.positional_embedding = conch_model.text.positional_embedding
        self.ln_final = conch_model.text.ln_final
        self.text_projection = conch_model.text.text_projection
        self.dtype = next(conch_model.parameters()).dtype

    def forward(self, prompts1, prompts2):
        prompts1 = prompts1 + self.positional_embedding.type(self.dtype)
        prompts2 = prompts2 + self.positional_embedding.type(self.dtype)
        x = torch.cat([prompts1, prompts2], dim=1)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[:, 0] @ self.text_projection
        return x


"""
Region_Encoder: Region level Attention pooling based aggregation considering patch representations in each region with 
text-guided attention weights refinement
WSI_Encoder: WSI level Attention pooling based aggregation considering region representations in WSI with text-guided
attention weights refinement
"""


class Region_Encoder(nn.Module):
    def __init__(self, L=512, D=128, K=1, region_encoder_lambda=10.0, region_encoder_alpha=0.2):
        super(Region_Encoder, self).__init__()
        self.L = L
        self.D = D
        self.K = K
        self.region_encoder_lambda = region_encoder_lambda
        self.region_encoder_alpha = region_encoder_alpha

        self.attention_V1 = nn.Sequential(
            nn.Linear(L, D),
            nn.Tanh()
        )

        self.attention_V2 = nn.Sequential(
            nn.Linear(L, D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(D, K)

    def forward(self,
                x,
                relevance_scores=None,
                isNorm=True):
        A_V1 = self.attention_V1(x)
        A_V2 = self.attention_V2(x)
        A = self.attention_weights(A_V1 * A_V2)
        logits = A.transpose(1, 2)

        if relevance_scores is not None:
            new_relevance_scores = torch.where(relevance_scores >= self.region_encoder_alpha,
                                               relevance_scores * self.region_encoder_lambda,
                                               torch.where(relevance_scores > 0, relevance_scores,
                                                           torch.zeros_like(relevance_scores)))

            logits = logits + new_relevance_scores.unsqueeze(1)

        if isNorm:
            A = F.softmax(logits, dim=-1)
        return A


class WSI_Encoder(nn.Module):
    def __init__(self, L=512, D=128, K=1, wsi_encoder_lambda=10.0, wsi_encoder_alpha=0.2):
        super(WSI_Encoder, self).__init__()

        self.L = L
        self.D = D
        self.K = K
        self.wsi_encoder_lambda = wsi_encoder_lambda
        self.wsi_encoder_alpha = wsi_encoder_alpha

        self.attention_U1 = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U2 = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

    def forward(self,
                x,
                relevance_scores=None,
                isNorm=True):
        A_U1 = self.attention_U1(x)
        A_U2 = self.attention_U2(x)
        A = self.attention_weights(A_U1 * A_U2)
        logits = torch.transpose(A, 1, 0)

        if relevance_scores is not None:
            new_relevance_scores = torch.where(relevance_scores >= self.wsi_encoder_alpha,
                                               relevance_scores * self.wsi_encoder_lambda,
                                               torch.where(relevance_scores > 0, relevance_scores,
                                                           torch.zeros_like(relevance_scores)))
            logits = logits + new_relevance_scores.unsqueeze(0)

        if isNorm:
            A = F.softmax(logits, dim=1)

        return A


@torch.no_grad()
def make_prompts(conch_model, prompts, device):
    dtype = next(conch_model.parameters()).dtype
    tokenizer = get_tokenizer()
    tokenized_prompts = tokenize(texts=prompts, tokenizer=tokenizer).to(device)
    with torch.no_grad():
        embedding = conch_model.text.token_embedding(tokenized_prompts).type(dtype)
    return embedding


class HIPSS(nn.Module):
    def __init__(self, num_classes=3):
        super(HIPSS, self).__init__()
        self.num_classes = num_classes
        self.loss_ce = nn.CrossEntropyLoss()
        conch_model_cfg = 'conch_ViT-B-16'
        # Add the Path for the Conch Model
        conch_checkpoint_path = 'conch.pth'
        conch_model, preprocess = create_model_from_pretrained(conch_model_cfg, conch_checkpoint_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        conch_model = conch_model.to(self.device)
        _ = conch_model.eval()

        wsi_level_descriptions = [wsi_level_description_class_1, wsi_level_description_class_2]
        region_level_descriptions = [region_level_description_class_1, region_level_description_class_2]

        self.text_encoder = TextEncoder(conch_model)
        self.base_text_embedding = (
            make_prompts(conch_model=conch_model, prompts=wsi_level_descriptions, device=self.device))
        self.base_region_text_embedding = (
            make_prompts(conch_model=conch_model, prompts=region_level_descriptions, device=self.device))

        self.region_feature_encoder = Region_Encoder()
        self.wsi_feature_encoder = WSI_Encoder()

        self.logit_scale = conch_model.logit_scale

    def forward(self, x, label):
        x_gpu = x.to(self.device)
        text_features = self.text_encoder(self.base_text_embedding, self.base_region_text_embedding)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        patch_features = x_gpu.squeeze(0) / x_gpu.squeeze(0).norm(dim=-1, keepdim=True)
        text_scores_p = torch.einsum('rpd,cd->rpc', patch_features, text_features)
        text_relevance_p = text_scores_p.mean(dim=-1)
        AA = self.region_feature_encoder(x_gpu.squeeze(0), text_relevance_p)

        regions_outputs = torch.bmm(AA, x_gpu.squeeze(0))
        regions_outputs = regions_outputs.squeeze(1)
        regions_outputs_norm = regions_outputs / regions_outputs.norm(dim=-1, keepdim=True)

        text_relevance = torch.matmul(regions_outputs_norm, text_features.T).mean(-1)

        AA2 = self.wsi_feature_encoder(regions_outputs, text_relevance)
        wsi_embeddings = torch.mm(AA2, regions_outputs)
        image_features = wsi_embeddings / wsi_embeddings.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        loss = self.loss_ce(logits, label.to(self.device).squeeze(0))
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(Y_prob, 1, dim=1)[1]

        return Y_prob, Y_hat, loss
