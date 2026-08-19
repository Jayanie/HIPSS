import os
import torch
import sys

import torch.nn.functional as F
import torch.nn as nn
from torchvision.transforms import transforms

sys.path.append('../')

from dataloaders.WSI_Loader import camelyon16_dataloader
from conch.open_clip_custom import create_model_from_pretrained

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PatchEncoder(nn.Module):
    def __init__(self, conch_model, preprocess, device):
        super().__init__()
        self.conch_model = conch_model
        self.preprocess = preprocess
        self.device = device

    @torch.no_grad()
    def forward(self, patches):
        preprocessed_patches = (torch.stack([self.preprocess(transforms.ToPILImage()(img)) for img in patches]).
                                to(self.device))
        patch_embeddings, _ = self.conch_model.visual.forward_no_head_2(preprocessed_patches, normalize=False)
        return patch_embeddings


@torch.no_grad()
def embed_patches(region_tensor, patch_encoder):
    batch_size, num_regions, num_patches, c, h, w = region_tensor.shape
    region_tensor = region_tensor.squeeze(0)

    patch_embeddings = []
    for r in range(region_tensor.shape[0]):
        patches = region_tensor[r]
        for i in range(0, patches.shape[0], 256):
            minibatch_256 = patches[i:i + 256].to(device, non_blocking=True)
            minibatch_256 = F.interpolate(minibatch_256, size=(448, 448), mode='bilinear', align_corners=False)
            with torch.no_grad():
                patch_embeddings_minibatch = patch_encoder(minibatch_256)
            patch_embeddings.append(patch_embeddings_minibatch.detach().cpu())

    patch_embeddings = torch.vstack(patch_embeddings)
    dim = patch_embeddings.shape[1]
    patch_embeddings = patch_embeddings.view(batch_size, num_regions, num_patches, dim)
    return patch_embeddings


if __name__ == '__main__':
    conch_model_cfg = 'conch_ViT-B-16'
    # Replace the CONCH Model checkpoint path with the correct path
    conch_checkpoint_path = 'conch.pth'
    conch_model, preprocess = create_model_from_pretrained(conch_model_cfg, conch_checkpoint_path)
    conch_model = conch_model.to(device)

    # Keeping the weights of VLM image encoder frozen
    with torch.no_grad():
        patch_encoder = PatchEncoder(conch_model, preprocess, device)

    for param in patch_encoder.parameters():
        param.requires_grad = False

    """
    Repeat this process for Train and Validation folds and the test sets. The csv files are included in csv_files folder.
    """

    # Add the csv file path to Train Fold 1
    csv_train_fold_1 = ''
    # Add the path to Train Fold 1 h5 files
    h5_dir_train_fold_1 = ''

    wsi_loader_train_fold_1 = camelyon16_dataloader(csv_file=csv_train_fold_1, h5_dir=h5_dir_train_fold_1,
                                                    is_train=True,
                                                    is_val=False)
    tensors_save_dir_train = ''

    os.makedirs(tensors_save_dir_train, exist_ok=True)

    for batch_idx, (regions, label, image_name) in enumerate(wsi_loader_train_fold_1):
        regions_embedding = embed_patches(regions, patch_encoder)
        tensor_per_image = regions_embedding.squeeze(0)
        tensors_save_path = os.path.join(tensors_save_dir_train, image_name[0].replace('.tif', '.pt'))
        data = {
            "tensor": tensor_per_image,
            "label": label,
        }
        torch.save(data, tensors_save_path)
