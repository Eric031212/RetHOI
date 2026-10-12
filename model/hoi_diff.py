import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from model.mdm import MDM
from model.mdm import *
from model.retnet import RetNetEncoder, CrossRetentionCommunication
from model.gabor import MultiScaleGaborSmoother
from model.contrastive import ProjectionHead, masked_mean_pool, info_nce


class HOIDiff(MDM):
    """Dual-branch HOI diffusion model (HOI-DM).

    backbone (args.arch):
        'trans_enc' - original HOI-Diff: Transformer branches + cross-attention communication.
        'retnet'    - RetHOI: multi-scale retention branches + cross-retention communication.
    args.use_gabor       - learnable multi-scale Gabor smoothing of the predicted clean motion.
    args.use_contrastive - InfoNCE alignment of pooled human/object features with CLIP text.
    """

    def __init__(self, modeltype, njoints, nfeats, num_actions, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu", legacy=False, data_rep='rot6d', dataset='amass', clip_dim=512,
                 arch='trans_enc', emb_trans_dec=False, clip_version=None, args=None, **kargs):
        # the MDM constructor only knows Transformer variants; the backbone is replaced below
        super(HOIDiff, self).__init__(modeltype, njoints-6, nfeats, num_actions, translation, pose_rep, glob, glob_rot,
                         latent_dim, ff_size, num_layers, num_heads, dropout,
                         ablation, activation, legacy, data_rep, dataset, clip_dim,
                         'trans_enc', emb_trans_dec, clip_version, **kargs)

        self.args = args
        self.arch = arch
        self.use_gabor = getattr(args, 'use_gabor', False)
        self.use_contrastive = getattr(args, 'use_contrastive', False)
        self.last_aux = {}

        assert 0 < self.args.multi_backbone_split <= self.num_layers
        print(f'[HOI-DM] backbone={self.arch} split@{self.args.multi_backbone_split} '
              f'gabor={self.use_gabor} contrastive={self.use_contrastive}')
        del self.seqTransEncoder

        if self.arch == 'trans_enc':
            def trans_enc(n):
                layer = nn.TransformerEncoderLayer(d_model=self.latent_dim, nhead=self.num_heads,
                                                   dim_feedforward=self.ff_size, dropout=self.dropout,
                                                   activation=self.activation)
                return nn.TransformerEncoder(layer, num_layers=n)

            self.seqTransEncoder_start = trans_enc(self.args.multi_backbone_split)
            self.seqTransEncoder_end = trans_enc(self.num_layers - self.args.multi_backbone_split)
            self.seqTransEncoder_obj_pose = trans_enc(2)
            self.mutual_attn = MutualAttention(num_layers=2, latent_dim=self.latent_dim,
                                               input_feats=self.input_feats)
        elif self.arch == 'retnet':
            def retnet(n):
                return RetNetEncoder(n, self.latent_dim, self.num_heads, self.ff_size, self.dropout,
                                     causal=getattr(args, 'causal_retention', False))

            self.seqTransEncoder_start = retnet(self.args.multi_backbone_split)
            self.seqTransEncoder_end = retnet(self.num_layers - self.args.multi_backbone_split)
            self.seqTransEncoder_obj_pose = retnet(2)
            self.mutual_attn = CrossRetentionCommunication(num_layers=2, latent_dim=self.latent_dim,
                                                           num_heads=self.num_heads, ff_size=self.ff_size,
                                                           dropout=self.dropout)
        else:
            raise ValueError(f'Unsupported HOI-DM backbone [{self.arch}]; use trans_enc or retnet.')

        self.input_process_obj = InputProcess(self.data_rep, 6, self.latent_dim)
        self.output_process_obj = OutputProcess(self.data_rep, 6, self.latent_dim, 6, self.nfeats)

        if self.use_gabor:
            self.gabor_human = MultiScaleGaborSmoother(learnable=True, init_blend=0.1)
            self.gabor_obj = MultiScaleGaborSmoother(learnable=True, init_blend=0.1)

        if self.use_contrastive:
            self.proj_human = ProjectionHead(self.latent_dim, self.clip_dim)
            self.proj_obj = ProjectionHead(self.latent_dim, self.clip_dim)

    def mask_cond_obj(self, cond, force_mask=False):
        seq, bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(1, bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond

    def encode_obj(self, obj_points):
        # obj_points - [bs, n_points, 3]
        obj_emb = self.objEmbedding(obj_points) # [bs, n_points, d]
        obj_emb = obj_emb.permute(1, 0, 2) # [n_points, bs, d]
        return obj_emb

    def _smooth(self, smoother, out):
        # out: [bs, feats, 1, frames] -> smooth along frames
        return smoother(out.squeeze(2)).unsqueeze(2)

    def forward(self, x, timesteps, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        x_human, x_obj = x[:,:-6], x[:,-6:]
        emb = self.embed_timestep(timesteps)  # [1, bs, d]

        force_mask = y.get('uncond', False)
        enc_text = None
        if 'text' in self.cond_mode:
            enc_text = self.encode_text(y['text'])
            emb += self.embed_text(self.mask_cond(enc_text, force_mask=force_mask))

        x_human = self.input_process(x_human)
        x_obj = self.input_process_obj(x_obj)

        xseq_human = torch.cat((emb, x_human), axis=0)  # [seqlen+1, bs, d]
        xseq_human = self.sequence_pos_encoder(xseq_human)  # [seqlen+1, bs, d]
        human_mid = self.seqTransEncoder_start(xseq_human)

        xseq_obj = torch.cat((emb, x_obj), axis=0)
        xseq_obj = self.sequence_pos_encoder(xseq_obj)
        obj_mid = self.seqTransEncoder_obj_pose(xseq_obj)

        dec_output_human, dec_output_obj = self.mutual_attn(human_mid[1:], obj_mid[1:])
        feat_human = self.seqTransEncoder_end(torch.cat([human_mid[:1], dec_output_human], 0))[1:]
        feat_obj = dec_output_obj

        output_human = self.output_process(feat_human)
        output_obj = self.output_process_obj(feat_obj)

        if self.use_gabor:
            output_human = self._smooth(self.gabor_human, output_human)
            output_obj = self._smooth(self.gabor_obj, output_obj)

        self.last_aux = {}
        if self.use_contrastive and self.training and enc_text is not None:
            frame_mask = y['mask'].squeeze(1).squeeze(1) if 'mask' in y else None  # [bs, frames]
            z_human = self.proj_human(masked_mean_pool(feat_human, frame_mask))
            z_obj = self.proj_obj(masked_mean_pool(feat_obj, frame_mask))
            z_motion = 0.5 * (z_human + z_obj)
            self.last_aux['contrastive'] = info_nce(z_motion, enc_text.detach(), captions=y['text'])

        output = torch.cat([output_human, output_obj], dim=1)
        return output

    def trainable_parameters(self):
        return [p for name, p in self.named_parameters() if p.requires_grad]

    def freeze_block(self, block):
        block.eval()
        for p in block.parameters():
            p.requires_grad = False


class MutualAttention(nn.Module):
    def __init__(self, num_layers, latent_dim, input_feats):
        super().__init__()

        self.num_layers = num_layers
        self.latent_dim = latent_dim
        self.num_heads = 4
        self.ff_size = 1024
        self.dropout = 0.1
        self.activation = 'gelu'
        self.input_feats = input_feats

        seqTransDecoderLayer_obj = nn.TransformerDecoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=self.activation)

        seqTransDecoderLayer_human = nn.TransformerDecoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=self.activation)

        self.seqTransDecoder_human_pose = nn.TransformerDecoder(seqTransDecoderLayer_human,
                                                         num_layers=self.num_layers)

        self.seqTransDecoder_obj_pose = nn.TransformerDecoder(seqTransDecoderLayer_obj,
                                                         num_layers=self.num_layers)

    def forward(self, x_human, x_obj):
        dec_output_human = self.seqTransDecoder_human_pose(tgt=x_human, memory=x_obj)
        dec_output_obj = self.seqTransDecoder_obj_pose(tgt=x_obj, memory=x_human)
        return dec_output_human, dec_output_obj
