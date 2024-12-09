import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import json


def freeze_model(m):
    m.requires_grad_(False)


def freeze_all_but_bn(m):
    if not isinstance(m, torch.nn.LayerNorm):
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.requires_grad_(False)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.requires_grad_(False)


class Position_Embedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(Position_Embedding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.position_embedding = nn.Embedding(max_len, d_model)


    def forward(self, x):
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), -1)
        return self.position_embedding(positions)


class Adapter(nn.Module):
    def __init__(self, args):
        super(Adapter, self).__init__()

        self.args = args

        self.mlp1 = nn.Linear(self.args['d_model'], self.args['adapter_dim'])
        self.activation = nn.ReLU()
        self.mlp2 = nn.Linear(self.args['adapter_dim'], self.args['d_model'])
    
    
    def forward(self, x):
        x = self.mlp2(self.activation(self.mlp1(x))) + x
        return x


class EncoderLayerWithAdapter(nn.Module):
    def __init__(self, args):
        super(EncoderLayerWithAdapter, self).__init__()

        self.args = args
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.args['d_model'], nhead=self.args['nhead'], 
                            dim_feedforward=self.args['dim_feedforward'], dropout=self.args['dropout'])
        self.norm1 = nn.LayerNorm(self.args['d_model'])
        self.norm2 = nn.LayerNorm(self.args['d_model'])
        self.Adapter1 = Adapter(args)
        self.Adapter2 = Adapter(args)

    
    def forward(self, x):
        x = self.norm1(x + self.Adapter1(self.encoder_layer._sa_block(x, None, None)))
        x = self.norm2(x + self.Adapter2(self.encoder_layer._ff_block(x)))

        return x


class EncoderWithAdapter(nn.Module):
    def __init__(self, args):
        super(EncoderWithAdapter, self).__init__()

        self.args = args
        # self.encoder_layer = EncoderLayerWithAdapter(args)
        self.encoder = nn.Sequential()

        for i in range(self.args['num_encoder_layers']):
            # self.encoder.append(self.encoder_layer)
            self.encoder.append(EncoderLayerWithAdapter(args))


    def forward(self, src):
        return self.encoder(src)


class DecoderLayerWithAdapter(nn.Module):
    def __init__(self, args):
        super(DecoderLayerWithAdapter, self).__init__()

        self.args = args
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=self.args['d_model'], nhead=self.args['nhead'], 
                            dim_feedforward=self.args['dim_feedforward'], dropout=self.args['dropout'])
        self.norm1 = nn.LayerNorm(self.args['d_model'])
        self.norm2 = nn.LayerNorm(self.args['d_model'])
        self.norm3 = nn.LayerNorm(self.args['d_model'])
        self.Adapter1 = Adapter(args)
        self.Adapter2 = Adapter(args)
        self.Adapter3 = Adapter(args)

    
    def forward(self, x, memory, tgt_mask):
        x = self.norm1(x + self.Adapter1(self.decoder_layer._sa_block(x, tgt_mask, None)))
        x = self.norm2(x + self.Adapter2(self.decoder_layer._mha_block(x, memory, None, None)))
        x = self.norm3(x + self.Adapter3(self.decoder_layer._ff_block(x)))

        return x


class DecoderWithAdapter(nn.Module):
    def __init__(self, args):
        super(DecoderWithAdapter, self).__init__()

        self.args = args
        # self.decoder_layer = DecoderLayerWithAdapter(args)
        self.decoder = nn.Sequential()#[]

        for i in range(self.args['num_decoder_layers']):
            # self.decoder.append(self.decoder_layer)
            self.decoder.append(DecoderLayerWithAdapter(args))


    def forward(self, src, memory, tgt_mask):
        for layer in self.decoder:
            src = layer(src, memory, tgt_mask)
        return src


class TimeSeriesTransformerWithInnerAdapter(nn.Module):
    def __init__(self, args):
        super(TimeSeriesTransformerWithInnerAdapter, self).__init__()

        self.args = args

        self.encoder_layer = EncoderLayerWithAdapter(args)
        self.encoder = EncoderWithAdapter(args)

        self.decoder_layer = DecoderLayerWithAdapter(args)
        self.decoder = DecoderWithAdapter(args)

        self.source_projection = nn.Linear(self.args['source_dim'], self.args['d_model'])
        self.target_projection = nn.Linear(self.args['target_dim'], self.args['d_model'])
        self.output_projection = nn.Linear(self.args['d_model'], self.args['output_dim'])

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).double()
        return mask


    def forward(self, src, tgt):
        src = self.source_projection(src)
        tgt = self.target_projection(tgt)
        
        memory = self.encoder(src)

        tgt_mask = self._generate_square_subsequent_mask(len(tgt)).to(src.device)
        
        output = self.decoder(tgt, memory, tgt_mask)

        output = self.output_projection(output)
        return output

    
    @staticmethod
    def load_config(config_path: str):
        """Load model configuration from file.
    
        Args:
            config_path: Path to model configuration file
    
        Returns:
            args: ModelArgs object
    
        """
        with open(config_path, 'r') as f:
            args = json.load(f)
        model = TimeSeriesTransformerWithInnerAdapter(args)
        return model