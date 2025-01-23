import torch
import torch.nn as nn
from modules.tailored_decoder import EncoderDecoder
import torchvision.models as models
from modules.pvtv2 import pvt_v2_b2
import math
import torch.nn.functional as F
from info_nce import InfoNCE
import random
import torchvision
from einops import repeat, rearrange
from torchvision.models import ResNet18_Weights
from torchvision.models import ResNet50_Weights
from torchvision.models import ResNet101_Weights
from torchvision.models import ResNet152_Weights
from torchvision.models import DenseNet169_Weights
from torchvision.models import regnet_y_128gf, RegNet_Y_128GF_Weights
from torchvision.models import regnet_y_16gf, RegNet_Y_16GF_Weights
from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPVisionModel


class Model(nn.Module):
    def __init__(self, args, tokenizer):
        super(Model, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.encoder_decoder = EncoderDecoder(args, tokenizer)
        self.mse_loss = torch.nn.MSELoss(reduction="mean")
        self.kl_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=False)
        self.L = (8-self.args.kernel)**2
        # text encoder
        self.word_embd = nn.Embedding(self.encoder_decoder.vocab_size + 1, args.d_model)
        self.word_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=args.d_model, nhead=8), num_layers=3)
        self.word_mlp = nn.Sequential(nn.Linear(args.d_model, args.d_model), nn.Tanh(), nn.Linear(args.d_model, self.L))
        self.att_embed_report = nn.Sequential(nn.Linear(args.d_model, args.d_model), nn.ReLU(), nn.Linear(args.d_model, args.d_model), nn.Dropout(args.drop_prob_lm))

        pe = torch.zeros(120, args.d_model)
        position = torch.arange(0, 120).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, args.d_model, 2).float() * -(math.log(10000.0) / args.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

        # image encoder
        if args.visual_extractor == "densenet121":
            model = torchvision.models.densenet121(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1)  # no pretrained weights
            model.classifier = nn.Linear(model.classifier.in_features, 512, bias=False)  # projection head
            modules = nn.ModuleList(model.children())[:-1]  # Removing the final classifier layer
            self.vision = nn.Sequential(*modules)
            self.att_feat_size = 1024
        
        elif args.visual_extractor == "densenet169":
            model = torchvision.models.densenet169(weights=torchvision.models.DenseNet169_Weights.IMAGENET1K_V1)  # 使用预训练权重
            model.classifier = nn.Linear(model.classifier.in_features, 512, bias=False)  # 投影头
            modules = nn.ModuleList(model.children())[:-1]  # 移除最后的分类器层
            self.vision = nn.Sequential(*modules)
            self.att_feat_size = 1664

        elif args.visual_extractor == "resnet18":
            model = getattr(models, args.visual_extractor)(weights=ResNet18_Weights.IMAGENET1K_V1)
            modules = nn.ModuleList(model.children())[:-2]
            self.vision = nn.Sequential(*modules)
            self.att_feat_size = 512

        elif args.visual_extractor == "resnet50":
            model = torchvision.models.resnet50(weights=None)
            model.fc = nn.Linear(model.fc.in_features, 512, bias=False) # projection head
            state_dict = torch.load("/root/autodl-tmp/DFFG-MLLMR/medclip/pytorch_model.bin")
            diction = {}
            for key in state_dict:
                if key.split(".")[0]== "vision_model":
                    diction_key = key.replace("vision_model.model.","")
                    diction[diction_key] = state_dict[key]
            model.load_state_dict(diction, strict=False)
            modules = nn.ModuleList(model.children())[:-2]
            self.vision = nn.Sequential(*modules) 
            self.att_feat_size = 2048
            
        # elif args.visual_extractor == "resnet50":
        #     model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)  # no pretrained weights
        #     model.fc = nn.Linear(model.fc.in_features, 512, bias=False)  # projection head
        #     modules = nn.ModuleList(model.children())[:-2]  # Removing the final fully connected layer
        #     self.vision = nn.Sequential(*modules) 
        #     self.att_feat_size = 2048
        
        elif args.visual_extractor == "resnet101":
            model = torchvision.models.resnet101(weights = ResNet101_Weights.IMAGENET1K_V2)  # no pretrained weights
            model.fc = nn.Linear(model.fc.in_features, 512, bias = False)  # projection head
            modules = nn.ModuleList(model.children())[:-2]  # Removing the final fully connected layer
            self.vision = nn.Sequential(*modules)
            self.att_feat_size = 2048
            
        elif args.visual_extractor == "resnet152":
            model = torchvision.models.resnet152(weights = ResNet152_Weights.IMAGENET1K_V1)  # no pretrained weights
            model.fc = nn.Linear(model.fc.in_features, 512, bias = False)  # projection head
            modules = nn.ModuleList(model.children())[:-2]  # Removing the final fully connected layer
            self.vision = nn.Sequential(*modules)
            self.att_feat_size = 2048
               # 添加 EfficientNet-B7
        elif args.visual_extractor == "efficientnet_b7":
            model = timm.create_model('efficientnet_b7', pretrained=True)
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, 512, bias=False)  # 投影头
            modules = list(model.children())[:-1]  # 移除分类器层
            self.vision = nn.Sequential(*modules)
            self.att_feat_size = 2560  # EfficientNet-B7 的输出特征尺寸
            
        # 添加 Vision Transformer (ViT)
        elif args.visual_extractor == "vit_base_patch16_224":
            model = timm.create_model('vit_base_patch16_224', pretrained=True)
            in_features = model.head.in_features
            model.head = nn.Linear(in_features, 512, bias=False)  # 投影头
            modules = list(model.children())[:-1]  # 移除分类器层
            self.vision = nn.Sequential(*modules)
            self.att_feat_size = 768  # ViT Base 的输出特征尺寸

        # 添加 Swin Transformer
        elif args.visual_extractor == "swin_large_patch4_window7_224":
            model = timm.create_model('swin_large_patch4_window7_224', pretrained=True)
            in_features = model.head.in_features
            model.head = nn.Linear(in_features, 512, bias=False)  # 投影头
            modules = list(model.children())[:-1]  # 移除分类器层
            self.vision = nn.Sequential(*modules)
            self.att_feat_size = 1024  # Swin Large 的输出特征尺寸

        # 添加 ConvNeXt
        elif args.visual_extractor == "convnext_large":
            model = timm.create_model('convnext_large', pretrained=True)
            in_features = model.classifier[2].in_features  # ConvNeXt 的分类器通常是 Sequential 模块
            model.classifier[2] = nn.Linear(in_features, 512, bias=False)  # 投影头
            modules = list(model.children())[:-1]  # 移除分类器层
            self.vision = nn.Sequential(*modules)
            self.att_feat_size = 1536  # ConvNeXt Large 的输出特征尺寸     
        
        elif args.visual_extractor == "regnet_y_128gf":
            # 加载带有 SWAG 优化权重的 RegNet_Y_128GF 模型
            model = regnet_y_128gf(weights=RegNet_Y_128GF_Weights.IMAGENET1K_SWAG_E2E_V1)
            # 提取 RegNet 模型的所有层并移除最后的全连接层
            # 这里我们不需要替换全连接层（因为 RegNet 不同于 ResNet, 它通常不含有 FC 层），
            # 但需要确保去除最后的分类头
            modules = nn.ModuleList(model.children())[:-2]  # 移除最后的全连接层和池化层
            # 将剩余的模块作为新的视觉特征提取器
            self.vision = nn.Sequential(*modules)
            # 设置特征的维度大小（根据 RegNet_Y_128GF 的输出特征维度）
            self.att_feat_size = 1920  # RegNet_Y_128GF的输出通道数是1920（与模型架构相关）
        elif args.visual_extractor == "regnet_y_16gf":
            # 加载带有 SWAG 优化权重的 RegNet_Y_128GF 模型
            model = regnet_y_16gf(weights=RegNet_Y_16GF_Weights.IMAGENET1K_SWAG_E2E_V1)
            # 提取 RegNet 模型的所有层并移除最后的全连接层
            # 这里我们不需要替换全连接层（因为 RegNet 不同于 ResNet, 它通常不含有 FC 层），
            # 但需要确保去除最后的分类头
            modules = nn.ModuleList(model.children())[:-2]  # 移除最后的全连接层和池化层
            # 将剩余的模块作为新的视觉特征提取器
            self.vision = nn.Sequential(*modules)
            # 设置特征的维度大小（根据 RegNet_Y_128GF 的输出特征维度）
            self.att_feat_size = 3024  # RegNet_Y_128GF的输出通道数是1920（与模型架构相关）
    
        else:
            self.vision = pvt_v2_b2()  # Use a different visual extractor
            self.att_feat_size = 512

        d_middle = 1024
        self.cnn = nn.Conv2d(self.att_feat_size, d_middle, self.args.kernel, stride=1)
        self.att_embed_image = nn.Sequential(nn.Linear(d_middle, args.d_model), nn.ReLU(),nn.Linear(args.d_model, args.d_model), nn.Dropout(args.drop_prob_lm))
        
        # 初始化alpha为可学习参数，初始值设为0.5，你可以根据经验调整
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
        # 初始化注意力权重参数，确保它们是可训练的
        self.attention_weight_img = nn.Parameter(torch.ones(1))  # 图像特征的权重
        self.attention_weight_txt = nn.Parameter(torch.ones(1))  # 文本特征的权重
        
    def forward(self, images, targets=None, tok=None, mode='train', tags=0, epoch_id=0):
        # in training, sample_v and sample_t

        if mode == 'train':
            if self.args.visual_extractor == "pvt":
                patch_feats = self.cnn(self.vision(images)[3])
            else:
                patch_feats = self.cnn(self.vision(images))
            batch_size, feat_size, _, _ = patch_feats.shape
            patch_feats_f = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1).contiguous()
            att_feats_0 = self.att_embed_image(patch_feats_f)
            
            if tags == 1:
                word_embeddings = self.word_embd(tok)  # targets or tok
            else:
                word_embeddings = self.word_embd(targets)  # targets or tok
            word_embeddings = word_embeddings + self.pe[:, : word_embeddings.size(1)]  # x = x + self.pe[:, : x.size(1)]
            H = self.word_encoder(word_embeddings)
            mid = self.word_mlp(H)  # BS * n * r
            p_attn = F.softmax(mid.transpose(-2, -1), dim=-1)
            sturctured_emb_0 = self.att_embed_report(torch.matmul(p_attn, H))


# 特征融合部分
            img_weighted_feats = self.attention_weight_img * att_feats_0 + (1 - self.alpha) * sturctured_emb_0
            txt_weighted_feats = self.attention_weight_txt * sturctured_emb_0 + (1 - self.alpha) * att_feats_0

            if self.alpha < 0.5:
                feats = self.alpha * img_weighted_feats + (1 - self.alpha) * txt_weighted_feats
            else:
                feats = (1 - self.alpha) * img_weighted_feats + self.alpha * txt_weighted_feats

            output_t = self.encoder_decoder(txt_weighted_feats, targets, mode='forward')
            output_v = self.encoder_decoder(img_weighted_feats, targets, mode='forward')

            return output_t, output_v
            
        elif mode == 'sample_v':
            if self.args.visual_extractor == "pvt":
                patch_feats = self.cnn(self.vision(images)[3])
            else:
                patch_feats = self.cnn(self.vision(images))
            batch_size, feat_size, _, _ = patch_feats.shape
            patch_feats_f = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1).contiguous()  
            att_feats_0 = self.att_embed_image(patch_feats_f)
            output_v, probabilities = self.encoder_decoder(att_feats_0, att_feats_0, mode='sample')
            return output_v

        elif mode == 'sample_t':
            if tags == 1:
                word_embeddings = self.word_embd(tok)  # targets or tok
            else:
                word_embeddings = self.word_embd(targets)  # targets or tok
            word_embeddings = word_embeddings + self.pe[:, : word_embeddings.size(1)]  # x = x + self.pe[:, : x.size(1)]
            H = self.word_encoder(word_embeddings)
            mid = self.word_mlp(H)  # BS * n * r
            p_attn = F.softmax(mid.transpose(-2, -1), dim=-1)
            sturctured_emb_0 = self.att_embed_report(torch.matmul(p_attn, H))
            output_t, probabilities = self.encoder_decoder(sturctured_emb_0, sturctured_emb_0, mode='sample')
            return output_t


