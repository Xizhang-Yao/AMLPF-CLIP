import torch
import torch.nn as nn
import timm
from CLIP import clip
import clip as clip_module
class UNI_Classifier(nn.Module):
    def __init__(self,num_classes=2, args=None):
        super(UNI_Classifier, self).__init__()
        self.args = args
        self.backbone = timm.create_model(
            "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
        )
        if args.pretrained_path_UNI is not None:
            self.backbone.load_state_dict(torch.load(args.pretrained_path_UNI, map_location="cuda"),strict=True)
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.size(0)
        num_patch = x.size(1)
        x = x.view(-1, x.size(2), x.size(3), x.size(4))
        x = self.backbone(x)
        x = self.classifier(x)
        x = x.view(batch_size, num_patch, -1)
        x = (torch.mean(x[:, :num_patch - 1, :], dim=1) + x[:, num_patch - 1, :]) / 2
        return x, self.softmax(x)


class UNI_CLIP_Model(nn.Module):
    def __init__(self, clip_net='RN50', device='cuda', args=None, split_num=1):
        super(UNI_CLIP_Model, self).__init__()
        self.args = args
        self.stage = args.stage
        self.raw_weights = nn.Parameter(torch.ones(3))
        if self.args.model == "UNI_CLIP":
            self.UNI = UNI_Classifier()
            self.UNI.load_state_dict(torch.load(args.pretrained_teacher_path, map_location="cuda"))
            for param in self.UNI.parameters():
                param.requires_grad = False
        self.softmax = nn.Softmax(dim=1)
        self.base, _ = clip.load(clip_net, device=device)
        self.base = self.base.float()
        self.encode_image = self.base.encode_image
        # for param in self.base.visual.parameters():
        #     param.requires_grad = False
        self.encode_text = self.base.encode_text
        self.logit_scale = self.base.logit_scale

        # 冻住encode_text
        for p in self.base.token_embedding.parameters():
            p.requires_grad = False
        for p in self.base.transformer.parameters():
            p.requires_grad = False
        self.base.positional_embedding.requires_grad = False
        self.base.text_projection.requires_grad = False
        for p in self.base.ln_final.parameters():
            p.requires_grad = False
        self.get_text_feature()

    def get_text_feature(self):
        if self.args.Dataset == "Chaoyang":
            text_only_label = ["normal", "serrated", "adenocarcinoma", "adenoma"]
            text_basic_sentence = ["The type of the tissue is normal.",
                                       "The tissue shows serrated morphological features.",
                                       "The tissue is identified as adenocarcinoma.",
                                       "The tissue sample is adenoma."]
            text_detail_sentence = [
                    "This is an image of normal tissue with no tumor cells, characterized by healthy structure and organized cellular arrangements without signs of abnormal growth or malignancy.",
                    "This is an image of serrated tissue, showing saw-tooth patterns in the epithelial layer, often associated with precancerous conditions or abnormal glandular growth.",
                    "This is an image of adenocarcinoma, a malignant tumor originating in glandular tissues, characterized by irregular cellular architecture and invasive growth patterns under microscopic examination.",
                    "This is an image of adenoma, a benign tumor composed of glandular structures, typically well-circumscribed with regular cellular arrangements and non-invasive growth behavior."]
        elif self.args.Dataset == "BreaKHis_v1_8class":
            text_only_label = ["adenosis", "fibroadenoma", "phyllodes tumor", "tubular adenoma",
                               "ductal carcinoma", "lobular carcinoma", "mucinous carcinoma", "papillary carcinoma"]

            text_basic_sentence = ["The type of the tumor is " + t for t in
                                   ["adenosis", "fibroadenoma", "phyllodes tumor", "tubular adenoma",
                                    "ductal carcinoma", "lobular carcinoma", "mucinous carcinoma",
                                    "papillary carcinoma"]]
            text_detail_sentence = [
                "The type of the tumor is adenosis. Adenosis is a benign breast condition involving enlarged lobules with more glands than usual, often associated with fibrosis or sclerosing features.",
                "The type of the tumor is fibroadenoma. Fibroadenoma is a benign breast tumor consisting of fibrous and glandular tissues, typically well-circumscribed and mobile, common in young women.",
                "The type of the tumor is phyllodes tumor. Phyllodes tumor is a rare breast tumor with leaf-like stromal projections, classified as benign, borderline, or malignant based on histological features.",
                "The type of the tumor is tubular adenoma. Tubular adenoma is a rare benign breast tumor composed of small, uniform tubules lined by epithelial and myoepithelial cells.",
                "The type of the tumor is ductal carcinoma. Ductal carcinoma is a malignant breast cancer originating in the milk ducts, including non-invasive (DCIS) and invasive (IDC) subtypes.",
                "The type of the tumor is lobular carcinoma. Lobular carcinoma is a malignant breast cancer arising in the lobules, including in situ (LCIS) and invasive (ILC) forms, often presenting with diffuse growth patterns.",
                "The type of the tumor is mucinous carcinoma. Mucinous carcinoma is a rare breast cancer subtype characterized by extracellular mucin production, often with a favorable prognosis.",
                "The type of the tumor is papillary carcinoma. Papillary carcinoma is a rare breast cancer with papilla-like structures, often encapsulated and slow-growing, with varying malignant potential.", ]

        elif self.args.Dataset == "LungHist700_3class":
            text_only_label = ["adenocarcinomas", "squamous cell carcinomas", "normal lung tissues"]
            text_basic_sentence = ["The type of histopathological image is " + t for t in
                                   ["adenocarcinomas",
                                    "squamous cell carcinomas",
                                    "normal lung tissues"]]
            text_detail_sentence = [
                "The type of histopathological image is adenocarcinomas. Adenocarcinomas are malignant tumors arising from glandular epithelial cells, commonly found in the lungs, characterized by irregular cell structures and mucin production.",
                "The type of histopathological image is squamous cell carcinomas. Squamous cell carcinomas are malignant tumors originating from squamous epithelial cells, featuring keratinization and intercellular bridges in lung tissues.",
                "The type of histopathological image is normal lung tissues. Normal lung tissues exhibit healthy alveolar structures, devoid of atypical cells, with intact epithelial linings and clear airspaces."]

        self.text_feature_only_label = []
        for t in text_only_label:
            text_feature_CLIP = self.encode_text(clip_module.tokenize(t).to("cuda")).to(torch.float32)
            self.text_feature_only_label.append(text_feature_CLIP)

        self.text_feature_basic_sentence = []
        for t in text_basic_sentence:
            text_feature_CLIP = self.encode_text(clip_module.tokenize(t).to("cuda")).to(torch.float32)
            self.text_feature_basic_sentence.append(text_feature_CLIP)
        self.text_feature_detail_sentence = []
        for t in text_detail_sentence:
            text_feature_CLIP = self.encode_text(clip_module.tokenize(t).to("cuda")).to(torch.float32)
            self.text_feature_detail_sentence.append(text_feature_CLIP)

        self.text_feature_only_label = torch.cat(self.text_feature_only_label, dim=0)
        self.text_feature_basic_sentence = torch.cat(self.text_feature_basic_sentence, dim=0)
        self.text_feature_detail_sentence = torch.cat(self.text_feature_detail_sentence, dim=0)


    def test(self, image):
        batch_size = image.size(0)
        num_patch = image.size(1)
        image = image.view(-1, image.size(2), image.size(3), image.size(4))
        image_feature = self.encode_image(image).to(torch.float32)
        image_features = image_feature / image_feature.norm(dim=1, keepdim=True)
        self.text_feature_only_label = self.text_feature_only_label / self.text_feature_only_label.norm(dim=1, keepdim=True)
        self.text_feature_basic_sentence = self.text_feature_basic_sentence / self.text_feature_basic_sentence.norm(
            dim=1, keepdim=True)
        self.text_feature_detail_sentence = self.text_feature_detail_sentence / self.text_feature_detail_sentence.norm(
            dim=1, keepdim=True)
        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_only_label = logit_scale * image_features @ self.text_feature_only_label.t()
        logits_basic_sentence = logit_scale * image_features @ self.text_feature_basic_sentence.t()
        logits_detail_sentence = logit_scale * image_features @ self.text_feature_detail_sentence.t()
        logits_only_label = logits_only_label.view(batch_size, num_patch, -1)
        logits_basic_sentence = logits_basic_sentence.view(batch_size, num_patch, -1)
        logits_detail_sentence = logits_detail_sentence.view(batch_size, num_patch, -1)
        logits_only_label = (torch.mean(logits_only_label[:, :num_patch - 1, :], dim=1) + logits_only_label[:, num_patch - 1,
                                                             :]) / 2
        logits_basic_sentence = (torch.mean(logits_basic_sentence[:, :num_patch - 1, :], dim=1) + logits_basic_sentence[:, num_patch - 1,
                                                                :]) / 2
        logits_detail_sentence = (torch.mean(logits_detail_sentence[:, :num_patch - 1, :], dim=1) + logits_detail_sentence[:, num_patch - 1,
                                                                :]) / 2
        logits_final = logits_only_label * self.raw_weights[0] + logits_basic_sentence * self.raw_weights[1] + logits_detail_sentence * self.raw_weights[2]
        y_pred_CLIP = self.softmax(logits_final)
        return logits_final, self.softmax(logits_final)

    def forward(self, image):
        batch_size = image.size(0)
        num_patch = image.size(1)
        image = image.view(-1, image.size(2), image.size(3), image.size(4))
        image_feature = self.encode_image(image).to(torch.float32)
        image_features = image_feature / image_feature.norm(dim=1, keepdim=True)
        self.text_feature_only_label = self.text_feature_only_label / self.text_feature_only_label.norm(dim=1, keepdim=True)
        self.text_feature_basic_sentence = self.text_feature_basic_sentence / self.text_feature_basic_sentence.norm(
            dim=1, keepdim=True)
        self.text_feature_detail_sentence = self.text_feature_detail_sentence / self.text_feature_detail_sentence.norm(
            dim=1, keepdim=True)
        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_only_label = logit_scale * image_features @ self.text_feature_only_label.t()
        logits_basic_sentence = logit_scale * image_features @ self.text_feature_basic_sentence.t()
        logits_detail_sentence = logit_scale * image_features @ self.text_feature_detail_sentence.t()
        logits_only_label = logits_only_label.view(batch_size, num_patch, -1)
        logits_basic_sentence = logits_basic_sentence.view(batch_size, num_patch, -1)
        logits_detail_sentence = logits_detail_sentence.view(batch_size, num_patch, -1)
        logits_only_label = (torch.mean(logits_only_label[:, :num_patch - 1, :], dim=1) + logits_only_label[:, num_patch - 1,
                                                             :]) / 2
        logits_basic_sentence = (torch.mean(logits_basic_sentence[:, :num_patch - 1, :], dim=1) + logits_basic_sentence[:, num_patch - 1,
                                                                :]) / 2
        logits_detail_sentence = (torch.mean(logits_detail_sentence[:, :num_patch - 1, :], dim=1) + logits_detail_sentence[:, num_patch - 1,
                                                                :]) / 2
        logits_final = logits_only_label * self.raw_weights[0] + logits_basic_sentence * self.raw_weights[1] + logits_detail_sentence * self.raw_weights[2]
        y_pred_CLIP = self.softmax(logits_final)
        if self.args.model== "UNI_CLIP":
            y_pred_UNI_logits, y_pred_UNI = self.UNI(
                        image.view(batch_size, num_patch, -1, image.size(2), image.size(3)))
            loss_distillation = torch.norm(y_pred_CLIP - y_pred_UNI, 2)
            return logits_final, self.softmax(logits_final), loss_distillation
        return logits_final, self.softmax(logits_final)
