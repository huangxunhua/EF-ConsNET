
"""EF-ConsNET.py
运行示例：
  python EF-ConsNET.py \
    --pos_excel /path/pos.xlsx \
    --neg_excel /path/neg.xlsx \
    --image_root /path/images \
    --ecg_root /path/ecg \
    --ckpt /path/best.pth \
    --device cuda:0

说明：
- 目录约定：image_root/阳性, image_root/阴性；ecg_root/阳性, ecg_root/阴性
- Excel 约定：至少包含列：图片文件名、心电图文件名、性别、年龄、结果1..4、参考1..4、EF；阳性表还需“分类”(1轻/2重)
- 若不使用 ROI mask：--roi_parts none
"""

from __future__ import annotations

import argparse
import os
import pickle
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

from torchvision import transforms
from torchvision.models import (
    swin_t,
    Swin_T_Weights,
    resnet50,
    ResNet50_Weights,
    convnext_tiny,
    ConvNeXt_Tiny_Weights,
)

from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)


# =================== Utils ===================

def set_seed(seed: int = 2025) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_range(value: Any) -> Tuple[float, float]:
    """解析参考区间 '0.1-1.0' / '0.1～1.0' / '0.1—1.0'；失败返回 (nan,nan)"""
    try:
        s = str(value).strip().replace("～", "-").replace("—", "-")
        parts = s.split("-")
        if len(parts) == 2:
            lo, hi = float(parts[0]), float(parts[1])
            if hi > lo:
                return lo, hi
    except Exception:
        pass
    return float("nan"), float("nan")


def rin(x: Any, low: float, high: float, clip: float = 3.0) -> Tuple[float, int, int, float]:
    """Reference-Interval Normalization：将化验值映射到[-clip,clip]并给出方向/强度"""
    try:
        x = float(x)
    except Exception:
        return 0.0, 0, 0, 0.0
    if np.isnan(low) or np.isnan(high) or high <= low:
        return 0.0, 0, 0, 0.0
    mid = 0.5 * (low + high)
    half = 0.5 * (high - low)
    if half <= 0:
        return 0.0, 0, 0, 0.0
    s = (x - mid) / half
    s = float(np.clip(s, -clip, clip))
    return s, int(s > 0.999), int(s < -0.999), float(abs(s))


def clean_age_days_to_years(age_in_days: Any) -> float:
    """年龄以'天'为单位 -> 换算成年，并裁剪到 [0,120] 岁"""
    try:
        days = float(age_in_days)
    except Exception:
        return 0.0
    years = days / 365.25
    return float(np.clip(years, 0.0, 120.0))


def down_sample(signal: np.ndarray, old_fs: int = 500, new_fs: int = 100) -> np.ndarray:
    """线性插值重采样（保守、稳定）。输入 (C, T_old) -> 输出 (C, T_new)"""
    C, T = signal.shape
    t_old = np.arange(T) / float(old_fs)
    t_new = np.arange(0, t_old[-1], 1.0 / float(new_fs))
    out = np.zeros((C, len(t_new)), dtype=np.float32)
    for i in range(C):
        out[i] = np.interp(t_new, t_old, signal[i].astype(np.float32))
    return out


def _load_roi_mask_pil(
    img_path: str,
    y_main: int,
    image_root: str,
    roi_mask_pos_root: str,
    roi_mask_neg_root: str,
    roi_mask_suffix: str,
    placeholder_size: Tuple[int, int] = (512, 512),
) -> Image.Image:
    """根据 img_path 在 ROI mask 根目录中找对应 mask；找不到则返回全1 mask。"""

    subdir = "阳性" if int(y_main) == 1 else "阴性"
    roi_root = roi_mask_pos_root if int(y_main) == 1 else roi_mask_neg_root

    if roi_root is None or str(roi_root).strip() == "":
        return Image.new("L", placeholder_size, color=255)

    base_dir = os.path.join(str(image_root), subdir)

    try:
        rel = os.path.relpath(str(img_path), base_dir)
    except Exception:
        rel = os.path.basename(str(img_path))

    rel_dir = os.path.dirname(rel)
    stem = os.path.splitext(os.path.basename(rel))[0]
    mask_name = f"{stem}{roi_mask_suffix}.png"

    mask_path = os.path.join(str(roi_root), rel_dir, mask_name)
    if os.path.exists(mask_path):
        try:
            return Image.open(mask_path).convert("L")
        except Exception:
            return Image.new("L", placeholder_size, color=255)

    mask_path2 = os.path.join(str(roi_root), mask_name)
    if os.path.exists(mask_path2):
        try:
            return Image.open(mask_path2).convert("L")
        except Exception:
            return Image.new("L", placeholder_size, color=255)

    return Image.new("L", placeholder_size, color=255)


def stratified_split_by_main(labels_main: Sequence[int], val_ratio: float = 0.2, seed: int = 2025) -> Tuple[List[int], List[int]]:
    rng = np.random.default_rng(seed)
    labels_main = np.asarray(labels_main)
    idxs = np.arange(len(labels_main))
    tr_idx: List[int] = []
    va_idx: List[int] = []
    for c in np.unique(labels_main):
        cidx = idxs[labels_main == c]
        rng.shuffle(cidx)
        k = int(len(cidx) * val_ratio)
        va_idx += cidx[:k].tolist()
        tr_idx += cidx[k:].tolist()
    return tr_idx, va_idx


# =================== Model ===================


class SE1D(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.fc1 = nn.Linear(c, max(4, c // 4))
        self.fc2 = nn.Linear(max(4, c // 4), c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x:[B,C,T]
        s = x.mean(-1)  # [B,C]
        a = torch.sigmoid(self.fc2(torch.relu(self.fc1(s))))
        return x * a.unsqueeze(-1)


class ECGResNet1D(nn.Module):
    def __init__(self, in_ch: int = 12, mode: str = "resnet1d"):
        super().__init__()
        self.mode = mode

        if mode == "resnet1d":

            def block(ci: int, co: int, stride: int = 2) -> nn.Module:
                return nn.Sequential(
                    nn.Conv1d(ci, co, 7, stride, 3),
                    nn.BatchNorm1d(co),
                    nn.ReLU(),
                    nn.Conv1d(co, co, 3, 1, 1),
                    nn.BatchNorm1d(co),
                )

            self.stem = nn.Sequential(block(in_ch, 64, 2), nn.ReLU())
            self.b2 = nn.Sequential(block(64, 128, 2), nn.ReLU())
            self.b3 = nn.Sequential(block(128, 256, 2), nn.ReLU())
            self.b4 = nn.Sequential(block(256, 512, 2), nn.ReLU(), SE1D(512))
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.out_dim = 512

        elif mode == "inception1d":

            def inc_branch(ci: int, co: int, k: int) -> nn.Module:
                mid = max(8, co // 4)
                return nn.Sequential(
                    nn.Conv1d(ci, mid, 1, 1, 0),
                    nn.BatchNorm1d(mid),
                    nn.ReLU(),
                    nn.Conv1d(mid, co // 4, k, 1, k // 2),
                    nn.BatchNorm1d(co // 4),
                    nn.ReLU(),
                )

            class Inception1DBlock(nn.Module):
                def __init__(self, ci: int, co: int):
                    super().__init__()
                    self.b1 = inc_branch(ci, co, 3)
                    self.b2 = inc_branch(ci, co, 5)
                    self.b3 = inc_branch(ci, co, 7)
                    self.b4 = nn.Sequential(
                        nn.Conv1d(ci, co // 4, 1, 1, 0),
                        nn.BatchNorm1d(co // 4),
                        nn.ReLU(),
                    )
                    self.se = SE1D(co)

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    y = torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)
                    return self.se(y)

            def down(ci: int, co: int) -> nn.Module:
                return nn.Sequential(
                    nn.Conv1d(ci, co, 3, 2, 1),
                    nn.BatchNorm1d(co),
                    nn.ReLU(),
                )

            self.stem = nn.Sequential(
                nn.Conv1d(in_ch, 64, 7, 2, 3),
                nn.BatchNorm1d(64),
                nn.ReLU(),
            )
            self.b1 = nn.Sequential(Inception1DBlock(64, 128), down(128, 128))
            self.b2 = nn.Sequential(Inception1DBlock(128, 256), down(256, 256))
            self.b3 = nn.Sequential(Inception1DBlock(256, 512), down(512, 512))
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.out_dim = 512

        else:
            raise ValueError(f"Unsupported ECG backbone: {mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B,12,L]
        if self.mode == "resnet1d":
            x = self.stem(x)
            x = self.b2(x)
            x = self.b3(x)
            x = self.b4(x)
        else:
            x = self.stem(x)
            x = self.b1(x)
            x = self.b2(x)
            x = self.b3(x)
        return self.pool(x).squeeze(-1)


class DRBackbone(nn.Module):
    def __init__(self, backbone: str = 'swin_t'):
        super().__init__()
        self.kind = backbone
        if backbone == 'swin_t':
            m = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
            self.backbone = nn.Sequential(*(list(m.children())[:-1]))  # -> [B, 768]
            self.out_dim = 768
        elif backbone == 'resnet50':
            m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            self.backbone = nn.Sequential(*(list(m.children())[:-1]))  # -> [B, 2048, 1, 1]
            self.out_dim = 2048
        elif backbone == 'convnext_tiny':
            m = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
            self.backbone = nn.Sequential(m.features)  # 与 Test11 训练时一致：额外包一层 Sequential
            self.out_dim = 768
        else:
            raise ValueError(f"Unsupported image backbone: {backbone}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, 3, 512, 512]
        fmap = self.backbone(x)
        if isinstance(fmap, (list, tuple)):
            fmap = fmap[0]
        self._last_featmap = fmap
        if fmap.dim() == 4:
            h = F.adaptive_avg_pool2d(fmap, 1).flatten(1)
        else:
            h = fmap
        return h


class TabEncoder(nn.Module):
    """轻量 TabEncoder：Transformer 或 MLP，输出维持为 hid（默认 128）"""

    def __init__(self, in_dim: int, hid: int = 128, n_heads: int = 4, n_layers: int = 2, mode: str = "transformer"):
        super().__init__()
        self.mode = mode
        if mode == "transformer":
            self.proj = nn.Linear(in_dim, hid)
            enc_layer = nn.TransformerEncoderLayer(d_model=hid, nhead=n_heads, batch_first=True)
            self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
            self.cls = nn.Parameter(torch.randn(1, 1, hid))
            self.out_dim = hid
        elif mode == "mlp":
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, hid),
                nn.GELU(),
                nn.LayerNorm(hid),
                nn.Linear(hid, hid),
                nn.GELU(),
                nn.LayerNorm(hid),
            )
            self.out_dim = hid
        else:
            raise ValueError(f"Unsupported tab backbone: {mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B,D]
        if self.mode == "transformer":
            z = self.proj(x).unsqueeze(1)  # [B,1,H]
            cls = self.cls.expand(x.size(0), -1, -1)
            h = self.enc(torch.cat([cls, z], dim=1))[:, 0]
            return h
        return self.mlp(x)


class TriEncoderGPoE_BinSev(nn.Module):
    """三编码器 + Gated-PoE 融合（二分类） + 辅助严重度头（轻/重，仅阳性）"""

    def __init__(
        self,
        tab_in_dim: int,
        num_classes_main: int = 2,
        num_classes_sev: int = 2,
        image_backbone: str = "swin_t",
        ecg_backbone: str = "resnet1d",
        tab_backbone: str = "transformer",
    ):
        super().__init__()

        # ====== Confidence-gated fusion 超参（推理时保持默认）======
        self.conf_tau = 1.5
        self.beta_conf_ef = 0.1
        self.lambda_gate_entropy = 0.0

        self.img = DRBackbone(backbone=image_backbone)
        self.ecg = ECGResNet1D(mode=ecg_backbone)
        self.tab = TabEncoder(tab_in_dim, hid=128, mode=tab_backbone)

        self.img_head = nn.Linear(self.img.out_dim, num_classes_main)
        self.tab_head = nn.Linear(self.tab.out_dim, num_classes_main)
        self.ecg_head = nn.Linear(self.ecg.out_dim, num_classes_main)

        # ROI attention head（保留结构，推理阶段不使用也不影响）
        self.img_att_head = nn.LazyConv2d(out_channels=1, kernel_size=1, bias=False)

        # from TabHidden -> [g_img,g_tab,g_ecg]（保留，但最终 gate 使用置信度 softmax）
        self.gating = nn.Linear(self.tab.out_dim, 3)

        # 三塔 EF 回归头（推理阶段不参与 loss，但参与 gate 置信度）
        self.ef_head_tab = nn.Linear(self.tab.out_dim, 1)
        self.ef_head_ecg = nn.Linear(self.ecg.out_dim, 1)
        self.ef_head_img = nn.Linear(self.img.out_dim, 1)

        # EF bucket logits（仅缓存，不改变返回）
        self.ef_bucket_num_classes = 4
        self.ef_bucket_head_tab = nn.Linear(self.tab.out_dim, self.ef_bucket_num_classes)
        self.ef_bucket_head_ecg = nn.Linear(self.ecg.out_dim, self.ef_bucket_num_classes)
        self.ef_bucket_head_img = nn.Linear(self.img.out_dim, self.ef_bucket_num_classes)

        # 辅助严重度头
        self.sev_head = nn.Sequential(
            nn.Linear(self.img.out_dim + self.tab.out_dim + self.ecg.out_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes_sev),
        )

        self.register_buffer("uni_logp", torch.log(torch.ones(1, num_classes_main) / num_classes_main))

    def forward(
        self,
        x_img: torch.Tensor,
        x_tab: torch.Tensor,
        x_ecg: torch.Tensor,
        mask_img: Optional[torch.Tensor] = None,
        mask_ecg: Optional[torch.Tensor] = None,
        mask_tab: Optional[torch.Tensor] = None,
    ):
        h_tab = self.tab(x_tab)
        h_img = self.img(x_img)
        self._last_img_featmap = getattr(self.img, "_last_featmap", None)
        h_ecg = self.ecg(x_ecg)

        B = x_tab.size(0)
        if mask_img is None:
            mask_img = torch.ones(B, 1, device=h_tab.device)
        if mask_ecg is None:
            mask_ecg = torch.ones(B, 1, device=h_tab.device)
        if mask_tab is None:
            mask_tab = torch.ones(B, 1, device=h_tab.device)

        logp_img = torch.log_softmax(self.img_head(h_img), 1)
        logp_tab = torch.log_softmax(self.tab_head(h_tab), 1)
        logp_ecg = torch.log_softmax(self.ecg_head(h_ecg), 1)

        uni = self.uni_logp.expand(B, -1)
        logp_img_eff = torch.where(mask_img > 0.5, logp_img, uni)
        logp_ecg_eff = torch.where(mask_ecg > 0.5, logp_ecg, uni)
        logp_tab_eff = torch.where(mask_tab > 0.5, logp_tab, uni)

        ef_tab = self.ef_head_tab(h_tab)
        ef_ecg = self.ef_head_ecg(h_ecg)
        ef_img = self.ef_head_img(h_img)

        efb_tab = self.ef_bucket_head_tab(h_tab)
        efb_ecg = self.ef_bucket_head_ecg(h_ecg)
        efb_img = self.ef_bucket_head_img(h_img)
        self._last_ef_bucket_logits = (efb_tab, efb_ecg, efb_img)

        # gate = softmax(置信度)，置信度 = 分类置信 + EF一致性
        with torch.no_grad():
            tau = max(1e-6, float(self.conf_tau))
            p_img = torch.softmax(self.img_head(h_img) / tau, dim=1)
            p_tab = torch.softmax(self.tab_head(h_tab) / tau, dim=1)
            p_ecg = torch.softmax(self.ecg_head(h_ecg) / tau, dim=1)
            c_img_cls = p_img.max(dim=1, keepdim=True).values
            c_tab_cls = p_tab.max(dim=1, keepdim=True).values
            c_ecg_cls = p_ecg.max(dim=1, keepdim=True).values

            beta = float(self.beta_conf_ef)
            d_img = (ef_img - ef_tab).abs() + (ef_img - ef_ecg).abs()
            d_tab = (ef_tab - ef_img).abs() + (ef_tab - ef_ecg).abs()
            d_ecg = (ef_ecg - ef_img).abs() + (ef_ecg - ef_tab).abs()
            c_img_ef = torch.exp(-beta * d_img)
            c_tab_ef = torch.exp(-beta * d_tab)
            c_ecg_ef = torch.exp(-beta * d_ecg)

            c_img = 0.7 * c_img_cls + 0.3 * c_img_ef
            c_tab = 0.7 * c_tab_cls + 0.3 * c_tab_ef
            c_ecg = 0.7 * c_ecg_cls + 0.3 * c_ecg_ef

        c_img = c_img * mask_img
        c_tab = c_tab * mask_tab
        c_ecg = c_ecg * mask_ecg
        c_all = torch.cat([c_img, c_tab, c_ecg], dim=1) + 1e-8
        g = torch.softmax(c_all, dim=1)

        logtilde = g[:, 0:1] * logp_img_eff + g[:, 1:2] * logp_tab_eff + g[:, 2:3] * logp_ecg_eff
        y_logit_main = torch.log_softmax(logtilde, dim=1)

        h_img_eff = h_img * mask_img
        h_tab_eff = h_tab * mask_tab
        h_ecg_eff = h_ecg * mask_ecg
        fuse_feat = torch.cat([h_img_eff, h_tab_eff, h_ecg_eff], dim=1)
        y_logit_sev = self.sev_head(fuse_feat)

        return y_logit_main, y_logit_sev, (logp_img, logp_tab, logp_ecg, g, (ef_tab, ef_ecg, ef_img))


# =================== Dataset ===================


class FusionDataset(Dataset):
    """输出：

    image: [3,512,512]（缺失用占位图）
    tab:   [D]
    ecg:   [12,L]（缺失补零）
    modmask: [img_mask, ecg_mask, tab_mask]
    y_main: 0/1
    y_sev:  0/1（轻/重）；阴性为 -100

    返回 tuple：
      - 无 ROI: (image, tab, ecg, modmask, y_main, y_sev, ef_gt)
      - 有 ROI: (image, tab, ecg, modmask, roi_mask, y_main, y_sev, ef_gt)

    额外属性：
      tab_dim: 表格维度
    """

    def __init__(
        self,
        pos_excel: str,
        neg_excel: str,
        image_root: str,
        ecg_root: str,
        transform=None,
        target_len: int = 3000,
        tab_mode: str = "rin",
        roi_parts: str = "heart",
        roi_mask_pos_root: str = "/home/public/XinJiYan/DR_Seg/阳性",
        roi_mask_neg_root: str = "/home/public/XinJiYan/DR_Seg/阴性",
        roi_mask_heart_suffix: str = "_heartmask",
        roi_mask_lung_suffix: str = "_lungmask",
    ):
        self.transform = transform or transforms.Compose(
            [
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        self.image_root = image_root
        self.ecg_root = ecg_root
        self.target_len = int(target_len)
        self.placeholder_size = (512, 512)
        self.tab_mode = str(tab_mode).lower()
        self.samples: List[Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, float, float, int, int]] = []

        # ROI mask 配置
        self.roi_parts = str(roi_parts).lower() if roi_parts is not None else "none"
        self.roi_mask_pos_root = roi_mask_pos_root
        self.roi_mask_neg_root = roi_mask_neg_root
        self.roi_mask_heart_suffix = roi_mask_heart_suffix
        self.roi_mask_lung_suffix = roi_mask_lung_suffix
        self._mask_transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])

        pos_df = pd.read_excel(pos_excel)
        neg_df = pd.read_excel(neg_excel)

        def sev_map(v: Any) -> int:
            try:
                v = int(v)
                return 0 if v == 1 else 1
            except Exception:
                return 1

        pos_df["y_main"] = 1
        pos_df["y_sev"] = pos_df["分类"].apply(sev_map)
        neg_df["y_main"] = 0
        neg_df["y_sev"] = -100

        full_df = pd.concat([pos_df, neg_df], ignore_index=True)

        # tab 维度
        if self.tab_mode == "rin":
            # rin_feats(4) + rin_abs(4) + rin_flags(8) + rin_miss(4) + sex age injury miss_count(4)
            self.tab_dim = 24
            self._raw_meds = None
            self._raw_iqrs = None
        else:
            # raw: 4 labs + sex + age
            self.tab_dim = 6
            meds, iqrs = [], []
            for i in range(1, 5):
                col = pd.to_numeric(full_df.get(f"结果{i}", pd.Series([np.nan] * len(full_df))), errors="coerce")
                m = np.nanmedian(col.values)
                q75 = np.nanpercentile(col.values, 75)
                q25 = np.nanpercentile(col.values, 25)
                iqr = max(1e-6, float(q75 - q25))
                meds.append(float(m))
                iqrs.append(float(iqr))
            self._raw_meds = np.asarray(meds, dtype=np.float32)
            self._raw_iqrs = np.asarray(iqrs, dtype=np.float32)

        for _, row in full_df.iterrows():
            y_main = int(row["y_main"])
            y_sev = int(row["y_sev"])

            img_path = os.path.join(self.image_root, "阳性" if y_main == 1 else "阴性", str(row["图片文件名"]))
            ecg_path = os.path.join(self.ecg_root, "阳性" if y_main == 1 else "阴性", str(row["心电图文件名"]) + ".pickle")

            if self.tab_mode == "rin":
                rin_feats, rin_abs, rin_flags, rin_miss = [], [], [], []
                for i in range(1, 5):
                    low, high = parse_range(row.get(f"参考{i}", ""))
                    val = row.get(f"结果{i}", np.nan)
                    s, hi, lo, mag = rin(val, low, high)
                    rin_feats.append(s)
                    rin_abs.append(mag)
                    rin_flags.extend([hi, lo])
                    rin_miss.append(1 if pd.isna(val) or np.isnan(low) or np.isnan(high) else 0)

                sex_raw = row.get("性别", np.nan)
                sex = 1.0 if str(sex_raw).strip() in ["1", "男", "male", "M", "m"] else 0.0
                age_years = clean_age_days_to_years(row.get("年龄", np.nan))

                ef_raw = row.get("EF", np.nan)
                if pd.isna(ef_raw):
                    ef_gt = 0.0
                else:
                    try:
                        ef_gt = float(np.clip(float(ef_raw), 10.0, 85.0))
                    except Exception:
                        ef_gt = 0.0

                injury = max(abs(rin_feats[0]), abs(rin_feats[1]), abs(rin_feats[2]), abs(rin_feats[3]))
                miss_count = float(sum(rin_miss))
                tab_vec = rin_feats + rin_abs + rin_flags + rin_miss + [sex, age_years, injury, miss_count]

            else:
                labs = []
                for i in range(1, 5):
                    v = row.get(f"结果{i}", np.nan)
                    try:
                        v = float(v)
                    except Exception:
                        v = np.nan
                    m = float(self._raw_meds[i - 1])
                    s = float(self._raw_iqrs[i - 1])
                    if np.isnan(v):
                        z = 0.0
                    else:
                        z = float(np.clip((v - m) / s, -5.0, 5.0))
                    labs.append(z)

                sex_raw = row.get("性别", np.nan)
                sex = 1.0 if str(sex_raw).strip() in ["1", "男", "male", "M", "m"] else 0.0
                age_years = clean_age_days_to_years(row.get("年龄", np.nan))

                ef_raw = row.get("EF", np.nan)
                try:
                    ef_gt = float(np.clip(float(ef_raw), 10.0, 85.0))
                except Exception:
                    ef_gt = 0.0

                tab_vec = labs + [sex, age_years]

            tab_tensor = torch.tensor(tab_vec, dtype=torch.float32)
            ef_tensor = torch.tensor(float(ef_gt), dtype=torch.float32)

            # ECG
            expected_leads = 12
            ecg_mask = 1.0
            try:
                if not os.path.exists(ecg_path):
                    raise FileNotFoundError(ecg_path)
                data = pickle.load(open(ecg_path, "rb"))
                signal = data[0] if isinstance(data, (list, tuple)) else data  # (C,T)
                signal = down_sample(signal)

                if signal.shape[1] < self.target_len:
                    signal = np.pad(signal, ((0, 0), (0, self.target_len - signal.shape[1])), "constant")
                else:
                    signal = signal[:, : self.target_len]

                C, T = signal.shape
                if C < expected_leads:
                    pad = np.zeros((expected_leads - C, T), dtype=signal.dtype)
                    signal = np.vstack([signal, pad])
                elif C > expected_leads:
                    signal = signal[:expected_leads, :]

                ecg_tensor = torch.tensor(signal, dtype=torch.float32)
            except Exception:
                ecg_tensor = torch.zeros((expected_leads, self.target_len), dtype=torch.float32)
                ecg_mask = 0.0

            img_mask = 1.0 if os.path.exists(img_path) else 0.0

            self.samples.append((img_path, tab_tensor, ecg_tensor, ef_tensor, float(img_mask), float(ecg_mask), y_main, y_sev))

    def __len__(self) -> int:
        return len(self.samples)

    def _placeholder(self) -> Image.Image:
        w, h = self.placeholder_size
        return Image.new("RGB", (w, h), color=(0, 0, 0))

    def __getitem__(self, idx: int):
        img_path, tab, ecg, ef_gt, img_mask, ecg_mask, y_main, y_sev = self.samples[idx]

        if img_mask < 0.5:
            image = self._placeholder()
        else:
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception:
                image = self._placeholder()
                img_mask = 0.0

        image = self.transform(image)

        # ROI mask
        roi_mask = None
        parts = getattr(self, "roi_parts", "heart")
        if parts in ("none", "null", ""):
            roi_mask = None
        else:
            masks = []
            if parts in ("heart", "heart+lung"):
                mimg_h = _load_roi_mask_pil(
                    img_path=img_path,
                    y_main=int(y_main),
                    image_root=self.image_root,
                    roi_mask_pos_root=self.roi_mask_pos_root,
                    roi_mask_neg_root=self.roi_mask_neg_root,
                    roi_mask_suffix=self.roi_mask_heart_suffix,
                    placeholder_size=getattr(self, "placeholder_size", (512, 512)),
                )
                mh = self._mask_transform(mimg_h)
                mh = (mh > 0.5).float()
                masks.append(mh)

            if parts in ("lung", "heart+lung"):
                mimg_l = _load_roi_mask_pil(
                    img_path=img_path,
                    y_main=int(y_main),
                    image_root=self.image_root,
                    roi_mask_pos_root=self.roi_mask_pos_root,
                    roi_mask_neg_root=self.roi_mask_neg_root,
                    roi_mask_suffix=self.roi_mask_lung_suffix,
                    placeholder_size=getattr(self, "placeholder_size", (512, 512)),
                )
                ml = self._mask_transform(mimg_l)
                ml = (ml > 0.5).float()
                masks.append(ml)

            if len(masks) == 1:
                roi_mask = masks[0]
            elif len(masks) >= 2:
                roi_mask = torch.clamp(masks[0] + masks[1], 0.0, 1.0)

        tab_mask = 1.0
        modmask = torch.tensor([img_mask, ecg_mask, tab_mask], dtype=torch.float32)

        if roi_mask is None:
            return image, tab, ecg, modmask, int(y_main), int(y_sev), ef_gt
        return image, tab, ecg, modmask, roi_mask, int(y_main), int(y_sev), ef_gt


# =================== Eval ===================


@dataclass
class TestConfig:
    single_modality: str = "all"  # 'all' | 'image' | 'ecg' | 'tab'
    thr_min: float = 0.10
    thr_max: float = 0.90
    thr_step: float = 0.01
    sev_thr: float = 0.50


def _unpack_batch(batch: Sequence[Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """支持两种 batch：

    1) (img, tab, ecg, modmask, y_main, y_sev, ef_gt)
    2) (img, tab, ecg, modmask, roi_mask, y_main, y_sev, ef_gt)
    """
    if not isinstance(batch, (list, tuple)):
        raise TypeError(f"Batch must be a tuple/list, got: {type(batch)}")
    if len(batch) < 6:
        raise ValueError(f"Unexpected batch length: {len(batch)}")

    if len(batch) >= 8:
        img, tab, ecg, modmask, y_main, y_sev = batch[0], batch[1], batch[2], batch[3], batch[5], batch[6]
        return img, tab, ecg, modmask, y_main, y_sev

    img, tab, ecg, modmask, y_main, y_sev = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5]
    return img, tab, ecg, modmask, y_main, y_sev


def _apply_single_modality(
    img: torch.Tensor,
    tab: torch.Tensor,
    ecg: torch.Tensor,
    modmask: torch.Tensor,
    single_modality: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if single_modality == "all":
        return img, tab, ecg, modmask

    modmask = modmask.clone()

    if single_modality == "image":
        ecg = ecg * 0.0
        tab = tab * 0.0
        modmask[:, 1:2] = 0.0
        modmask[:, 2:3] = 0.0
        return img, tab, ecg, modmask

    if single_modality == "ecg":
        img = img * 0.0
        tab = tab * 0.0
        modmask[:, 0:1] = 0.0
        modmask[:, 2:3] = 0.0
        return img, tab, ecg, modmask

    if single_modality == "tab":
        img = img * 0.0
        ecg = ecg * 0.0
        modmask[:, 0:1] = 0.0
        modmask[:, 1:2] = 0.0
        return img, tab, ecg, modmask

    raise ValueError("single_modality must be one of ['all','image','ecg','tab']")


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: Iterable[Sequence[Any]],
    device: Union[str, torch.device],
    cfg: Optional[TestConfig] = None,
    name: str = "normal",
) -> Dict[str, float]:
    if cfg is None:
        cfg = TestConfig()

    device = torch.device(device)
    model.eval()

    y_main_t: List[int] = []
    y_main_prob: List[float] = []
    y_sev_t: List[int] = []
    y_sev_p: List[int] = []
    y_sev_prob: List[float] = []

    for batch in loader:
        img, tab, ecg, modmask, y_main, y_sev = _unpack_batch(batch)

        img = img.to(device)
        tab = tab.to(device)
        ecg = ecg.to(device)
        modmask = modmask.to(device)

        img, tab, ecg, modmask = _apply_single_modality(img, tab, ecg, modmask, cfg.single_modality)

        img_m = modmask[:, 0:1]
        ecg_m = modmask[:, 1:2]
        tab_m = modmask[:, 2:3]

        logit_main, logit_sev, _ = model(img, tab, ecg, img_m, ecg_m, tab_m)

        prob_main = torch.softmax(logit_main, dim=1)[:, 1].detach().cpu().numpy()
        y_main_t.extend(y_main.detach().cpu().numpy().astype(int).tolist())
        y_main_prob.extend(prob_main.astype(float).tolist())

        ym = y_main.detach().cpu().numpy()
        ys = y_sev.detach().cpu().numpy()
        pos_mask = (ym == 1) & (ys >= 0)
        if logit_sev is not None and np.any(pos_mask):
            sev_probs = torch.softmax(logit_sev[pos_mask], dim=1)[:, 1].detach().cpu().numpy()
            sev_preds = (sev_probs >= cfg.sev_thr).astype(int)
            y_sev_t.extend(ys[pos_mask].astype(int).tolist())
            y_sev_p.extend(sev_preds.astype(int).tolist())
            y_sev_prob.extend(sev_probs.astype(float).tolist())

    y_main_t_arr = np.asarray(y_main_t, dtype=np.int64)
    y_main_prob_arr = np.asarray(y_main_prob, dtype=np.float32)

    # 阈值调优
    best_thr, best_f1, best_p, best_r = 0.5, -1.0, 0.0, 0.0
    for thr in np.arange(cfg.thr_min, cfg.thr_max + 1e-9, cfg.thr_step, dtype=np.float32):
        pred = (y_main_prob_arr >= float(thr)).astype(np.int64)
        p, r, f1, _ = precision_recall_fscore_support(y_main_t_arr, pred, average="binary", zero_division=0)
        if f1 > best_f1:
            best_thr, best_f1, best_p, best_r = float(thr), float(f1), float(p), float(r)

    print(f"\n P={best_p:.4f} | R={best_r:.4f} | F1={best_f1:.4f}")

    y_main_pred = (y_main_prob_arr >= best_thr).astype(np.int64)

    cm_main = confusion_matrix(y_main_t_arr, y_main_pred, labels=[0, 1])
    tn, fp, fn, tp = cm_main.ravel()
    spe_main = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    ppv_main = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    npv_main = (tn / (tn + fn)) if (tn + fn) > 0 else 0.0

    try:
        from sklearn.metrics import precision_recall_curve, auc

        auc_main = roc_auc_score(y_main_t_arr, y_main_prob_arr)
        ap_main = average_precision_score(y_main_t_arr, y_main_prob_arr)
        p_curve, r_curve, _ = precision_recall_curve(y_main_t_arr, y_main_prob_arr)
        auprc_main = auc(r_curve, p_curve)
    except Exception:
        auc_main, ap_main, auprc_main = float("nan"), float("nan"), float("nan")

    print(f"\n==[{name}] 主任务（二分类）==")
    print(
        f"Main | Precision: {best_p:.4f} | Recall: {best_r:.4f} | F1: {best_f1:.4f} | "
        f"AUROC: {auc_main:.4f} | AP: {ap_main:.4f} | AUPRC: {auprc_main:.4f} | "
        f"SPE: {spe_main:.4f} | PPV: {ppv_main:.4f} | NPV: {npv_main:.4f}"
    )

    sev_metrics = {
        "Sev_P": float("nan"),
        "Sev_R": float("nan"),
        "Sev_F1": float("nan"),
        "Sev_AUROC": float("nan"),
        "Sev_AP": float("nan"),
        "Sev_AUPRC": float("nan"),
        "Sev_SPE": float("nan"),
        "Sev_PPV": float("nan"),
        "Sev_NPV": float("nan"),
    }

    if len(y_sev_t) > 0:
        y_sev_t_arr = np.asarray(y_sev_t, dtype=np.int64)
        y_sev_p_arr = np.asarray(y_sev_p, dtype=np.int64)
        y_sev_prob_arr = np.asarray(y_sev_prob, dtype=np.float32)

        cm_sev = confusion_matrix(y_sev_t_arr, y_sev_p_arr, labels=[0, 1])
        tn_s, fp_s, fn_s, tp_s = cm_sev.ravel()
        spe_sev = (tn_s / (tn_s + fp_s)) if (tn_s + fp_s) > 0 else 0.0
        ppv_sev = (tp_s / (tp_s + fp_s)) if (tp_s + fp_s) > 0 else 0.0
        npv_sev = (tn_s / (tn_s + fn_s)) if (tn_s + fn_s) > 0 else 0.0

        try:
            from sklearn.metrics import precision_recall_curve, auc

            auc_sev = roc_auc_score(y_sev_t_arr, y_sev_prob_arr)
            ap_sev = average_precision_score(y_sev_t_arr, y_sev_prob_arr)
            p_curve_s, r_curve_s, _ = precision_recall_curve(y_sev_t_arr, y_sev_prob_arr)
            auprc_sev = auc(r_curve_s, p_curve_s)
        except Exception:
            auc_sev, ap_sev, auprc_sev = float("nan"), float("nan"), float("nan")

        p_sev, r_sev, f_sev, _ = precision_recall_fscore_support(y_sev_t_arr, y_sev_p_arr, average="binary", zero_division=0)

        print(f"\n==[{name}] 辅助严重度（轻/重，仅阳性")
        print(
            f"Sev | Precision: {p_sev:.4f} | Recall: {r_sev:.4f} | F1: {f_sev:.4f} | "
            f"AUROC: {auc_sev:.4f} | AP: {ap_sev:.4f} | AUPRC: {auprc_sev:.4f} | "
            f"SPE: {spe_sev:.4f} | PPV: {ppv_sev:.4f} | NPV: {npv_sev:.4f}"
        )

        sev_metrics = {
            "Sev_P": float(p_sev),
            "Sev_R": float(r_sev),
            "Sev_F1": float(f_sev),
            "Sev_AUROC": float(auc_sev),
            "Sev_AP": float(ap_sev),
            "Sev_AUPRC": float(auprc_sev),
            "Sev_SPE": float(spe_sev),
            "Sev_PPV": float(ppv_sev),
            "Sev_NPV": float(npv_sev),
        }
    else:
        print("\n[提示] 验证集中没有阳性样本，无法评估严重度头。")

    return {
        "Main_Thr": float(best_thr),
        "Main_P": float(best_p),
        "Main_R": float(best_r),
        "Main_F1": float(best_f1),
        "Main_AUROC": float(auc_main),
        "Main_AP": float(ap_main),
        "Main_AUPRC": float(auprc_main),
        "Main_SPE": float(spe_main),
        "Main_PPV": float(ppv_main),
        "Main_NPV": float(npv_main),
        **sev_metrics,
    }


# =================== Main ===================


def infer_image_backbone_from_state_dict(state: Dict[str, torch.Tensor]) -> Optional[str]:
    """从 ckpt 的 key 粗略推断 image backbone，避免大量 missing/unexpected 导致性能异常。"""
    keys = list(state.keys())
    # Swin 特征：相对位置偏置、attn 结构
    if any("relative_position_bias_table" in k or ".attn." in k for k in keys):
        return "swin_t"
    # ConvNeXt 特征：layer_scale、stages blocks
    if any("layer_scale" in k or ".block." in k for k in keys):
        return "convnext_tiny"
    # ResNet 特征：layer1/2/3/4
    if any("layer1." in k or "layer2." in k or "layer3." in k or "layer4." in k for k in keys):
        return "resnet50"
    return None



# === Checkpoint loading: strict (Test11 style) ===
def _load_checkpoint(model: nn.Module, ckpt_path: str, device: torch.device) -> None:
    """按 Test11 的方式加载：ckpt 必须与模型结构完全一致。"""
    obj = torch.load(ckpt_path, map_location=device)

    # 兼容保存为 {'state_dict':...} 或 {'model':...}
    if isinstance(obj, dict) and "state_dict" in obj:
        state = obj["state_dict"]
    elif isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        state = obj["model"]
    else:
        state = obj

    # 兼容 DDP: 去掉 'module.'
    clean_state: Dict[str, torch.Tensor] = {}
    if isinstance(state, dict):
        for k, v in state.items():
            if k.startswith("module."):
                clean_state[k[len("module.") :]] = v
            else:
                clean_state[k] = v
    else:
        raise ValueError("Checkpoint is not a state_dict dict")

    # 关键：strict=True，确保与 Test11 一致（任何不一致都立刻暴露）
    model.load_state_dict(clean_state, strict=True)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--pos_excel", type=str, required=True)
    p.add_argument("--neg_excel", type=str, required=True)
    p.add_argument("--image_root", type=str, required=True)
    p.add_argument("--ecg_root", type=str, required=True)

    p.add_argument("--tab_mode", type=str, default="rin", choices=["rin", "raw"])

    # roi
    p.add_argument("--roi_parts", type=str, default="none", help="none|heart|lung|heart+lung")
    p.add_argument("--roi_mask_pos_root", type=str, default="")
    p.add_argument("--roi_mask_neg_root", type=str, default="")
    p.add_argument("--roi_mask_heart_suffix", type=str, default="_heartmask")
    p.add_argument("--roi_mask_lung_suffix", type=str, default="_lungmask")

    # loader
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=2025)

    # model
    p.add_argument("--ckpt", type=str, default='')
    p.add_argument("--image_backbone", type=str, default="convnext_tiny", choices=["swin_t", "resnet50", "convnext_tiny"])
    p.add_argument("--ecg_backbone", type=str, default="inception1d", choices=["resnet1d", "inception1d"])
    p.add_argument("--tab_backbone", type=str, default="mlp", choices=["transformer", "mlp"])
    # --no_infer_backbone argument removed

    # eval
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--single_modality", type=str, default="all", choices=["all", "image", "ecg", "tab"])
    p.add_argument("--thr_min", type=float, default=0.10)
    p.add_argument("--thr_max", type=float, default=0.90)
    p.add_argument("--thr_step", type=float, default=0.01)
    p.add_argument("--sev_thr", type=float, default=0.50)

    return p


def main() -> None:
    args = build_argparser().parse_args()
    set_seed(int(args.seed))

    device = torch.device(args.device)

    dataset = FusionDataset(
        pos_excel=args.pos_excel,
        neg_excel=args.neg_excel,
        image_root=args.image_root,
        ecg_root=args.ecg_root,
        tab_mode=args.tab_mode,
        roi_parts=args.roi_parts,
        roi_mask_pos_root=args.roi_mask_pos_root,
        roi_mask_neg_root=args.roi_mask_neg_root,
        roi_mask_heart_suffix=args.roi_mask_heart_suffix,
        roi_mask_lung_suffix=args.roi_mask_lung_suffix,
    )

    labels_main = [int(s[-2]) for s in dataset.samples]  # (.., y_main, y_sev)
    tr_idx, va_idx = stratified_split_by_main(labels_main, val_ratio=float(args.val_ratio), seed=int(args.seed))

    val_set = Subset(dataset, va_idx)
    val_loader = DataLoader(
        val_set,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(args.ckpt)
    print(f"[Load] ckpt: {args.ckpt}")

    model = TriEncoderGPoE_BinSev(
        tab_in_dim=int(dataset.tab_dim),
        num_classes_main=2,
        num_classes_sev=2,
        image_backbone=args.image_backbone,
        ecg_backbone=args.ecg_backbone,
        tab_backbone=args.tab_backbone,
    ).to(device)
    print(f"[Model] image_backbone={args.image_backbone} | ecg_backbone={args.ecg_backbone} | tab_backbone={args.tab_backbone}")

    # 加载 checkpoint (strict, 按 Test11 方式)
    _load_checkpoint(model, args.ckpt, device=device)

    cfg = TestConfig(
        single_modality=args.single_modality,
        thr_min=float(args.thr_min),
        thr_max=float(args.thr_max),
        thr_step=float(args.thr_step),
        sev_thr=float(args.sev_thr),
    )

    metrics = evaluate(model, val_loader, device=device, cfg=cfg, name="val")
    print("\n[Done] Metrics:")
    for k in sorted(metrics.keys()):
        v = metrics[k]
        print(f"  {k}: {v:.6f}" if isinstance(v, float) and np.isfinite(v) else f"  {k}: {v}")


if __name__ == "__main__":
    main()