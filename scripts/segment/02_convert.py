import argparse
import time
import types
from pathlib import Path

import onnx
import onnxslim
import torch
import torch.nn.functional as F
from ultralytics.models.sam.build import build_sam2_b, build_sam2_s, build_sam2_t
from ultralytics.models.sam.modules.sam import SAM2Model
from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.export.engine import best_onnx_opset
from ultralytics.utils.export.mnn import onnx2mnn
from ultralytics.utils.files import file_size

OPSET = best_onnx_opset(onnx)

sam2_model_map = {"sam2.1_t.pt": build_sam2_t, "sam2.1_s.pt": build_sam2_s, "sam2.1_b.pt": build_sam2_b}


def _embed_points_onnx(self, points: torch.Tensor, labels: torch.Tensor, pad: bool) -> torch.Tensor:
    """Embed point prompts by applying positional encoding and label-specific embeddings."""
    points = points + 0.5  # Shift to center of pixel
    if pad:
        padding_point = torch.zeros((points.shape[0], 1, 2), dtype=points.dtype, device=points.device)
        padding_label = -torch.ones((labels.shape[0], 1), dtype=labels.dtype, device=labels.device)
        points = torch.cat([points, padding_point], dim=1)
        labels = torch.cat([labels, padding_label], dim=1)
    points = points / self.input_image_size[0]
    point_embedding = self.pe_layer._pe_encoding(points)

    labels_expanded = labels.unsqueeze(-1).expand_as(point_embedding)
    point_embedding = point_embedding * (labels_expanded != -1).to(point_embedding.dtype)
    point_embedding = point_embedding + self.not_a_point_embed.weight * (labels_expanded == -1).to(point_embedding.dtype)
    for k in range(self.num_point_embeddings):
        point_embedding = point_embedding + self.point_embeddings[k].weight * (labels_expanded == k).to(point_embedding.dtype)
    return point_embedding


class Encoder(torch.nn.Module):
    def __init__(self, model: SAM2Model):
        super().__init__()
        self.no_mem_embed = model.no_mem_embed
        self.image_encoder = model.image_encoder
        self.mask_decoder = model.sam_mask_decoder

    @torch.no_grad()
    def forward(self, image: torch.Tensor):
        backbone_out = self.image_encoder(image)
        high_res_feats_0 = self.mask_decoder.conv_s0(backbone_out["backbone_fpn"][0])
        high_res_feats_1 = self.mask_decoder.conv_s1(backbone_out["backbone_fpn"][1])

        image_embed_flatten = backbone_out["backbone_fpn"][2].flatten(2).permute(2, 0, 1)  # flatten NxCxHxW to HWxNxC
        image_embed_flatten = image_embed_flatten + self.no_mem_embed
        image_embed = image_embed_flatten.permute(1, 2, 0).view(1, -1, 64, 64)

        return image_embed, high_res_feats_0, high_res_feats_1


class Decoder(torch.nn.Module):
    def __init__(self, model: SAM2Model):
        super().__init__()
        self.prompt_encoder = model.sam_prompt_encoder
        self.mask_decoder = model.sam_mask_decoder

    @torch.no_grad()
    def forward(
        self,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        image_embed: torch.Tensor,
        high_res_feats_0: torch.Tensor,
        high_res_feats_1: torch.Tensor,
    ):
        sparse_embeddings, dense_embeddings = self.prompt_encoder(points=(point_coords, point_labels), boxes=None, masks=None)
        pred_mask, _, _, _ = self.mask_decoder(
            image_embeddings=image_embed,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=[high_res_feats_0, high_res_feats_1],
        )
        mask = F.interpolate(pred_mask, self.prompt_encoder.input_image_size, mode="bilinear", align_corners=False)
        return mask


def export_encoder(model: Encoder, abspath_stem: str, quantize: int):
    image = torch.randn(1, 3, 1024, 1024)
    # _ = model(image)

    onnx_path = abspath_stem + "_enc.onnx"
    LOGGER.info(f"\n{colorstr("ONNX:")} starting export with onnx {onnx.__version__} opset {OPSET}...")
    t0 = time.time()
    torch.onnx.export(
        model,
        image,
        onnx_path,
        opset_version=OPSET,
        external_data=False,
        input_names=["image"],
        output_names=["image_embed", "high_res_feats_0", "high_res_feats_1"],
    )
    onnx_model = onnx.load(onnx_path)
    LOGGER.info(f"{colorstr("ONNX:")} slimming with onnxslim {onnxslim.__version__}...")
    onnx_model = onnxslim.slim(onnx_model)
    onnx.save(onnx_model, onnx_path)
    t1 = time.time()
    mb = file_size(onnx_path)
    assert mb > 0.0, "0.0 MB output model size"
    LOGGER.info(f"{colorstr("ONNX:")} export success ✅ {(t1 - t0):.1f}s, saved as '{onnx_path}' ({mb:.1f} MB)")

    t2 = time.time()
    mnn_path = abspath_stem + "_enc.mnn"
    onnx2mnn(onnx_path, mnn_path, quantize, "biz", colorstr("MNN:"))
    t3 = time.time()
    mb = file_size(mnn_path)
    assert mb > 0.0, "0.0 MB output model size"
    LOGGER.info(f"{colorstr("MNN:")} export success ✅ {(t3 - t2):.1f}s, saved as '{mnn_path}' ({mb:.1f} MB)")

    return mnn_path


def export_decoder(model: Decoder, abspath_stem: str, quantize: int):
    point_coords = torch.rand((1, 6, 2))  # [1, num_points, 2]
    point_labels = torch.randint(4, (1, 6), dtype=torch.float)  # [1, num_points]
    image_embed = torch.randn((1, 256, 64, 64))
    high_res_feats_0 = torch.randn((1, 32, 256, 256))
    high_res_feats_1 = torch.randn((1, 64, 128, 128))
    # _ = model(point_coords, point_labels, image_embed, high_res_feats_0, high_res_feats_1)
    model.prompt_encoder._embed_points = types.MethodType(_embed_points_onnx, model.prompt_encoder)

    onnx_path = abspath_stem + "_dec.onnx"
    LOGGER.info(f"\n{colorstr("ONNX:")} starting export with onnx {onnx.__version__} opset {OPSET}...")
    t0 = time.time()
    num_points = torch.export.Dim("num_points", min=1, max=8)
    torch.onnx.export(
        model,
        (point_coords, point_labels, image_embed, high_res_feats_0, high_res_feats_1),
        onnx_path,
        opset_version=OPSET,
        external_data=False,
        input_names=["point_coords", "point_labels", "image_embed", "high_res_feats_0", "high_res_feats_1"],
        output_names=["mask"],
        dynamic_shapes={
            "point_coords": {1: num_points},
            "point_labels": {1: num_points},
            "image_embed": None,
            "high_res_feats_0": None,
            "high_res_feats_1": None,
        },
    )
    onnx_model = onnx.load(onnx_path)
    LOGGER.info(f"{colorstr("ONNX:")} slimming with onnxslim {onnxslim.__version__}...")
    onnx_model = onnxslim.slim(onnx_model)
    onnx.save(onnx_model, onnx_path)
    t1 = time.time()
    mb = file_size(onnx_path)
    assert mb > 0.0, "0.0 MB output model size"
    LOGGER.info(f"{colorstr("ONNX:")} export success ✅ {(t1 - t0):.1f}s, saved as '{onnx_path}' ({mb:.1f} MB)")

    t2 = time.time()
    mnn_path = abspath_stem + "_dec.mnn"
    onnx2mnn(onnx_path, mnn_path, quantize, "biz", colorstr("MNN:"))
    t3 = time.time()
    mb = file_size(mnn_path)
    assert mb > 0.0, "0.0 MB output model size"
    LOGGER.info(f"{colorstr("MNN:")} export success ✅ {(t3 - t2):.1f}s, saved as '{mnn_path}' ({mb:.1f} MB)")

    return mnn_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/sam2.1_b.pt", help="The model file for training")
    parser.add_argument("--quantize", type=int, default=32, help="Export precision. e.g. 32(FP32), 16(FP16) or 8(INT8)")
    args = parser.parse_args()

    for k in sam2_model_map:
        if str(args.model).endswith(k):
            sam2_builder = sam2_model_map.get(k)
    if not sam2_builder:
        raise FileNotFoundError(f"{args.model} is not a supported model. Available models are: \n {sam2_model_map.keys()}")

    t0 = time.time()
    model = sam2_builder(args.model)
    abspath_stem = str(Path(args.model).with_suffix(""))
    enc_path = export_encoder(Encoder(model).eval(), abspath_stem, args.quantize)
    dec_path = export_decoder(Decoder(model).eval(), abspath_stem, args.quantize)
    t1 = time.time()
    print(f"\nExport complete ({(t1 - t0):.1f}s)\nResults saved to {Path(enc_path).absolute()}\n                 {Path(dec_path).absolute()}")
