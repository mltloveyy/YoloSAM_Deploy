import argparse
import time
from pathlib import Path

import onnx
import onnxslim
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.models.sam.build import build_sam2_b
from ultralytics.models.sam.modules.sam import SAM2Model
from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.export.engine import best_onnx_opset
from ultralytics.utils.export.mnn import onnx2mnn
from ultralytics.utils.files import file_size

OPSET = best_onnx_opset(onnx)


class Encoder(torch.nn.Module):
    def __init__(self, model: SAM2Model):
        super().__init__()
        self.directly_add_no_mem_embed = model.directly_add_no_mem_embed
        self.no_mem_embed = model.no_mem_embed  # [1,1,256]
        self.image_encoder = model.image_encoder
        self.mask_decoder = model.sam_mask_decoder
        self.num_feature_levels = model.num_feature_levels
        self._bb_feat_sizes = [(256, 256), (128, 128), (64, 64)]

    @torch.no_grad()
    def forward(self, image: torch.Tensor):
        backbone_out = self.image_encoder(image)  # {"vision_features","vision_pos_enc","backbone_fpn"}
        backbone_out["backbone_fpn"][0] = self.mask_decoder.conv_s0(backbone_out["backbone_fpn"][0])
        backbone_out["backbone_fpn"][1] = self.mask_decoder.conv_s1(backbone_out["backbone_fpn"][1])

        feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels :]
        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]  # flatten NxCxHxW to HWxNxC
        if self.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.no_mem_embed
        feats = [feat.permute(1, 2, 0).view(1, -1, *feat_size) for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])][::-1]

        image_embed = feats[-1]
        high_res_feats_0 = feats[0]
        high_res_feats_1 = feats[1]
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
        mask = F.interpolate(pred_mask, (1024, 1024), mode="bilinear", align_corners=False)
        return mask


def export_encoder(model, abspath_stem, half: bool, int8: bool):
    LOGGER.info(f"\n{colorstr("ONNX:")} starting export with onnx {onnx.__version__} opset {OPSET}...")
    image = torch.randn(1, 3, 1024, 1024)
    # _ = model(image)

    onnx_path = abspath_stem + "_enc.onnx"
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
    model_onnx = onnx.load(onnx_path)
    LOGGER.info(f"{colorstr("ONNX:")} slimming with onnxslim {onnxslim.__version__}...")
    model_onnx = onnxslim.slim(model_onnx)
    onnx.checker.check_model(model_onnx)
    onnx.save(model_onnx, onnx_path)
    t1 = time.time()
    mb = file_size(onnx_path)
    assert mb > 0.0, "0.0 MB output model size"
    LOGGER.info(f"{colorstr("ONNX:")} export success ✅ {(t1 - t0):.1f}s, saved as '{onnx_path}' ({mb:.1f} MB)")

    mnn_path = abspath_stem + "_enc.mnn"
    onnx2mnn(onnx_path, mnn_path, half, int8, "biz", colorstr("MNN:"))
    t2 = time.time()
    mb = file_size(mnn_path)
    assert mb > 0.0, "0.0 MB output model size"
    LOGGER.info(f"{colorstr("MNN:")} export success ✅ {(t2 - t1):.1f}s, saved as '{mnn_path}' ({mb:.1f} MB)")

    return mnn_path


def export_decoder(model, abspath_stem, half: bool, int8: bool):
    LOGGER.info(f"\n{colorstr("ONNX:")} starting export with onnx {onnx.__version__} opset {OPSET}...")
    point_coords = torch.rand((1, 4, 2))  # [1, num_points, 2]
    point_labels = torch.randint(-1, 4, (1, 4))  # [1, num_points]
    image_embed = torch.randn((1, 256, 64, 64))
    high_res_feats_0 = torch.randn((1, 32, 256, 256))
    high_res_feats_1 = torch.randn((1, 64, 128, 128))
    # _ = model(point_coords, point_labels, image_embed, high_res_feats_0, high_res_feats_1)

    onnx_path = abspath_stem + "_dec.onnx"
    t0 = time.time()
    torch.onnx.export(
        model,
        (point_coords, point_labels, image_embed, high_res_feats_0, high_res_feats_1),
        onnx_path,
        opset_version=OPSET,
        external_data=False,
        input_names=["point_coords", "point_labels", "image_embed", "high_res_feats_0", "high_res_feats_1"],
        output_names=["mask"],
        dynamic_axes={"point_coords": {1: "num_points"}, "point_labels": {1: "num_points"}},
    )
    model_onnx = onnx.load(onnx_path)
    LOGGER.info(f"{colorstr("ONNX:")} slimming with onnxslim {onnxslim.__version__}...")
    model_onnx = onnxslim.slim(model_onnx)
    onnx.checker.check_model(model_onnx)
    onnx.save(model_onnx, onnx_path)
    t1 = time.time()
    mb = file_size(onnx_path)
    assert mb > 0.0, "0.0 MB output model size"
    LOGGER.info(f"{colorstr("ONNX:")} export success ✅ {(t1 - t0):.1f}s, saved as '{onnx_path}' ({mb:.1f} MB)")

    mnn_path = abspath_stem + "_dec.mnn"
    onnx2mnn(onnx_path, mnn_path, half, int8, "biz", colorstr("MNN:"))
    t2 = time.time()
    mb = file_size(mnn_path)
    assert mb > 0.0, "0.0 MB output model size"
    LOGGER.info(f"{colorstr("MNN:")} export success ✅ {(t2 - t1):.1f}s, saved as '{mnn_path}' ({mb:.1f} MB)")

    return mnn_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/yolo26x.pt", help="The model file for training")
    parser.add_argument("--type", type=str, default="YOLO", help="Model type. Options: YOLO, SAM")
    parser.add_argument("--precision", type=str, default="fp32", help="Export precision. Options: fp32, fp16, int8")
    args = parser.parse_args()

    half = True if args.precision == "fp16" else False
    int8 = True if args.precision == "int8" else False

    if args.type == "YOLO":
        model = YOLO(args.model)
        model.export(format="mnn", half=half, int8=int8, simplify=True)
    elif args.type == "SAM":
        t0 = time.time()
        model = build_sam2_b(args.model)
        abspath_stem = str(Path(args.model).with_suffix(""))
        enc_path = export_encoder(Encoder(model).eval(), abspath_stem, half, int8)
        dec_path = export_decoder(Decoder(model).eval(), abspath_stem, half, int8)
        t1 = time.time()
        print(f"\nExport complete ({(t1 - t0):.1f}s)\nResults saved to {Path(enc_path).absolute()}\n                 {Path(dec_path).absolute()}")
    else:
        raise ValueError(f"Unknown type: '{args.type}', only support 'YOLO' or 'SAM'")
