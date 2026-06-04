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


class ImageEncoder(torch.nn.Module):
    def __init__(self, model: SAM2Model):
        super().__init__()
        self.image_encoder = model.image_encoder

    def forward(self, input: torch.Tensor):
        backbone_out = self.image_encoder(input)
        image_embeddings = backbone_out["backbone_fpn"][2]
        return image_embeddings


class PointDecoder(torch.nn.Module):
    def __init__(self, model: SAM2Model):
        super().__init__()
        self.prompt_encoder = model.sam_prompt_encoder
        self.mask_decoder = model.sam_mask_decoder

    def forward(
        self,
        image_embeddings: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        imgsz: torch.Tensor,
    ):
        sparse_emb, dense_emb = self.prompt_encoder(points=(point_coords, point_labels), boxes=None, masks=None)

        masks, ious, _, _ = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
            repeat_image=False,
            high_res_features=None,
        )
        masks = F.interpolate(masks, (imgsz[0], imgsz[1]), mode="bilinear", align_corners=False)

        return masks, ious


class BoxDecoder(torch.nn.Module):
    def __init__(self, model: SAM2Model):
        super().__init__()
        self.prompt_encoder = model.sam_prompt_encoder
        self.mask_decoder = model.sam_mask_decoder

    def forward(
        self,
        image_embeddings: torch.Tensor,
        boxes: torch.Tensor,
        imgsz: torch.Tensor,
    ):
        sparse_emb, dense_emb = self.prompt_encoder(points=None, boxes=boxes, masks=None)

        masks, ious, _, _ = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
            repeat_image=True,
            high_res_features=None,
        )
        masks = F.interpolate(masks, (imgsz[0], imgsz[1]), mode="bilinear", align_corners=False)

        return masks, ious


def export_encoder(model, abspath_stem, half: bool, int8: bool):
    LOGGER.info(f"\n{colorstr("ONNX:")} starting export with onnx {onnx.__version__} opset {OPSET}...")
    onnx_path = abspath_stem + "_enc.onnx"
    mnn_path = abspath_stem + "_enc.mnn"
    t0 = time.time()
    torch.onnx.export(
        model,
        torch.randn(1, 3, 1024, 1024),
        onnx_path,
        opset_version=OPSET,
        input_names=["input"],
        output_names=["image_embeddings"],
        external_data=False,
    )
    model_onnx = onnx.load(onnx_path)
    LOGGER.info(f"{colorstr("ONNX:")} slimming with onnxslim {onnxslim.__version__}...")
    model_onnx = onnxslim.slim(model_onnx)
    onnx.save(model_onnx, onnx_path)
    t1 = time.time()
    mb = file_size(onnx_path)
    assert mb > 0.0, "0.0 MB output model size"
    LOGGER.info(f"{colorstr("ONNX:")} export success ✅ {(t1 - t0):.1f}s, saved as '{onnx_path}' ({mb:.1f} MB)")

    onnx2mnn(onnx_path, mnn_path, half, int8, "biz", colorstr("MNN:"))
    t2 = time.time()
    mb = file_size(mnn_path)
    assert mb > 0.0, "0.0 MB output model size"
    LOGGER.info(f"{colorstr("MNN:")} export success ✅ {(t2 - t1):.1f}s, saved as '{mnn_path}' ({mb:.1f} MB)")

    return mnn_path


def export_decoder(model, mode, abspath_stem, half: bool, int8: bool):
    LOGGER.info(f"\n{colorstr("ONNX:")} starting export with onnx {onnx.__version__} opset {OPSET}...")
    image_embeddings = torch.randn(1, 256, 64, 64)
    point_coords = torch.randint(0, 1024, (1, 3, 2))  # [1, num_points, 2]
    point_labels = torch.randint(0, 2, (1, 3))  # [1, num_points]
    boxes = torch.randint(0, 1024, (3, 2, 2))  # [num_boxes, 2, 2]
    imgsz = torch.tensor([1024, 1024], dtype=torch.int64)
    if mode == "point":
        onnx_path = abspath_stem + "_pdec.onnx"
        mnn_path = abspath_stem + "_pdec.mnn"
        inputs = (image_embeddings, point_coords, point_labels, imgsz)
        input_names = ["image_embeddings", "point_coords", "point_labels", "imgsz"]
        num_points = torch.export.Dim("num_points", min=1, max=8)
        dynamic_shapes = {
            "image_embeddings": None,
            "point_coords": {1: num_points},
            "point_labels": {1: num_points},
            "imgsz": None,
        }
    elif mode == "box":
        onnx_path = abspath_stem + "_bdec.onnx"
        mnn_path = abspath_stem + "_bdec.mnn"
        inputs = (image_embeddings, boxes, imgsz)
        input_names = ["image_embeddings", "boxes", "imgsz"]
        num_boxes = torch.export.Dim("num_boxes", min=1, max=64)
        dynamic_shapes = {
            "image_embeddings": None,
            "boxes": {0: num_boxes},
            "imgsz": None,
        }
    else:
        raise ValueError(f"Unknown mode: '{mode}', only support 'point' or 'box'")
    t0 = time.time()
    torch.onnx.export(
        model,
        inputs,
        onnx_path,
        opset_version=OPSET,
        input_names=input_names,
        output_names=["masks", "ious"],
        dynamic_shapes=dynamic_shapes,
        external_data=False,
    )
    model_onnx = onnx.load(onnx_path)
    LOGGER.info(f"{colorstr("ONNX:")} slimming with onnxslim {onnxslim.__version__}...")
    model_onnx = onnxslim.slim(model_onnx)
    onnx.save(model_onnx, onnx_path)
    t1 = time.time()
    mb = file_size(onnx_path)
    assert mb > 0.0, "0.0 MB output model size"
    LOGGER.info(f"{colorstr("ONNX:")} export success ✅ {(t1 - t0):.1f}s, saved as '{onnx_path}' ({mb:.1f} MB)")

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

        image_enc = ImageEncoder(model).eval()
        enc_path = export_encoder(image_enc, abspath_stem, half, int8)

        point_dec = PointDecoder(model).eval()
        pdec_path = export_decoder(point_dec, "point", abspath_stem, half, int8)

        box_dec = BoxDecoder(model).eval()
        bdec_path = export_decoder(box_dec, "box", abspath_stem, half, int8)
        t1 = time.time()

        print(
            f"\nExport complete ({(t1 - t0):.1f}s)"
            f"\nResults saved to {Path(enc_path).absolute()}"
            f"\n                 {Path(pdec_path).absolute()}"
            f"\n                 {Path(bdec_path).absolute()}"
        )
