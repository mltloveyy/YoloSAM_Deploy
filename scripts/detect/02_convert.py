import argparse
import time

from ultralytics import YOLO
from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.export.mnn import onnx2mnn
from ultralytics.utils.files import file_size

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/yolo26x_best.pt", help="The model file for training")
    parser.add_argument("--quantize", type=int, default=32, help="Export precision. e.g. 32(FP32), 16(FP16) or 8(INT8)")
    args = parser.parse_args()

    model = YOLO(args.model)
    if args.quantize == 8:
        onnx_path = model.export(format="onnx", simplify=True)
        mnn_path = onnx_path.replace("onnx", "mnn")
        t0 = time.time()
        onnx2mnn(onnx_path, mnn_path, args.quantize, "biz", colorstr("MNN:"))
        t1 = time.time()
        mb = file_size(mnn_path)
        assert mb > 0.0, "0.0 MB output model size"
        LOGGER.info(f"{colorstr("MNN:")} export success ✅ {(t1 - t0):.1f}s, saved as '{mnn_path}' ({mb:.1f} MB)")
    else:
        model.export(format="mnn", quantize=args.quantize, simplify=True)
