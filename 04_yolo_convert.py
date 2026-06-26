import argparse

from ultralytics import YOLO

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/yolo26x_best.pt", help="The model file for training")
    parser.add_argument("--precision", type=str, default="fp32", help="Export precision. Options: fp32, fp16, int8")
    args = parser.parse_args()

    half = True if args.precision == "fp16" else False
    int8 = True if args.precision == "int8" else False
    model = YOLO(args.model)
    model.export(format="mnn", half=half, int8=int8, simplify=True)
