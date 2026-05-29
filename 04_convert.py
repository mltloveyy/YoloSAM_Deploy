import argparse

from ultralytics import SAM, YOLO

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument("--model", type=str, default="models/yolo26x.pt", help="The model file for training")
    parser.add_argument("--type", type=str, default="YOLO", help="Model type. Options: YOLO, SAM")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--precision", type=str, default="fp32", help="Export precision. Options: fp32, fp16, int8")
    args = parser.parse_args()

    if args.type == "YOLO":
        model = YOLO(args.model)
        if args.precision == "fp16":
            model.export(format="mnn", imgsz=args.imgsz, half=True)
        elif args.precision == "int8":
            model.export(format="mnn", imgsz=args.imgsz, int8=True)
        else:
            model.export(format="mnn", imgsz=args.imgsz)
    elif args.type == "SAM":
        model = SAM(args.model)
        model.export(format="mnn", imgsz=args.imgsz)
