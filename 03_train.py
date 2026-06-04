import argparse

from ultralytics import YOLO

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="data/datasets/20250531/config.yaml", help="The path to the dataset configuration file")
    parser.add_argument("--model", type=str, default="models/yolo26x.pt", help="The model file for training")
    parser.add_argument("--model_type", type=str, default="yolo26x.yaml", help="The model type for training")
    parser.add_argument("--epoch", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size for training")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--device", type=str, default="0", help="Device to run training on")
    parser.add_argument("--multi_scale", type=float, default=0.25, help="Randomly vary imgsz each batch by +/-")
    args = parser.parse_args()

    # Load a pretrained model
    model = YOLO(args.model_type).load(args.model)

    # Train
    results = model.train(
        data=args.dataset,
        epochs=args.epoch,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        save_period=10,
        # amp=False,
        multi_scale=args.multi_scale,
    )
    print("=== Train complete ===\n")

    # Evaluate
    metrics = model.val(data=args.dataset)
    print("=== Eval complete ===\n")
