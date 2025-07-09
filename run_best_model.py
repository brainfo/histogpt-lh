import argparse
import torch
import h5py

from lightning_trainer import LightningHistoGPT


def load_slide(h5_path):
    """Load features and coordinates from a slide H5 file."""
    with h5py.File(h5_path, "r") as f:
        feats = torch.tensor(f["features"][:], dtype=torch.float32)
        coords = None
        if "coordinates" in f:
            coords = torch.tensor(f["coordinates"][:], dtype=torch.float32)
    return feats, coords


def main():
    parser = argparse.ArgumentParser(description="Run HistoGPT-LH model on one slide")
    parser.add_argument("--checkpoint", required=True, help="Path to .ckpt file")
    parser.add_argument("--slide", required=True, help="Path to slide .h5 file")
    parser.add_argument("--output", default="predictions.txt", help="File to save outputs")
    args = parser.parse_args()

    # Load trained model
    model = LightningHistoGPT.load_from_checkpoint(args.checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Load slide data
    feats, coords = load_slide(args.slide)
    feats = feats.to(device)
    if coords is not None:
        coords = coords.to(device)

    # Predict diagnosis
    preds, short_texts = model.predict([feats], [coords], mode="binary")

    with open(args.output, "w") as f:
        f.write(f"Prediction: {short_texts[0]}\n")

    print("Prediction:", short_texts[0])


if __name__ == "__main__":
    main()
