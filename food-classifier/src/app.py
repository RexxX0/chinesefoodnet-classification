"""Gradio demo app for ChineseFoodNet classifier."""

from __future__ import annotations

import argparse

import gradio as gr
import torch
from config import DEFAULT_CONFIG, load_config
from infer import build_model, predict_topk, preprocess_image, read_class_names


def build_predict_fn(config_path: str):
    cfg = load_config(config_path)
    model_cfg = cfg["model"]
    paths_cfg = cfg["paths"]
    inf_cfg = cfg["inference"]
    num_classes = int(cfg["project"]["num_classes"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(model_cfg["name"], num_classes, paths_cfg["checkpoint_path"], device)
    class_names = read_class_names(paths_cfg["class_names_file"])

    def _predict(image):
        if image is None:
            return {}
        tensor = preprocess_image(image, int(model_cfg["img_size"]))
        preds = predict_topk(model, tensor, class_names, int(inf_cfg["top_k"]), device)
        return {label: score for label, score in preds}

    return _predict


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the ChineseFoodNet Gradio demo.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    config_path = parser.parse_args().config

    interface = gr.Interface(
        fn=build_predict_fn(config_path),
        inputs=gr.Image(type="pil"),
        outputs=gr.Label(num_top_classes=5),
        title="ChineseFoodNet Classifier (208 classes)",
        description="Upload a food image to get top-k class predictions.",
    )
    interface.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
