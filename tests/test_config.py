"""Regression checks for portable configuration paths."""

from pathlib import Path

from config import DEFAULT_CONFIG, load_config


def test_default_paths_do_not_depend_on_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = load_config(DEFAULT_CONFIG)
    root = DEFAULT_CONFIG.parents[2]
    assert Path(config["paths"]["class_names_file"]) == root / "data/metadata/class_names.csv"
    assert Path(config["paths"]["checkpoint_path"]) == root / "outputs/convnextv2_base/best.pth"
    assert config["data"]["class_limit"] is None
    assert config["cross_validation"]["enabled"] is False


def test_relative_absolute_and_empty_paths(tmp_path):
    config_file = tmp_path / "custom.yaml"
    absolute = (tmp_path / "dataset").as_posix()
    config_file.write_text(
        f'paths:\n  train_dir: "{absolute}"\n  output_dir: runs\n'
        '  resume_checkpoint_path: null\nmodel:\n  pretrained_checkpoint: weights/model.pt\n',
        encoding="utf-8",
    )
    config = load_config(config_file)
    assert Path(config["paths"]["train_dir"]) == tmp_path / "dataset"
    assert Path(config["paths"]["output_dir"]) == tmp_path / "runs"
    assert config["paths"]["resume_checkpoint_path"] is None
    assert Path(config["model"]["pretrained_checkpoint"]) == tmp_path / "weights/model.pt"
