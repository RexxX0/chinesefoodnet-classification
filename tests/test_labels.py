"""Keep shared metadata and both label readers aligned."""

import csv
from pathlib import Path

from infer import read_class_names as read_cli_labels
from inference import read_class_names as read_web_labels

LABELS = Path(__file__).resolve().parents[1] / "data/metadata/class_names.csv"


def test_shared_labels_are_contiguous_and_identical_for_both_consumers():
    with LABELS.open(encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.reader(stream) if row and row[0].isdigit()]
    assert [int(row[0]) for row in rows] == list(range(208))
    cli = read_cli_labels(str(LABELS))
    web = read_web_labels(LABELS)
    assert len(cli) == len(web) == 208
    for index, (zh, en) in enumerate(web):
        assert cli[index] == f"{index:03d} {zh}" + (f" | {en}" if en else "")


def test_bom_does_not_drop_first_class(tmp_path):
    labels = tmp_path / "labels.csv"
    labels.write_text("0,first,First\n1,second,Second\n", encoding="utf-8-sig")
    assert read_cli_labels(str(labels)) == ["000 first | First", "001 second | Second"]
    assert read_web_labels(labels, num_classes=2) == [("first", "First"), ("second", "Second")]
