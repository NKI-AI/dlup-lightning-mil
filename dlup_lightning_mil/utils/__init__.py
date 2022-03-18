# coding=utf-8
# Copyright (c) DLUP Contributors

from pathlib import Path
from typing import List, Tuple

def txt_of_paths_to_list(path: Path) -> List[Path]:
    """Reads a .txt file with a path per row and returns a list of paths"""
    content: List = []
    with open(path, "r") as f:
        while line := f.readline().rstrip():
            content.append(Path(line))
    return content


def txt_of_ints_to_list(path: Path) -> List[int]:
    """Reads a .txt file with a label per row and returns a list of labels"""
    content: List = []
    with open(path, "r") as f:
        while line := f.readline().rstrip():
            content.append(int(float(line)))
    return content


def txt_of_ids_to_list(path: Path) -> Tuple[List[str], List[str]]:
    """Reads a .txt file with a slide_id,case_id per row and returns two lists of slide ids and case ids"""
    content_slide: List = []
    content_patient: List = []
    with open(path, "r") as f:
        while line := f.readline().rstrip():
            patient, slide = line.split(',')
            content_slide.append(slide)
            content_patient.append(patient)
    return content_patient, content_slide
