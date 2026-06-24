"""Tests for the CLI argument parsing and demo-image command."""

import pytest

from visionforge.cli import build_parser, cmd_demo_image, main


def test_parser_version_exits():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--version"])


def test_parser_detect_defaults():
    parser = build_parser()
    args = parser.parse_args(["detect"])
    assert args.command == "detect"
    assert args.source == "demo"
    assert args.task == "detection"


def test_parser_detect_options():
    parser = build_parser()
    args = parser.parse_args(["detect", "--source", "0", "--task", "pose", "--backend", "yolo", "--max-frames", "5"])
    assert args.source == "0"
    assert args.task == "pose"
    assert args.backend == "yolo"
    assert args.max_frames == 5


def test_parser_serve():
    parser = build_parser()
    args = parser.parse_args(["serve", "--port", "9000"])
    assert args.command == "serve"
    assert args.port == 9000


def test_parser_rejects_bad_task():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["detect", "--task", "nonsense"])


def test_main_no_command_returns_1():
    assert main([]) == 1


def test_cmd_demo_image(tmp_path):
    out = tmp_path / "demo.jpg"
    import argparse

    args = argparse.Namespace(out=str(out), width=64, height=48)
    rc = cmd_demo_image(args)
    assert rc == 0
    assert out.exists()
    assert out.stat().st_size > 0
