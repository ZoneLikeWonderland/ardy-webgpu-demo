"""Shared CLI plumbing for non-default codec student architectures."""

from __future__ import annotations

import argparse

from .models import CodecStudentConfig


def add_codec_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--encoder-width", type=int, default=512)
    parser.add_argument("--encoder-blocks", type=int, default=4)
    parser.add_argument("--decoder-width", type=int, default=512)
    parser.add_argument("--decoder-blocks", type=int, default=8)
    parser.add_argument("--decoder-token-hidden", type=int, default=32)
    parser.add_argument("--codec-expansion", type=int, default=2)


def codec_config_from_args(args: argparse.Namespace) -> CodecStudentConfig:
    values = (
        args.encoder_width,
        args.encoder_blocks,
        args.decoder_width,
        args.decoder_blocks,
        args.decoder_token_hidden,
        args.codec_expansion,
    )
    if min(values) < 1:
        raise ValueError("codec widths, block counts, token hidden size and expansion must be positive")
    return CodecStudentConfig(
        encoder_width=args.encoder_width,
        encoder_blocks=args.encoder_blocks,
        decoder_width=args.decoder_width,
        decoder_blocks=args.decoder_blocks,
        decoder_token_hidden=args.decoder_token_hidden,
        expansion=args.codec_expansion,
    )
