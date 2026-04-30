# Copyright © 2026 Apple Inc.

import json
import math
import os
import re
from shutil import copyfile
from typing import Any, Dict, List, Optional, Pattern, Tuple

import numpy as np
from transformers import PretrainedConfig
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils import logging

try:
    from numba import njit
    from numba.core import types
    from numba.typed import Dict as NumbaDict

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

    def njit(fn=None, **kwargs):
        if fn is None:
            return lambda f: f
        return fn

    class _Types:
        int64 = None
        int32 = None

    class _NumbaDict:
        @staticmethod
        def empty(*args, **kwargs):
            return {}

    types = _Types()
    NumbaDict = _NumbaDict


VOCAB_FILES_NAMES = {"vocab_file": "tokenizer.jsonl"}
logger = logging.get_logger(__name__)

INVALID_SCORE = -20000000
UNKNOWN_SCORE = -10000000

TABLE_PIECE_LENGTH = 0
TABLE_TOKEN_ID = 1
TABLE_SCORE = 2
TABLE_PIECE_ID = 3

PATH_TOKEN_LENGTH = 0
PATH_TOKEN_ID = 1
PATH_NUM_TOKENS = 2

BOUNDARY_CHAR = "\uee00"
BOUNDARY_TOKEN_ID = 10000000


class Plamo3Config(PretrainedConfig):
    model_type = "plamo3"


class AhoCorasick:
    def __init__(self) -> None:
        self._tokens: List[str] = []
        self._bytes: np.ndarray
        self._to_suffix_id: Dict[Any, Any]
        self._table: np.ndarray
        self._sp_token_matcher: Optional[Pattern[str]] = None
        self._matcher: Optional[Pattern[str]] = None

    def build(
        self,
        vocab: List[Any],
        *,
        break_around_consecutive_spaces_threshold: Optional[int] = None,
        break_around_repeated_chars_threshold: Optional[int] = None,
    ) -> None:
        self._bytes = np.zeros(256, dtype=np.int32)
        if _HAS_NUMBA:
            self._to_suffix_id = NumbaDict.empty(
                key_type=types.int64, value_type=types.int32
            )
        else:
            self._to_suffix_id = {}

        suffix_to_score: Dict[str, float] = {}
        token_to_token_id: Dict[str, int] = {}
        self._tokens = []
        for token_id, row in list(enumerate(vocab)) + [
            (BOUNDARY_TOKEN_ID, [BOUNDARY_CHAR, 0, "CONTROL"])
        ]:
            assert isinstance(row[0], str), row
            assert isinstance(row[1], (int, float)), row

            token = str(row[0])
            self._tokens.append(token)
            token_to_token_id[token] = token_id

            if len(row) > 2 and row[2] == "BYTE":
                assert len(token) == 6 and token.startswith("<0x") and token.endswith(
                    ">"
                ), row[0]
                self._bytes[int(row[0][3:5], 16)] = token_id
                continue

            suffix_to_score[token] = float(row[1])
            for i in range(1, len(token)):
                suffix_to_score[token[i:]] = suffix_to_score.get(token[i:], math.nan)

        for i in range(256):
            assert self._bytes[i] != 0, "Byte token for <0x%02X> is not set." % i

        self._sp_token_matcher = re.compile(r"(<\|plamo:[^|\s]{,64}\|>)")

        patterns = []
        if break_around_repeated_chars_threshold is not None:
            patterns.append("(.)\\2{%d,}" % (break_around_repeated_chars_threshold - 1))
        if break_around_consecutive_spaces_threshold is not None:
            patterns.append(" {%d,}" % break_around_consecutive_spaces_threshold)
        self._matcher = re.compile("(%s)" % "|".join(patterns)) if patterns else None

        suffixes = list(suffix_to_score.keys())
        suffixes.append("")
        suffixes.sort(key=lambda x: x[::-1])

        suffix_to_id: Dict[str, int] = {}
        num_pieces = 0
        for suffix in suffixes:
            suffix_to_id[suffix] = num_pieces
            if suffix != "":
                self._to_suffix_id[
                    np.int64(ord(suffix[0]) << 32 | suffix_to_id[suffix[1:]])
                ] = np.int32(num_pieces)
            num_pieces += 1 + sum(
                suffix[:i] in suffix_to_score for i in range(1, len(suffix) + 1)
            )
        assert suffix_to_id[""] == 0, suffix_to_id[""]

        self._table = np.zeros((num_pieces, 4), dtype=np.int32)
        i = 0
        for suffix in suffixes:
            for piece_length in range(len(suffix), 0, -1):
                piece = suffix[:piece_length]
                score = suffix_to_score.get(piece, None)
                if score is None:
                    continue
                self._table[i, TABLE_PIECE_LENGTH] = piece_length
                self._table[i, TABLE_TOKEN_ID] = token_to_token_id.get(piece, -1)
                self._table[i, TABLE_SCORE] = (
                    round(score * 1e4) if math.isfinite(score) else INVALID_SCORE
                )
                self._table[i, TABLE_PIECE_ID] = suffix_to_id[piece]
                i += 1

            self._table[i, TABLE_PIECE_LENGTH] = 1
            self._table[i, TABLE_TOKEN_ID] = -1
            self._table[i, TABLE_SCORE] = UNKNOWN_SCORE
            i += 1
        assert i == num_pieces, (i, num_pieces)

    @staticmethod
    @njit
    def _encode(to_suffix_id, table, bytes_table, data):
        scores = np.full((len(data) + 1,), 2**60, dtype=np.int64)
        scores[-1] = 0

        path = np.zeros((len(data) + 1, 3), dtype=np.int32)
        suffix_id = np.int32(0)

        for i in range(len(data) - 1, -1, -1):
            c = data[i]

            for p in range(suffix_id, len(table)):
                suffix_id = to_suffix_id.get(
                    np.int64(c) << 32 | table[p, TABLE_PIECE_ID], np.int32(0)
                )
                if suffix_id > 0 or table[p, TABLE_SCORE] == UNKNOWN_SCORE:
                    break

            for p in range(suffix_id, len(table)):
                score = table[p, TABLE_SCORE]
                if score > INVALID_SCORE:
                    piece_length = table[p, TABLE_PIECE_LENGTH]
                    s = scores[i + piece_length] - score
                    if s < scores[i]:
                        scores[i] = s
                        path[i, PATH_TOKEN_LENGTH] = piece_length
                        path[i, PATH_TOKEN_ID] = table[p, TABLE_TOKEN_ID]
                        path[i, PATH_NUM_TOKENS] = (
                            path[i + piece_length, PATH_NUM_TOKENS] + 1
                        )
                        if score == UNKNOWN_SCORE:
                            path[i, PATH_NUM_TOKENS] += (
                                (c >= 0x80) + (c >= 0x800) + (c >= 0x10000)
                            )

                if score == UNKNOWN_SCORE:
                    break

        pos = 0
        token_ids = np.zeros(path[0, PATH_NUM_TOKENS], dtype=np.int32)
        token_pos = 0
        while pos < len(data):
            if path[pos, PATH_TOKEN_ID] >= 0:
                token_ids[token_pos] = path[pos, PATH_TOKEN_ID]
                if token_ids[token_pos] != BOUNDARY_TOKEN_ID:
                    token_pos += 1
            else:
                c = data[pos]
                s = 1 + (c >= 0x80) + (c >= 0x800) + (c >= 0x10000)
                for i in range(s):
                    b = c if s == 1 else (0xF00 >> s) & 0xFF if i == 0 else 0x80
                    token_ids[token_pos] = bytes_table[
                        b | ((c >> (s - i - 1) * 6) & 0x3F)
                    ]
                    token_pos += 1

            assert path[pos, PATH_TOKEN_LENGTH] > 0
            pos += path[pos, PATH_TOKEN_LENGTH]

        return token_ids[:token_pos]

    def encode(self, data: str) -> np.ndarray:
        if self._sp_token_matcher is not None:
            data = self._sp_token_matcher.sub(
                BOUNDARY_CHAR + "\\1" + BOUNDARY_CHAR, data
            )
        if self._matcher is not None:
            data = self._matcher.sub(BOUNDARY_CHAR + "\\1" + BOUNDARY_CHAR, data)
        return np.asarray(
            self._encode(
                self._to_suffix_id,
                self._table,
                self._bytes,
                np.frombuffer(data.encode("utf-32"), dtype=np.int32)[1:],
            )
        )

    def encode_as_tokens(self, data: str) -> List[str]:
        return [self._tokens[token_id] for token_id in self.encode(data)]


class Plamo3Tokenizer(PreTrainedTokenizer):
    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    _save_files = [
        "special_tokens_map.json",
        "tokenization_plamo.py",
        "tokenizer.jsonl",
        "tokenizer_config.json",
    ]

    def __init__(
        self,
        vocab_file: str,
        unk_token: str = "<|plamo:unk|>",
        bos_token: str = "<|plamo:bos|>",
        eos_token: str = "<|plamo:eos|>",
        pad_token: str = "<|plamo:pad|>",
        cls_token: Optional[str] = None,
        sep_token: Optional[str] = None,
        mask_token: Optional[str] = None,
        clean_up_tokenization_spaces: bool = False,
        break_around_consecutive_spaces_threshold: Optional[int] = None,
        break_around_repeated_chars_threshold: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        if "add_bos_token" not in kwargs:
            kwargs["add_bos_token"] = False
        if "add_eos_token" not in kwargs:
            kwargs["add_eos_token"] = False
        with open(vocab_file, encoding="utf-8") as f:
            self.data: List[Any] = [json.loads(line) for line in f]
        self.vocab: Dict[str, int] = {v[0]: i for i, v in enumerate(self.data)}
        self.aho_corasick = AhoCorasick()
        self.break_around_consecutive_spaces_threshold = (
            break_around_consecutive_spaces_threshold
        )
        self.break_around_repeated_chars_threshold = (
            break_around_repeated_chars_threshold
        )
        self.aho_corasick.build(
            self.data,
            break_around_consecutive_spaces_threshold=(
                self.break_around_consecutive_spaces_threshold
            ),
            break_around_repeated_chars_threshold=(
                self.break_around_repeated_chars_threshold
            ),
        )
        self.vocab_file = vocab_file
        self.add_bos_token = kwargs["add_bos_token"]
        self.add_eos_token = kwargs["add_eos_token"]

        super().__init__(
            vocab_file=vocab_file,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            cls_token=cls_token,
            sep_token=sep_token,
            mask_token=mask_token,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            break_around_consecutive_spaces_threshold=(
                break_around_consecutive_spaces_threshold
            ),
            break_around_repeated_chars_threshold=break_around_repeated_chars_threshold,
            **kwargs,
        )

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["aho_corasick"] = None
        return state

    def __setstate__(self, d: Dict[str, Any]) -> None:
        self.__dict__ = d
        self.aho_corasick = AhoCorasick()
        self.aho_corasick.build(
            self.data,
            break_around_consecutive_spaces_threshold=(
                self.break_around_consecutive_spaces_threshold
            ),
            break_around_repeated_chars_threshold=(
                self.break_around_repeated_chars_threshold
            ),
        )

    @property
    def vocab_size(self) -> int:
        return len(self.data)

    def token_to_score(self, token: str) -> Optional[float]:
        token_id = self.vocab.get(token, None)
        return None if token_id is None else self.data[token_id][1]

    def get_vocab(self) -> Dict[str, int]:
        vocab = self.vocab.copy()
        vocab.update(self.added_tokens_encoder)
        return vocab

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return b"".join(
            [
                bytes([int(t[3:5], 16)])
                if t.startswith("<0x")
                else t.encode("utf-8")
                for t in tokens
            ]
        ).decode("utf-8", errors="replace")

    def _tokenize(self, text: str, **kwargs: Any) -> List[str]:
        return self.aho_corasick.encode_as_tokens(text)

    def _convert_token_to_id(self, token: str) -> int:
        return self.vocab.get(token, 0)

    def _convert_id_to_token(self, index: int) -> str:
        return self.data[index][0]

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        output = bos_token_id + token_ids_0 + eos_token_id
        if token_ids_1 is not None:
            output = output + bos_token_id + token_ids_1 + eos_token_id
        return output

    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            logger.error("Vocabulary path (%s) should be a directory", save_directory)
            return ("",)
        out_vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "")
            + VOCAB_FILES_NAMES["vocab_file"],
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(
            out_vocab_file
        ) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "w") as f:
                for token in self.data:
                    print(json.dumps(token, ensure_ascii=False), file=f)

        return (out_vocab_file,)
