import os
import argparse
from tokenizers import BertWordPieceTokenizer

parser = argparse.ArgumentParser()

parser.add_argument("--corpus_file", type=str)
parser.add_argument("--vocab_size", type=int, default=32000)
parser.add_argument("--limit_alphabet", type=int, default=6000)

args = parser.parse_args()

tokenizer = BertWordPieceTokenizer(
    vocab=None,
    clean_text=True,
    handle_chinese_chars=True,
    strip_accents=False, # Must be False if cased model
    lowercase=False,
    wordpieces_prefix="##"
)

tokenizer.train(
    files=[args.corpus_file],
    limit_alphabet=args.limit_alphabet,
    vocab_size=args.vocab_size
)

tokenizer.save(f"ch-{args.limit_alphabet}-wpm-{args.vocab_size}")
