"""The memory-transplant demo: predictions follow the bank, not the prompt.

Reads two different documents into two separate banks, then decodes the
expected continuation under each bank from the SAME blank input. Swapping
the memory swaps the prediction — the clearest way to see that the bank,
not the context window, is doing the work.

    python scripts/swap_banks.py <checkpoint.pt> <file_A> <file_B>
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # run from a bare clone
from fractale import BankSession


def main():
    ckpt, path_a, path_b = sys.argv[1], sys.argv[2], sys.argv[3]

    sess = BankSession.load(ckpt)
    banks = {}
    for name, path in (("A", path_a), ("B", path_b)):
        sess.reset().read(open(path, encoding="utf-8", errors="replace").read())
        banks[name] = sess.bank
        print(f"document {name}: {path} -> {sess.chunks_read} chunks in bank")

    for name in ("A", "B"):
        sess.swap_bank(banks[name])
        print(f"\n── continuation under bank {name} " + "─" * 30)
        print(sess.continuation(32))


if __name__ == "__main__":
    main()
