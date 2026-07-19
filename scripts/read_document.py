"""Feed a document to a Fractale model and see the memory at work.

Reads a file chunk by chunk (the bank accumulates), then asks the model to
predict how the text continues — from the bank ALONE (blank input) — and
shows the amnesic control next to it so you can see what the memory adds.

    python scripts/read_document.py <checkpoint.pt> <some_file> [n_tokens]
"""
import sys

from fractale import BankSession


def main():
    ckpt, path = sys.argv[1], sys.argv[2]
    n_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 32

    sess = BankSession.load(ckpt)
    text = open(path, encoding="utf-8", errors="replace").read()

    # hold back the tail so there is a real "next page" to expect
    cut = int(len(text) * 0.8)
    sess.read(text[:cut])
    print(f"read {sess.chunks_read} chunks -> bank stats: {sess.bank_stats()}\n")

    print("── with memory (bank only, blank input) " + "─" * 24)
    print(sess.continuation(n_tokens))
    print("\n── amnesic control (fresh bank) " + "─" * 32)
    print(sess.continuation(n_tokens, use_bank=False))
    print("\n── actual continuation " + "─" * 41)
    print(text[cut:cut + 200])


if __name__ == "__main__":
    main()
