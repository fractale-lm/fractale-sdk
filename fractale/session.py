"""BankSession — the friendly way to drive a Fractale model.

A Fractale model is not used like a classic LLM: instead of one growing
prompt, you feed it text chunk by chunk and it maintains a *thought bank* —
8 self-written memory vectors — carried across forward passes. This wrapper
owns that loop: it chunks text, carries the bank, and exposes it as a state
you can save, restore, reset, swap or inspect.

    sess = BankSession.load("fractale-350m-base.pt")
    sess.read(open("long_file.py").read())     # any length; bank accumulates
    print(sess.continuation(32))               # what the model expects next,
                                               # predicted from the bank ALONE
    sess.save_bank("memory.pt")                # the whole memory, a few kB
"""
from __future__ import annotations

from typing import Optional

import torch
from transformers import AutoTokenizer

from ._core import ThoughtBankConfig, ThoughtBankLM

SPECIAL_TOKENS = ("<think>", "<blank>")
DEFAULT_TOKENIZER = "HuggingFaceTB/SmolLM2-135M"


class BankSession:
    def __init__(self, model: ThoughtBankLM, tokenizer, device: Optional[str] = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer
        self.think_id = tokenizer.convert_tokens_to_ids("<think>")
        self.blank_id = tokenizer.convert_tokens_to_ids("<blank>")
        self.chunk_len = self.model.cfg.max_seq_len - 128  # room for <think>
        self.bank: Optional[torch.Tensor] = None           # [1, n_slots, mem_dim]
        self.chunks_read = 0

    # ── constructors ─────────────────────────────────────────────────────────
    @classmethod
    def load(cls, checkpoint_path: str, tokenizer_name: str = DEFAULT_TOKENIZER,
             device: Optional[str] = None) -> "BankSession":
        """Load a Fractale checkpoint (.pt). The architecture config is stored
        inside the checkpoint, so the file is all you need."""
        tok = AutoTokenizer.from_pretrained(tokenizer_name)
        missing = [t for t in SPECIAL_TOKENS if t not in tok.get_vocab()]
        if missing:
            tok.add_special_tokens({"additional_special_tokens": missing})
        ck = torch.load(checkpoint_path, map_location="cpu")
        cfg = ThoughtBankConfig(**ck["cfg"])
        model = ThoughtBankLM(cfg)
        model.load_state_dict(ck["model"])
        return cls(model, tok, device=device)

    @classmethod
    def from_pretrained(cls, repo_id: str, filename: str = "model.pt",
                        device: Optional[str] = None) -> "BankSession":
        """Download a checkpoint from the Hugging Face Hub and load it."""
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id, filename)
        return cls.load(path, device=device)

    # ── reading: feed text, accumulate memory ────────────────────────────────
    @torch.no_grad()
    def read(self, text: str) -> "BankSession":
        """Read text of any length. It is split into chunks; after each chunk
        the model writes one gist vector into the bank (FIFO, 8 slots)."""
        ids = self.tokenizer(text, return_tensors="pt").input_ids[0]
        for start in range(0, len(ids), self.chunk_len):
            chunk = ids[start:start + self.chunk_len]
            self._read_chunk(chunk)
        return self

    @torch.no_grad()
    def _read_chunk(self, chunk_ids: torch.Tensor) -> None:
        think = torch.tensor([self.think_id], dtype=chunk_ids.dtype)
        x = torch.cat([chunk_ids, think]).unsqueeze(0).to(self.device)
        out = self.model(x, init_mem=self.bank, compute_logits=False)
        self.bank = out["mem_bank"].detach()
        self.chunks_read += 1

    # ── generation from the bank ─────────────────────────────────────────────
    @torch.no_grad()
    def continuation(self, n_tokens: int = 16, use_bank: bool = True,
                     temperature: float = 0.7, top_p: float = 0.95,
                     seed: Optional[int] = None) -> str:
        """Decode what the model expects the NEXT chunk to open with, from
        blank input — the bank is the only source of information.
        `use_bank=False` gives the amnesic control (fresh bank), so the
        difference between the two IS the memory.

        Sampling (nucleus, default temperature 0.7 / top_p 0.95) is the
        default: a base model decoded greedily falls into repetition loops.
        `temperature=0` gives deterministic greedy decoding."""
        bank = self.bank if use_bank else None
        gen = None
        if seed is not None:
            gen = torch.Generator(device="cpu").manual_seed(seed)
        x = torch.full((1, n_tokens), self.blank_id, dtype=torch.long, device=self.device)
        out_ids = []
        for i in range(n_tokens):
            o = self.model(x, init_mem=bank)
            logits = o["logits"].float()[0, i]
            if temperature <= 0:
                nxt = int(logits.argmax(-1))
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                sp, si = probs.sort(descending=True)
                keep = (sp.cumsum(-1) - sp) < top_p   # smallest set covering top_p
                sp = sp * keep
                pick = torch.multinomial(sp.cpu(), 1, generator=gen)
                nxt = int(si[int(pick)])
            out_ids.append(nxt)
            if i + 1 < n_tokens:
                x[0, i + 1] = nxt
        return self.tokenizer.decode(out_ids)

    # ── the bank as a first-class object ─────────────────────────────────────
    def reset(self) -> "BankSession":
        """Forget everything: next forward starts from the seed bank."""
        self.bank = None
        self.chunks_read = 0
        return self

    def save_bank(self, path: str) -> None:
        """Persist the session's memory (a few kB). Reload it any time —
        the 'conversation' survives the process."""
        torch.save({"bank": self.bank, "chunks_read": self.chunks_read}, path)

    def load_bank(self, path: str) -> "BankSession":
        state = torch.load(path, map_location=self.device)
        self.bank = state["bank"]
        self.chunks_read = state.get("chunks_read", 0)
        return self

    def swap_bank(self, other: torch.Tensor) -> "BankSession":
        """Install a bank written by another session — the model's predictions
        will follow the memory, not the prompt."""
        self.bank = other.to(self.device)
        return self

    def bank_stats(self) -> dict:
        """Slot norms and pairwise cosine similarity — a quick look at what
        the memory physically is right now."""
        if self.bank is None:
            return {"slots": 0, "note": "empty bank (will seed on next read)"}
        b = self.bank[0].float()
        sim = torch.nn.functional.normalize(b, dim=-1)
        return {
            "slots": b.size(0),
            "chunks_read": self.chunks_read,
            "slot_norms": [round(float(n), 2) for n in b.norm(dim=-1)],
            "pairwise_cos": (sim @ sim.T).round(decimals=2).tolist(),
        }
