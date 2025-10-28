import os
import json
import hashlib
import time

# --- Key derivation ---
def derive_key_from_passphrase(passphrase: str) -> bytes:
    return hashlib.sha256(passphrase.encode("utf-8")).digest()

class SimpleIVGenerator:
    def __init__(self, seed: int = None):
        if seed is None:
            # seed от времени и pid — обеспечивает изменение при каждом запуске
            seed = (int(time.time_ns()) ^ os.getpid()) & 0xFFFFFFFF
        self.state = seed
        self.a = 1103515245
        self.c = 12345
        self.m = 2 ** 32

    def next_byte(self) -> int:
        self.state = (self.a * self.state + self.c) % self.m
        return self.state & 0xFF

    def generate(self, n: int) -> bytes:
        b = bytearray()
        for _ in range(n):
            b.append(self.next_byte())
        return bytes(b)


def generate_iv() -> bytes:
    gen = SimpleIVGenerator()
    return gen.generate(16)


# --- Meta helpers ---
def save_meta(outfile: str, meta: dict):
    with open(outfile + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def load_meta(infile: str) -> dict:
    with open(infile + ".meta.json", "r", encoding="utf-8") as f:
        return json.load(f)


# --- Изменить ровно 1 бит в ключе (полезно для sensitivity) ---
def flip_one_bit(key: bytes) -> bytes:
    if not key:
        return key
    kb = bytearray(key)
    kb[0] ^= 0x01
    return bytes(kb)


def derive_key_and_modified(passphrase: str):
    k = derive_key_from_passphrase(passphrase)
    k2 = flip_one_bit(k)
    # debug: посчитать биты отличия
    diff = sum(bin(a ^ b).count("1") for a, b in zip(k, k2))
    # print(f"[utils] key diff bits = {diff}")
    return k, k2
