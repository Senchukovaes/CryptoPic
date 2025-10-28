import numpy as np
from PIL import Image
import os
import json
import math
import statistics
import matplotlib.pyplot as plt
from matplotlib.table import Table
from typing import List, Tuple, Dict

# ---------- Utilities ----------

# precompute bitcounts for bytes 0..255 for avalanche
_BYTE_BITCOUNT = [bin(i).count("1") for i in range(256)]


def _image_to_channel_lists(img: Image.Image) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    px = img.load()
    R = [[0] * w for _ in range(h)]
    G = [[0] * w for _ in range(h)]
    B = [[0] * w for _ in range(h)]
    for y in range(h):
        for x in range(w):
            r, g, b = px[x, y]
            R[y][x] = int(r)
            G[y][x] = int(g)
            B[y][x] = int(b)
    return R, G, B


def _flatten_channel(channel_2d: List[List[int]]) -> List[int]:
    out = []
    for row in channel_2d:
        out.extend(row)
    return out


def _histogram_from_channel_flat(flat_channel: List[int]) -> List[int]:
    hist = [0] * 256
    for v in flat_channel:
        hist[v] += 1
    return hist


# ---------- Entropy ----------
def entropy_channel(channel: List[List[int]]) -> float:
    flat = _flatten_channel(channel)
    total = len(flat)
    if total == 0:
        return 0.0
    hist = _histogram_from_channel_flat(flat)
    ent = 0.0
    for count in hist:
        if count == 0:
            continue
        p = count / total
        ent -= p * math.log2(p)
    return float(ent)


# ---------- Correlations H/V/D ----------
def correlation_direction(channel: List[List[int]], direction: str) -> float:
    h = len(channel)
    if h == 0:
        return 0.0
    w = len(channel[0])

    xs = []
    ys = []

    if direction == 'H':
        if w < 2:
            return 0.0
        for y in range(h):
            for x in range(w - 1):
                xs.append(channel[y][x])
                ys.append(channel[y][x + 1])
    elif direction == 'V':
        if h < 2:
            return 0.0
        for y in range(h - 1):
            for x in range(w):
                xs.append(channel[y][x])
                ys.append(channel[y + 1][x])
    elif direction == 'D':
        if h < 2 or w < 2:
            return 0.0
        for y in range(h - 1):
            for x in range(w - 1):
                xs.append(channel[y][x])
                ys.append(channel[y + 1][x + 1])
    else:
        raise ValueError("direction must be 'H','V' or 'D'")

    if not xs or not ys:
        return 0.0


    try:
        mean_x = statistics.mean(xs)
        mean_y = statistics.mean(ys)
    except statistics.StatisticsError:
        return 0.0


    cov = 0.0
    var_x = 0.0
    var_y = 0.0
    n = len(xs)
    for i in range(n):
        dx = xs[i] - mean_x
        dy = ys[i] - mean_y
        cov += dx * dy
        var_x += dx * dx
        var_y += dy * dy

    if var_x == 0.0 or var_y == 0.0:
        return 0.0

    corr = cov / math.sqrt(var_x * var_y)

    return float(corr)


def compute_correlations_for_image(arr_img) -> Dict[str, Dict[str, float]]:
    if isinstance(arr_img, Image.Image):
        R, G, B = _image_to_channel_lists(arr_img)
    else:
        # если передали уже списки каналов (R,G,B)
        R, G, B = arr_img
    return {
        'R': {'H': correlation_direction(R, 'H'), 'V': correlation_direction(R, 'V'), 'D': correlation_direction(R, 'D')},
        'G': {'H': correlation_direction(G, 'H'), 'V': correlation_direction(G, 'V'), 'D': correlation_direction(G, 'D')},
        'B': {'H': correlation_direction(B, 'H'), 'V': correlation_direction(B, 'V'), 'D': correlation_direction(B, 'D')},
    }


# ---------- NPCR / UACI ----------
def npcr_uaci(arr1_img, arr2_img) -> Tuple[List[float], List[float]]:
    # Получаем каналы
    if isinstance(arr1_img, Image.Image):
        R1, G1, B1 = _image_to_channel_lists(arr1_img)
    else:
        R1, G1, B1 = arr1_img
    if isinstance(arr2_img, Image.Image):
        R2, G2, B2 = _image_to_channel_lists(arr2_img)
    else:
        R2, G2, B2 = arr2_img

    # ПРОСТОЕ ИСПРАВЛЕНИЕ: проверяем размеры и используем минимальный
    h1, h2 = len(R1), len(R2)
    w1, w2 = len(R1[0]) if h1 > 0 else 0, len(R2[0]) if h2 > 0 else 0

    h = min(h1, h2)
    w = min(w1, w2)

    def channel_npcr_uaci(c1, c2):
        total = h * w
        if total == 0:
            return 0.0, 0.0

        diff_count = 0
        sum_abs = 0
        for y in range(h):
            for x in range(w):
                v1 = c1[y][x]
                v2 = c2[y][x]
                if v1 != v2:
                    diff_count += 1
                sum_abs += abs(v1 - v2)
        npcr = (diff_count / total) * 100.0
        uaci = (sum_abs / (total * 255.0)) * 100.0
        return npcr, uaci

    npcr_r, uaci_r = channel_npcr_uaci(R1, R2)
    npcr_g, uaci_g = channel_npcr_uaci(G1, G2)
    npcr_b, uaci_b = channel_npcr_uaci(B1, B2)

    return [npcr_r, npcr_g, npcr_b], [uaci_r, uaci_g, uaci_b]



# def npcr_uaci(arr1_img, arr2_img) -> Tuple[List[float], List[float]]:
#     # Получаем каналы
#     if isinstance(arr1_img, Image.Image):
#         R1, G1, B1 = _image_to_channel_lists(arr1_img)
#     else:
#         R1, G1, B1 = arr1_img
#     if isinstance(arr2_img, Image.Image):
#         R2, G2, B2 = _image_to_channel_lists(arr2_img)
#     else:
#         R2, G2, B2 = arr2_img
#
#
#     def channel_npcr_uaci(c1, c2):
#         h = len(c1)
#         if h == 0:
#             return 0.0, 0.0
#         w = len(c1[0])
#         total = h * w
#         diff_count = 0
#         sum_abs = 0
#         for y in range(h):
#             row1 = c1[y]
#             row2 = c2[y]
#             for x in range(w):
#                 v1 = row1[x]
#                 v2 = row2[x]
#                 if v1 != v2:
#                     diff_count += 1
#                 sum_abs += abs(v1 - v2)
#         npcr = (diff_count / total) * 100.0
#         uaci = (sum_abs / (total * 255.0)) * 100.0
#         return npcr, uaci
#
#     npcr_r, uaci_r = channel_npcr_uaci(R1, R2)
#     npcr_g, uaci_g = channel_npcr_uaci(G1, G2)
#     npcr_b, uaci_b = channel_npcr_uaci(B1, B2)
#
#     return [npcr_r, npcr_g, npcr_b], [uaci_r, uaci_g, uaci_b]


# ---------- Avalanche (bit change) ----------
def avalanche_bitwise(img1: Image.Image, img2: Image.Image) -> float:
    if img1.mode != "RGB":
        img1 = img1.convert("RGB")
    if img2.mode != "RGB":
        img2 = img2.convert("RGB")

    w1, h1 = img1.size
    w2, h2 = img2.size
    if (w1, h1) != (w2, h2):
        raise ValueError("Images must be the same size for avalanche calculation")

    px1 = img1.load()
    px2 = img2.load()

    total_bits = w1 * h1 * 3 * 8
    if total_bits == 0:
        return 0.0

    changed_bits = 0
    for y in range(h1):
        for x in range(w1):
            r1, g1, b1 = px1[x, y]
            r2, g2, b2 = px2[x, y]
            changed_bits += _BYTE_BITCOUNT[(r1 ^ r2) & 0xFF]
            changed_bits += _BYTE_BITCOUNT[(g1 ^ g2) & 0xFF]
            changed_bits += _BYTE_BITCOUNT[(b1 ^ b2) & 0xFF]

    return (changed_bits / total_bits) * 100.0


# ---------- Basic metrics (orig vs encrypted) ----------
def compute_basic_metrics(original_path: str, encrypted_path: str) -> dict:
    orig_img = Image.open(original_path).convert('RGB')
    enc_img = Image.open(encrypted_path).convert('RGB')

    # Каналы (как 2D lists)
    R_o, G_o, B_o = _image_to_channel_lists(orig_img)
    R_e, G_e, B_e = _image_to_channel_lists(enc_img)

    ent_orig = [entropy_channel(R_o), entropy_channel(G_o), entropy_channel(B_o)]
    ent_enc = [entropy_channel(R_e), entropy_channel(G_e), entropy_channel(B_e)]

    corr_orig = {
        'R': {'H': correlation_direction(R_o, 'H'), 'V': correlation_direction(R_o, 'V'), 'D': correlation_direction(R_o, 'D')},
        'G': {'H': correlation_direction(G_o, 'H'), 'V': correlation_direction(G_o, 'V'), 'D': correlation_direction(G_o, 'D')},
        'B': {'H': correlation_direction(B_o, 'H'), 'V': correlation_direction(B_o, 'V'), 'D': correlation_direction(B_o, 'D')},
    }
    corr_enc = {
        'R': {'H': correlation_direction(R_e, 'H'), 'V': correlation_direction(R_e, 'V'), 'D': correlation_direction(R_e, 'D')},
        'G': {'H': correlation_direction(G_e, 'H'), 'V': correlation_direction(G_e, 'V'), 'D': correlation_direction(G_e, 'D')},
        'B': {'H': correlation_direction(B_e, 'H'), 'V': correlation_direction(B_e, 'V'), 'D': correlation_direction(B_e, 'D')},
    }

    npcr_orig_enc, uaci_orig_enc = npcr_uaci(orig_img, enc_img)

    return {
        "entropy_original": ent_orig,
        "entropy_processed": ent_enc,
        "correlation_original": corr_orig,
        "correlation_processed": corr_enc,
        "npcr_original_encrypted": npcr_orig_enc,
        "uaci_original_encrypted": uaci_orig_enc
    }


# ---------- Histograms ----------
def save_histograms(original_img: Image.Image, processed_img: Image.Image, outdir: str, basename: str):
    os.makedirs(outdir, exist_ok=True)
    orig = np.array(original_img.convert('RGB'))
    proc = np.array(processed_img.convert('RGB'))
    colors = ['red', 'green', 'blue']
    channel_names = ['R', 'G', 'B']

    # original
    plt.figure(figsize=(12, 4))
    for i, (c, name) in enumerate(zip(colors, channel_names)):
        plt.subplot(1, 3, i+1)
        plt.hist(orig[..., i].ravel(), bins=256, color=c, alpha=0.7)
        plt.title(f'Original {name}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'{basename}_hist_original.png'))
    plt.close()

    # encrypted
    plt.figure(figsize=(12, 4))
    for i, (c, name) in enumerate(zip(colors, channel_names)):
        plt.subplot(1, 3, i+1)
        plt.hist(proc[..., i].ravel(), bins=256, color=c, alpha=0.7)
        plt.title(f'Encrypted {name}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'{basename}_hist_encrypted.png'))
    plt.close()


# def save_histograms(original_img: Image.Image, processed_img: Image.Image, outdir: str, basename: str):
#     os.makedirs(outdir, exist_ok=True)
#
#     if original_img.mode != "RGB":
#         original_img = original_img.convert("RGB")
#     if processed_img.mode != "RGB":
#         processed_img = processed_img.convert("RGB")
#
#     R_o, G_o, B_o = _image_to_channel_lists(original_img)
#     R_e, G_e, B_e = _image_to_channel_lists(processed_img)
#
#     colors = ['red', 'green', 'blue']
#     channel_names = ['R', 'G', 'B']
#     orig_flat = [_flatten_channel(R_o), _flatten_channel(G_o), _flatten_channel(B_o)]
#     enc_flat = [_flatten_channel(R_e), _flatten_channel(G_e), _flatten_channel(B_e)]
#
#     # original
#     plt.figure(figsize=(12, 4))
#     for i, (c, name) in enumerate(zip(colors, channel_names)):
#         plt.subplot(1, 3, i + 1)
#         plt.hist(orig_flat[i], bins=range(257), alpha=0.7)
#         plt.title(f'Original {name}')
#         plt.xlabel('Value')
#         plt.ylabel('Frequency')
#     plt.tight_layout()
#     plt.savefig(os.path.join(outdir, f'{basename}_hist_original.png'))
#     plt.close()
#
#     # encrypted
#     plt.figure(figsize=(12, 4))
#     for i, (c, name) in enumerate(zip(colors, channel_names)):
#         plt.subplot(1, 3, i + 1)
#         plt.hist(enc_flat[i], bins=range(257), alpha=0.7)
#         plt.title(f'Encrypted {name}')
#         plt.xlabel('Value')
#         plt.ylabel('Frequency')
#     plt.tight_layout()
#     plt.savefig(os.path.join(outdir, f'{basename}_hist_encrypted.png'))
#     plt.close()


# ---------- Sensitivity metrics (key change) ----------
def compute_all_sensitivity_metrics(image_path: str, key_str: str, algo: str, iv: bytes = None) -> dict:
    from crypto_algos import encrypt_image_stream, block_permutation_encrypt
    from utils import derive_key_and_modified, generate_iv
    from PIL import Image

    img = Image.open(image_path).convert('RGB')

    key1, key2 = derive_key_and_modified(key_str)
    if iv is None:
        iv = generate_iv()

    if algo == 'stream':
        enc1 = encrypt_image_stream(img, key1, iv)
        enc2 = encrypt_image_stream(img, key2, iv)
    elif algo == 'perm':
        enc1, _, _ = block_permutation_encrypt(img, key1, iv)
        enc2, _, _ = block_permutation_encrypt(img, key2, iv)
    else:
        raise ValueError("Unknown algo for sensitivity")

    npcr_diff, uaci_diff = npcr_uaci(enc1, enc2)
    bit_change = avalanche_bitwise(enc1, enc2)

    return {
        "npcr_between_ciphers": npcr_diff,
        "uaci_between_ciphers": uaci_diff,
        "bit_change_percent": bit_change
    }


# ---------- Visualization summary (keeps previous behavior) ----------
def visualize_metrics_summary(metrics_file: str, sensitivity_file: str, outpath: str):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    base_name = os.path.splitext(os.path.basename(metrics_file))[0]
    hashcheck_path = os.path.join(os.path.dirname(metrics_file), base_name + "_hashcheck.json")
    reversibility = None
    if os.path.exists(hashcheck_path):
        with open(hashcheck_path, "r", encoding="utf-8") as f:
            h = json.load(f)
            reversibility = "Да" if h.get("is_reversible") else "Нет"

    with open(metrics_file, "r", encoding="utf-8") as f:
        m = json.load(f)
    with open(sensitivity_file, "r", encoding="utf-8") as f:
        s = json.load(f)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")
    ax.set_title("Crypto Metrics Summary", fontsize=14, fontweight="bold", pad=20)

    headers = ["Метрика", "R", "G", "B"]

    rows = [
        ["Энтропия (ориг.)"] + [f"{v:.3f}" for v in m["entropy_original"]],
        ["Энтропия (шифр)"] + [f"{v:.3f}" for v in m["entropy_processed"]],
        ["", "", "", ""]
    ]

    corr_orig = m["correlation_original"]
    corr_enc = m["correlation_processed"]
    rows.extend([
        ["Корр. H (ориг.)"] + [f"{corr_orig[ch]['H']:.3f}" for ch in ['R', 'G', 'B']],
        ["Корр. V (ориг.)"] + [f"{corr_orig[ch]['V']:.3f}" for ch in ['R', 'G', 'B']],
        ["Корр. D (ориг.)"] + [f"{corr_orig[ch]['D']:.3f}" for ch in ['R', 'G', 'B']],
        ["Корр. H (шифр)"] + [f"{corr_enc[ch]['H']:.3f}" for ch in ['R', 'G', 'B']],
        ["Корр. V (шифр)"] + [f"{corr_enc[ch]['V']:.3f}" for ch in ['R', 'G', 'B']],
        ["Корр. D (шифр)"] + [f"{corr_enc[ch]['D']:.3f}" for ch in ['R', 'G', 'B']],
        ["", "", "", ""]
    ])

    rows.extend([
        ["NPCR ориг./шифр"] + [f"{v:.2f}%" for v in m["npcr_original_encrypted"]],
        ["UACI ориг./шифр"] + [f"{v:.2f}%" for v in m["uaci_original_encrypted"]],
        ["NPCR ключи"] + [f"{v:.2f}%" for v in s["npcr_between_ciphers"]],
        ["UACI ключи"] + [f"{v:.2f}%" for v in s["uaci_between_ciphers"]],
        ["", "", "", ""]
    ])

    rows.extend([
        ["Avalanche (bit change %)"] + ["—", "—", f"{s.get('bit_change_percent', 0):.2f}%"],
        ["Обратимость"] + ["—", "—", reversibility or "—"]
    ])

    n_rows = len(rows)
    n_cols = len(headers)
    table = Table(ax, bbox=[0, 0, 1, 1])

    for i, h in enumerate(headers):
        table.add_cell(0, i, width=1.0 / n_cols, height=1.0 / (n_rows + 1),
                       text=h, loc="center", facecolor="#4CAF50", edgecolor="black")

    for r, row_data in enumerate(rows, start=1):
        for c, val in enumerate(row_data):
            facecolor = "#f8f9fa" if r % 2 == 0 else "#ffffff"
            table.add_cell(r, c, width=1.0 / n_cols, height=1.0 / (n_rows + 1),
                           text=val, loc="center", facecolor=facecolor, edgecolor="black")

    ax.add_table(table)
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
