import argparse
import os
import json
import hashlib

import numpy as np
from PIL import Image
from utils import derive_key_from_passphrase, generate_iv, save_meta, load_meta, derive_key_and_modified
from crypto_algos import encrypt_image_stream, decrypt_image_stream, block_permutation_encrypt, block_permutation_decrypt
from metrics import compute_basic_metrics, save_histograms, compute_all_sensitivity_metrics

RESULTS_DIR = "results"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["encrypt", "decrypt"], required=True)
    parser.add_argument("--in", dest="infile", required=True)
    parser.add_argument("--out", dest="outfile", required=True)
    parser.add_argument("--algo", choices=["stream", "perm"], required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--block", type=int, default=16)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.outfile) or ".", exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Преобразуем пароль в 32-байтный ключ с помозью SHA256
    key_bytes = derive_key_from_passphrase(args.key)

    if args.mode == "encrypt":
        img = Image.open(args.infile).convert("RGB")

        if args.algo == "stream":
            iv = generate_iv()
            enc = encrypt_image_stream(img, key_bytes, iv)
            enc.save(args.outfile)

            # Сохраняем meta данные
            save_meta(args.outfile, {"algo": args.algo, "iv": iv.hex()})
            print("Encrypted ->", args.outfile)

            # Считаем метрики
            metrics = compute_basic_metrics(args.infile, args.outfile)

            # Считаем метрики, зависящие от ключа
            sensitivity = compute_all_sensitivity_metrics(args.infile, args.key, algo='stream', iv=iv)

            # Save results
            base = os.path.splitext(os.path.basename(args.outfile))[0]
            metrics_path = os.path.join(RESULTS_DIR, base + ".metrics.json")
            sensitivity_path = os.path.join(RESULTS_DIR, base + ".sensitivity.json")
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            with open(sensitivity_path, "w", encoding="utf-8") as f:
                json.dump(sensitivity, f, indent=2, ensure_ascii=False)

            print(f"Метрики сохранены: {metrics_path}")
            print(f"Метрики чувствительности сохранены: {sensitivity_path}")

            # Histograms
            save_histograms(img, enc, os.path.join(RESULTS_DIR, "histograms"), base)

        elif args.algo == "perm":
            iv = generate_iv()
            enc, indices, grid = block_permutation_encrypt(img, key_bytes, iv, block_size=args.block)
            enc.save(args.outfile)

            # indices may be a list of ints -> json-serializable
            meta = {"algo": args.algo, "iv": iv.hex(), "indices": indices, "grid": list(grid), "block_size": args.block, "original_path": args.infile}
            save_meta(args.outfile, meta)
            print("Encrypted (perm) ->", args.outfile)

            metrics = compute_basic_metrics(args.infile, args.outfile)
            sensitivity = compute_all_sensitivity_metrics(args.infile, args.key, algo='perm', iv=iv)

            base = os.path.splitext(os.path.basename(args.outfile))[0]
            metrics_path = os.path.join(RESULTS_DIR, base + ".metrics.json")
            sensitivity_path = os.path.join(RESULTS_DIR, base + ".sensitivity.json")
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            with open(sensitivity_path, "w", encoding="utf-8") as f:
                json.dump(sensitivity, f, indent=2, ensure_ascii=False)

            print(f"Метрики сохранены: {metrics_path}")
            print(f"Метрики чувствительности сохранены: {sensitivity_path}")

            save_histograms(img, enc, os.path.join(RESULTS_DIR, "histograms"), base)

    elif args.mode == "decrypt":
        meta = load_meta(args.infile)
        img = Image.open(args.infile).convert("RGB")
        algo = meta.get("algo")

        if algo == "stream":
            iv = bytes.fromhex(meta["iv"])
            dec = decrypt_image_stream(img, key_bytes, iv)
            dec.save(args.outfile)
            print("Decrypted ->", args.outfile)

        elif algo == "perm":
            iv = bytes.fromhex(meta["iv"])
            indices = meta["indices"]
            grid = tuple(meta["grid"])
            dec = block_permutation_decrypt(img, key_bytes, iv, indices, grid, block_size=args.block)
            dec.save(args.outfile)
            print("Decrypted (perm) ->", args.outfile)

            # Pixel-hash check (SHA256 over pixel bytes)
            def get_image_pixel_hash(path):
                img_local = Image.open(path).convert("RGB")
                arr_local = np.array(img_local, dtype=np.uint8)
                return hashlib.sha256(arr_local.tobytes()).hexdigest()

            base_name = os.path.splitext(os.path.basename(args.infile))[0]
            # if base_name.endswith("_perm"):
            #     original_base = base_name[:-5]
            # else:
            #     original_base = base_name
            # original_path = os.path.join("imgs", "original", original_base + ".png")

            original_path = meta.get("original_path")

            original_hash = get_image_pixel_hash(original_path)
            print(original_path)
            decrypted_hash = get_image_pixel_hash(args.outfile)
            print(args.outfile)

            print("\nПроверка обратимости:")
            print(" - Хэш исходного файла :", original_hash)
            print(" - Хэш дешифрованного  :", decrypted_hash)

            is_rev = original_hash == decrypted_hash
            if is_rev:
                print("Обратимость подтверждена — файлы идентичны.")
            else:
                print("Внимание: файлы различаются (возможна ошибка в дешифровании).")

            # Save hashcheck
            os.makedirs(RESULTS_DIR, exist_ok=True)
            hashcheck_path = os.path.join(RESULTS_DIR, base_name + "_hashcheck.json")
            with open(hashcheck_path, "w", encoding="utf-8") as f:
                json.dump({
                    "original_file": original_path,
                    "decrypted_file": args.outfile,
                    "original_hash": original_hash,
                    "decrypted_hash": decrypted_hash,
                    "is_reversible": is_rev
                }, f, indent=2, ensure_ascii=False)
            print(f"Результаты проверки сохранены в {hashcheck_path}")


if __name__ == "__main__":
    main()
