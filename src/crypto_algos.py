from PIL import Image
import hashlib

# ---------- LCG PRNG ----------
class LCGPRNG:
    def __init__(self, seed: int):
        self.m = 2 ** 63
        self.a = 2806196910506780709
        self.c = 1013904223
        self.state = seed & 0xFFFFFFFFFFFFFFFF
        if self.state == 0:
            self.state = 0x12345678

    def next32(self) -> int:
        self.state = (self.a * self.state + self.c) % self.m
        # Циклический сдвиг битов
        self.state = ((self.state << 13) | (self.state >> 51)) & 0xFFFFFFFFFFFFFFFF
        return self.state & 0xFFFFFFFF

    def rand_byte(self) -> int:
        # Возвращаем случайный байт
        return self.next32() & 0xFF


# Вычисляем seed
def _seed_from_key_iv(key: bytes, iv: bytes) -> int:
    h = hashlib.sha256(key + iv).digest()
    return int.from_bytes(h[:8], "big")


# ---------- Stream XOR encryption ----------
def encrypt_image_stream(image: Image.Image, key: bytes, iv: bytes) -> Image.Image:
    if image.mode != "RGB":
        image = image.convert("RGB")

    pixels = list(image.getdata())  # список (R, G, B)
    # total_bytes = len(pixels) * 3

    seed = _seed_from_key_iv(key, iv)
    pr = LCGPRNG(seed)

    # Генерируем keystream и XOR'им побайтно
    cipher_bytes = bytearray()
    for (r, g, b) in pixels:
        cipher_bytes.append(r ^ pr.rand_byte())
        cipher_bytes.append(g ^ pr.rand_byte())
        cipher_bytes.append(b ^ pr.rand_byte())

    # Преобразуем обратно в изображение
    cipher_img = Image.new("RGB", image.size)
    cipher_img.frombytes(bytes(cipher_bytes))
    return cipher_img


def decrypt_image_stream(image: Image.Image, key: bytes, iv: bytes) -> Image.Image:
    return encrypt_image_stream(image, key, iv)


# ---------- Permutation + substitution ----------
def block_permutation_encrypt(image: Image.Image, key: bytes, iv: bytes, block_size: int = 16):
    if image.mode != "RGB":
        image = image.convert("RGB")

    w, h = image.size
    new_w = (w // block_size) * block_size
    new_h = (h // block_size) * block_size
    image = image.crop((0, 0, new_w, new_h))

    blocks_x = new_w // block_size # Сколько блоков по горизонтали
    blocks_y = new_h // block_size # Сколько блоков по вертикали
    total_blocks = blocks_x * blocks_y # Сколько всего блоков

    seed = _seed_from_key_iv(key, iv)
    pr = LCGPRNG(seed)

    # Создаём список индексов всех блоков
    indices = list(range(total_blocks))
    # Перемешиваем Fisher–Yates
    for i in range(total_blocks - 1, 0, -1):
        j = pr.next32() % (i + 1)
        indices[i], indices[j] = indices[j], indices[i]

    # Получаем пиксели как список байтов
    # data = list(image.getdata())

    # Собираем переставленные блоки
    # permuted_data = [None] * total_blocks * block_size * block_size
    permuted_img = Image.new("RGB", (new_w, new_h))
    src_pixels = image.load() # Пиксели исходного изображения
    dst_pixels = permuted_img.load() # Пиксели результирующего изображения

    for i in range(total_blocks):
        # Координаты левого верхнего угла исходного блока
        src_y = (i // blocks_x) * block_size
        src_x = (i % blocks_x) * block_size

        # Новая позиция блока
        dst_idx = indices[i]
        dst_y = (dst_idx // blocks_x) * block_size
        dst_x = (dst_idx % blocks_x) * block_size

        # Перенос пикселей блока
        for yy in range(block_size):
            for xx in range(block_size):
                dst_pixels[dst_x + xx, dst_y + yy] = src_pixels[src_x + xx, src_y + yy]

    # Второй шаг: XOR-диффузия
    # seed2 = _seed_from_key_iv(key, iv + b"diff")
    seed2 = _seed_from_key_iv(key, iv)
    pr2 = LCGPRNG(seed2)
    pixels = list(permuted_img.getdata())
    xor_data = bytearray()

    for (r, g, b) in pixels:
        xor_data.append(r ^ pr2.rand_byte())
        xor_data.append(g ^ pr2.rand_byte())
        xor_data.append(b ^ pr2.rand_byte())

    # Формируем Image из байтов
    encrypted_img = Image.new("RGB", permuted_img.size)
    # Возвращаем зашифрованное изображение
    encrypted_img.frombytes(bytes(xor_data))

    return encrypted_img, indices, (blocks_x, blocks_y)


def block_permutation_decrypt(image: Image.Image, key: bytes, iv: bytes, indices, grid, block_size: int = 16):
    if image.mode != "RGB":
        image = image.convert("RGB")

    blocks_x, blocks_y = grid
    w, h = image.size

    # Сначала снимаем XOR-mask
    # seed2 = _seed_from_key_iv(key, iv + b"diff")
    seed2 = _seed_from_key_iv(key, iv)
    pr2 = LCGPRNG(seed2)
    pixels = list(image.getdata())
    decoded_data = bytearray()
    for (r, g, b) in pixels:
        decoded_data.append(r ^ pr2.rand_byte())
        decoded_data.append(g ^ pr2.rand_byte())
        decoded_data.append(b ^ pr2.rand_byte())

    decoded_img = Image.new("RGB", image.size)
    decoded_img.frombytes(bytes(decoded_data))

    # Восстанавливаем порядок блоков
    total_blocks = blocks_x * blocks_y
    inverse = [0] * total_blocks
    for orig_pos, new_pos in enumerate(indices):
        inverse[new_pos] = orig_pos

    src_pixels = decoded_img.load()
    restored = Image.new("RGB", (w, h))
    dst_pixels = restored.load()

    for new_pos in range(total_blocks):
        src_y = (new_pos // blocks_x) * block_size
        src_x = (new_pos % blocks_x) * block_size

        orig_pos = inverse[new_pos]
        dst_y = (orig_pos // blocks_x) * block_size
        dst_x = (orig_pos % blocks_x) * block_size

        for yy in range(block_size):
            for xx in range(block_size):
                dst_pixels[dst_x + xx, dst_y + yy] = src_pixels[src_x + xx, src_y + yy]

    return restored
