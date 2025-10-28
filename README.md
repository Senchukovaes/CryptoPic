# Лабораторная 1: шифрование изображений

## Инструкция по запуску
Для запуска шифрования/дешифрования изображения необходимо создать конфигурацию запуска.

В разделе **script** нужно указать модуль, обрабатывающий входные команды и запускающий шифровку/дешифровку - cryptopic.py. В разделе **script parametrs** нужно ввести саму команду (они указаны ниже). В разделе **Working directory** необходимо указать рабочую директорию, это, непосредственно, сама CryptoPic.

## Команды для потокового XOR
### Шифрование
```
--mode encrypt --in imgs/original/squirrel.png --out imgs/encrypted/squirrel_stream.png --algo stream --key "мой секрет"
```
```
--mode encrypt --in imgs/original/noise_texture.png --out imgs/encrypted/noise_texture_stream.png --algo stream --key "мой секрет"
```
```
--mode encrypt --in imgs/original/gradient.png --out imgs/encrypted/gradient_stream.png --algo stream --key "мой секрет"
```
```
--mode encrypt --in imgs/original/checkerboard.png --out imgs/encrypted/checkerboard_stream.png --algo stream --key "мой секрет"
```

### Дешифрование
```
--mode decrypt --in imgs/encrypted/squirrel_stream.png --out imgs/decrypted/squirrel_stream.png --algo stream --key "мой секрет"
```
```
--mode decrypt --in imgs/encrypted/noise_texture_stream.png --out imgs/decrypted/noise_texture_stream.png --algo stream --key "мой секрет"
```
```
--mode decrypt --in imgs/encrypted/gradient_stream.png --out imgs/decrypted/gradient_stream.png --algo stream --key "мой секрет"
```
```
--mode decrypt --in imgs/encrypted/checkerboard_stream.png --out imgs/decrypted/checkerboard_stream.png --algo stream --key "мой секрет"
```

## Команды для перестановочного метода
### Шифрование
```
--mode encrypt --in imgs/original/squirrel.png --out imgs/encrypted/squirrel_perm.png --algo perm --key "мой секрет"
```
```
--mode encrypt --in imgs/original/noise_texture.png --out imgs/encrypted/noise_texture_perm.png --algo perm --key "мой секрет"
```
```
--mode encrypt --in imgs/original/checkerboard.png --out imgs/encrypted/checkerboard_perm.png --algo perm --key "мой секрет"
```
```
--mode encrypt --in imgs/original/gradient.png --out imgs/encrypted/gradient_perm.png --algo perm --key "мой секрет"
```

### Дешифрование
```
--mode decrypt --in imgs/encrypted/squirrel_perm.png --out imgs/decrypted/squirrel_perm.png --algo perm --key "мой секрет"
```
```
--mode decrypt --in imgs/encrypted/noise_texture_perm.png --out imgs/decrypted/noise_texture_perm.png --algo perm --key "мой секрет"
```
```
--mode decrypt --in imgs/encrypted/gradient_perm.png --out imgs/decrypted/gradient_perm.png --algo perm --key "мой секрет"
```
```
--mode decrypt --in imgs/encrypted/checkerboard_perm.png --out imgs/decrypted/checkerboard_perm.png --algo perm --key "мой секрет"
```
