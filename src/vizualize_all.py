# # src/visualize_all.py
# """
# visualize_all.py
# Автоматически находит все файлы *.metrics.json в папке results (в корне проекта)
# и создаёт сводные таблицы (summary/*.png) для каждого найденного набора метрик+keytest.
# Этот скрипт устойчив к тому, из какой папки его запустили.
# """
#
# import os
# import json
# import sys
# from metrics import visualize_metrics_summary
#
# # вычисляем корень проекта как родитель папки src (где лежит этот файл)
# THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))
# RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
# SUMMARY_DIR = os.path.join(RESULTS_DIR, 'summary')
#
# def find_metric_files(results_dir):
#     """Возвращает список файлов, заканчивающихся на .metrics.json в results_dir (не рекурсивно)."""
#     if not os.path.exists(results_dir):
#         return []
#     files = [f for f in os.listdir(results_dir) if f.endswith('.metrics.json')]
#     return files
#
# def main():
#     print("Project root:", PROJECT_ROOT)
#     print("Looking for results in:", RESULTS_DIR)
#
#     files = find_metric_files(RESULTS_DIR)
#     if not files:
#         print("❌ Не найдено файлов .metrics.json в", RESULTS_DIR)
#         print("Проверь, что ты запускал(а) шифрование и что файлы находятся в папке results/")
#         print("Содержимое папки results:", os.listdir(RESULTS_DIR) if os.path.exists(RESULTS_DIR) else "results folder missing")
#         return
#
#     os.makedirs(SUMMARY_DIR, exist_ok=True)
#     processed = 0
#
#     # В цикле main():
#     for metrics_file in files:
#         base = metrics_file.replace('.metrics.json', '')
#         metrics_path = os.path.join(RESULTS_DIR, metrics_file)
#         sensitivity_path = os.path.join(RESULTS_DIR, base + '.sensitivity.json')  # Изменено с keytest
#         summary_path = os.path.join(SUMMARY_DIR, base + '_summary.png')
#
#         if not os.path.exists(sensitivity_path):
#             print(" ⚠️ sensitivity file not found:", sensitivity_path)
#             continue
#
#         try:
#             visualize_metrics_summary(metrics_path, sensitivity_path, summary_path)
#             print(" ✅ Summary created ->", summary_path)
#             processed += 1
#         except Exception as e:
#             print(" ❌ Ошибка при визуализации:", e)
#
#     # for metrics_file in files:
#     #     base = metrics_file.replace('.metrics.json', '')
#     #     metrics_path = os.path.join(RESULTS_DIR, metrics_file)
#     #     keytest_path = os.path.join(RESULTS_DIR, base + '.keytest.json')
#     #     summary_path = os.path.join(SUMMARY_DIR, base + '_summary.png')
#     #
#     #     print("\nProcessing:", metrics_file)
#     #     print(" - metrics_path:", metrics_path)
#     #     print(" - keytest_path:", keytest_path)
#     #
#     #     if not os.path.exists(metrics_path):
#     #         print(" ⚠️ metrics file disappeared:", metrics_path)
#     #         continue
#     #     if not os.path.exists(keytest_path):
#     #         print(" ⚠️ keytest file not found (skipping):", keytest_path)
#     #         continue
#     #
#     #     try:
#     #         visualize_metrics_summary(metrics_path, keytest_path, summary_path)
#     #         print(" ✅ Summary created ->", summary_path)
#     #         processed += 1
#     #     except Exception as e:
#     #         print(" ❌ Ошибка при визуализации:", e)
#
#     print(f"\nDone. {processed} summaries created in {SUMMARY_DIR}")
#
# if __name__ == "__main__":
#     main()

# src/visualize.py

import os
import json
import matplotlib.pyplot as plt
from matplotlib.table import Table
from metrics import visualize_metrics_summary

# вычисляем корень проекта как родитель папки src (где лежит этот файл)
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
SUMMARY_DIR = os.path.join(RESULTS_DIR, 'summary')

def find_metric_files(results_dir):
    if not os.path.exists(results_dir):
        return []
    files = [f for f in os.listdir(results_dir) if f.endswith('.metrics.json')]
    return files

def main():
    print("Project root:", PROJECT_ROOT)
    print("Looking for results in:", RESULTS_DIR)

    files = find_metric_files(RESULTS_DIR)
    if not files:
        print("Не найдено файлов .metrics.json в", RESULTS_DIR)
        print("Надо проверить, находятся ли файлы в папке results/")
        print("Содержимое папки results:", os.listdir(RESULTS_DIR) if os.path.exists(RESULTS_DIR) else "results folder missing")
        return

    os.makedirs(SUMMARY_DIR, exist_ok=True)
    processed = 0

    for metrics_file in files:
        # Используем базовое имя без расширения
        base_name = metrics_file.replace('.metrics.json', '')
        metrics_path = os.path.join(RESULTS_DIR, metrics_file)
        sensitivity_path = os.path.join(RESULTS_DIR, base_name + '.sensitivity.json')
        summary_path = os.path.join(SUMMARY_DIR, base_name + '_summary.png')

        print(f"\n Processing: {metrics_file}")
        print(f"   Metrics: {metrics_path}")
        print(f"   Sensitivity: {sensitivity_path}")

        if not os.path.exists(metrics_path):
            print(" Metrics file not found:", metrics_path)
            continue
        if not os.path.exists(sensitivity_path):
            print(" ️ Sensitivity file not found:", sensitivity_path)
            print("   Available files:", [f for f in os.listdir(RESULTS_DIR) if 'sensitivity' in f])
            continue

        try:
            visualize_metrics_summary(metrics_path, sensitivity_path, summary_path)
            print(f"  Summary created -> {summary_path}")
            processed += 1
        except Exception as e:
            print(f"  Error creating summary: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n Done. {processed} summaries created in {SUMMARY_DIR}")

if __name__ == "__main__":
    main()
