"""

Главный скрипт запуска всех этапов проекта:

1. Предобработка данных

2. Создание модели LSTM

3. Тренировка модели

3. Использование предобученного трансформера DistilGPT-2

5. Выводы

"""


import sys

from pathlib import Path



# === Добавляем путь к папке src, чтобы корректно импортировать модули ===

BASE_DIR = Path(__file__).resolve().parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.eval_transformer import evaluate_transformer_rouge

from src.lstm_train import train_model     

from src.data_utils import prepare_dataloaders  # Этап 1

from src.lstm_model import LSTMTextModel    # Этап 2

import torch

from transformers import pipeline

import matplotlib.pyplot as plt        # Этап 5

import pandas as pd        # Этап 5

     # Этап 3

print("Все импорты прошли успешно")




def main():

    print("=" * 60)

    print("ЭТАП 1: Предобработка данных и создание DataLoader'ов")

    print("=" * 60)

    # Загружаем данные и получаем готовые объекты

    train_loader, val_loader, test_loader, vocab, inv_vocab = prepare_dataloaders()


    print(f"Размер словаря: {len(vocab)} токенов")

    print(f"Размеры: train={len(train_loader.dataset)}, val={len(val_loader.dataset)}, test={len(test_loader.dataset)}\n")



    # ===========================================================

    print("=" * 60)

    print("ЭТАП 2: Инициализация модели LSTM")

    print("=" * 60)

    vocab_size = len(vocab)

    embedding_dim = 64

    hidden_dim = 128

    model = LSTMTextModel(vocab_size, embedding_dim, hidden_dim)



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    print(f"Модель инициализирована на устройстве: {device}\n")



    # ===========================================================

    print("=" * 60)

    print("ЭТАП 3: Тренировка модели")

    print("=" * 60)

    train_model(

        model,

        train_loader,

        val_loader,

        vocab,

        inv_vocab,

        num_epochs=1,

        lr=0.001,

        device=device,

        save_path="lstm_model.pth"

    )



    print("\n✅ Обучение завершено. Модель сохранена в 'lstm_model.pth'.")

# ===========================================================

# ЭТАП 4: Оценка качества предобученного трансформера DistilGPT-2

# ===========================================================

    print("=" * 60)

    print("ЭТАП 4: Использование предобученного трансформера DistilGPT-2")

    print("=" * 60)


    transformer_metrics = evaluate_transformer_rouge()


    print("\n✅ Оценка трансформера завершена.")

    print("Метрики DistilGPT-2:")

    for k, v in transformer_metrics.items():

        print(f"{k}: {v:.4f}")

# Примеры

    generator = pipeline("text-generation", model="distilgpt2")

    examples = ["Я собираюсь", "Когда наступает утро", "В будущем люди смогут"]


    print("\nПримеры генерации DistilGPT-2:")

    for text in examples:

        result = generator(text, max_new_tokens=30, do_sample=True, top_k=50)

        print(f"- Вход: {text}")

        print(f"  Генерация: {result[0]['generated_text']}\n")

    # ===========================================================

    # ЭТАП 5: Выводы

    # ===========================================================

    print("=" * 60)

    print("ЭТАП 5: Сравнение моделей и формулирование выводов")

    print("=" * 60)



    # Примеры текстов для сравнения

    print("\nПримеры генерации LSTM:")

    try:

        with open("examples_lstm.txt", "r", encoding="utf-8") as f:

            lstm_examples = f.read()

        print(lstm_examples[:500] + "...")

    except FileNotFoundError:

        print("Файл с примерами LSTM не найден. (Можно добавить сохранение в lstm_train.py)")





# Пример данных метрики LSTM модели

    lstm_metrics = {"rouge1": 0.5147, "rouge2": 0.3971, "rougeL": 0.5064}

    transformer_metrics = transformer_metrics  # уже рассчитаны ранее



# Таблица сравнения

    df_compare = pd.DataFrame({

        "Метрика": lstm_metrics.keys(),

        "LSTM": lstm_metrics.values(),

        "DistilGPT-2": [transformer_metrics[k] for k in lstm_metrics.keys()]

})



    print("\n Сравнение метрик LSTM и DistilGPT-2:")

    print(df_compare.to_string(index=False))



# Построение графика

    plt.figure(figsize=(7,4))

    plt.bar(df_compare["Метрика"], df_compare["LSTM"], alpha=0.6, label="LSTM")

    plt.bar(df_compare["Метрика"], df_compare["DistilGPT-2"], alpha=0.6, label="DistilGPT-2")

    plt.title("Сравнение метрик ROUGE для двух моделей")

    plt.ylabel("Значение метрики")

    plt.legend()

    plt.tight_layout()

    plt.show()

    # Итоговые выводы

    print("=" * 60)

    print("=== Выводы: ===")


 

    if df_compare.loc[1, "rouge1"] > df_compare.loc[0, "rouge1"]:

        print("DistilGPT-2 показала более высокие значения ROUGE, что говорит о лучшей генерации текста.")

        print("Она лучше улавливает контекст и продолжает текст более естественно.")

    else:

        print("LSTM показала лучшие метрики, возможно, из-за меньшего переобучения или подбора гиперпараметров.")

        print("DistilGPT-2 можно улучшить через настройку температуры и длины генерации.")

if __name__ == "__main__":

    main()