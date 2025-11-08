import random

import torch

from tqdm import tqdm

from transformers import pipeline

import evaluate

from pathlib import Path

import re





def preprocess_text(text: str) -> str:

    """Очищает текст — тот же метод, что и в data_utils."""

    text = re.sub(r"[^а-яА-ЯёЁa-zA-Z\s]", "", text)

    text = text.lower()

    text = re.sub(r"\s+", " ", text).strip()

    return text





def evaluate_transformer_rouge(

    data_path="data/tweets.txt",

    num_samples=100,

    max_length_factor=1.3,

    top_k=50,

    temperature=0.8,

):
    



    BASE_DIR = Path(__file__).resolve().parent.parent

    file_path = BASE_DIR / data_path



    if not file_path.exists():

        raise FileNotFoundError(f"Файл не найден: {file_path}")



    # Загружаем тексты

    with open(file_path, "r", encoding="utf-8") as f:

        texts = [line.strip() for line in f if line.strip()]



    if len(texts) == 0:

        raise ValueError("Файл датасета пуст!")



    print(f" Загружено {len(texts)} текстов. Берем {num_samples} случайных примеров.")



    texts = random.sample(texts, min(num_samples, len(texts)))



    # Создаем пайплайн генерации

    device = 0 if torch.cuda.is_available() else -1

    generator = pipeline("text-generation", model="distilgpt2", device=device)



    rouge = evaluate.load("rouge")



    results = []



    print("\n Начинаем генерацию с DistilGPT-2...\n")



    for text in tqdm(texts):

        clean_text = preprocess_text(text)

        words = clean_text.split()

        if len(words) < 8:

            continue



        split_idx = int(len(words) * 0.75)

        prompt = " ".join(words[:split_idx])

        true_ending = " ".join(words[split_idx:])



        # Генерация продолжения

        generated_output = generator(

            prompt,

            max_length=int(len(words) * max_length_factor),

            do_sample=True,

            top_k=top_k,

            temperature=temperature,

            num_return_sequences=1,

        )[0]["generated_text"]



        # Извлекаем только добавленную часть

        generated_continuation = generated_output[len(prompt):].strip()



        # Добавляем в метрику

        rouge.add(prediction=generated_continuation, reference=true_ending)



        results.append({

            "prompt": prompt,

            "true_ending": true_ending,

            "generated": generated_continuation

        })



    # Вычисляем усреднённые результаты

    rouge_result = rouge.compute()



    print("\n Метрики ROUGE для DistilGPT-2:")

    for key, value in rouge_result.items():

        print(f"{key}: {value:.4f}")



    print("\nПримеры генерации:")

    for ex in results[:5]:

        print("──────────────────────────────")

        print("Начало:", ex["prompt"])

        print("Истинный конец:", ex["true_ending"])

        print("Сгенерировано:", ex["generated"])

        print()



    return rouge_result





if __name__ == "__main__":

    # Тестовый запуск

    metrics = evaluate_transformer_rouge()

    print("\n✅ Завершено. Средние метрики:")

    for k, v in metrics.items():

        print(f"{k}: {v:.4f}")