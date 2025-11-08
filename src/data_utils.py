import re

import json

import torch

from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

from collections import Counter

from pathlib import Path





def preprocess_text(text: str) -> str: #Очищает текст от лишних символов:

    
    text = re.sub(r"[^а-яА-ЯёЁa-zA-Z\s]", "", text) #убирает пунктуацию, цифры, спецсимволы

    text = text.lower() #приводит к нижнему регистру

    text = re.sub(r"\s+", " ", text).strip() #удаляет лишние пробелы

    return text





class TextDataset(Dataset):

    """Простой датасет: хранит закодированные тексты"""



    def __init__(self, encoded_texts, max_len=50):

        self.data = encoded_texts

        self.max_len = max_len



    def __len__(self):

        return len(self.data)



    def __getitem__(self, idx):

        # Получаем один текст и дополняем нулями до max_len

        x = self.data[idx][:self.max_len]

        if len(x) < self.max_len:

            x = x + [0] * (self.max_len - len(x))

        return torch.tensor(x, dtype=torch.long)





def prepare_dataloaders(data_path="data/tweets.txt", batch_size=32, max_len=50):

    """

    Главная функция: готовит данные и возвращает

    train_loader, val_loader, test_loader, vocab, inv_vocab.

    """

    BASE_DIR = Path(__file__).resolve().parent.parent

    file_path = BASE_DIR / data_path



    if not file_path.exists():

        raise FileNotFoundError(f"Dataset file not found: {file_path}")



    print("Используется файл:", file_path)



    # 1) Загружаем тексты

    with open(file_path, "r", encoding="utf-8") as f:

        texts = [line.strip() for line in f if line.strip()]



    if len(texts) == 0:

        raise ValueError("Файл датасета пуст или не содержит непустых строк.")



    print(f"Загружено {len(texts)} текстов. Пример:")

    print(texts[0])



    # 2) Предобработка

    clean_texts = [preprocess_text(t) for t in texts]

    tokenized = [t.split() for t in clean_texts]



    # 3) Сохраняем промежуточные файлы

    output_dir = BASE_DIR / "data"

    output_dir.mkdir(exist_ok=True)



    with open(output_dir / "clean_texts.txt", "w", encoding="utf-8") as f:

        for line in clean_texts:

            f.write(line + "\n")



    with open(output_dir / "tokenized_texts.json", "w", encoding="utf-8") as f:

        json.dump(tokenized, f, ensure_ascii=False, indent=2)



    # 4) Создаём словарь (vocab) — локальная переменная внутри функции

    word_counts = Counter(word for text in tokenized for word in text)

    vocab = {word: i + 2 for i, (word, _) in enumerate(word_counts.items())}

    vocab["<PAD>"] = 0

    vocab["<UNK>"] = 1

    inv_vocab = {i: w for w, i in vocab.items()}



    with open(output_dir / "vocab.json", "w", encoding="utf-8") as f:

        json.dump(vocab, f, ensure_ascii=False, indent=2)



    # 5) Кодируем тексты

    def encode(tokens):

        return [vocab.get(word, vocab["<UNK>"]) for word in tokens]



    encoded_texts = [encode(t) for t in tokenized]



    with open(output_dir / "encoded_texts.json", "w", encoding="utf-8") as f:

        json.dump(encoded_texts, f)



    # 6) Делим данные

    train_texts, temp_texts = train_test_split(encoded_texts, test_size=0.2, random_state=42)

    val_texts, test_texts = train_test_split(temp_texts, test_size=0.5, random_state=42)



    # 7) Создаём Dataset’ы

    train_dataset = TextDataset(train_texts, max_len=max_len)

    val_dataset = TextDataset(val_texts, max_len=max_len)

    test_dataset = TextDataset(test_texts, max_len=max_len)



    # 8) DataLoader’ы

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



    print("\n✅ DataLoader’ы созданы корректно.")

    print(f"Train: {len(train_texts)} | Val: {len(val_texts)} | Test: {len(test_texts)}")



    # Возвращаем в фиксированном порядке

    return train_loader, val_loader, test_loader, vocab, inv_vocab





if __name__ == "__main__":

    # Быстрая проверка при запуске модуля напрямую

    train_loader, val_loader, test_loader, vocab, inv_vocab = prepare_dataloaders()

    for batch in train_loader:

        print(f"\nПример батча: форма {batch.shape}")

        print(batch)

        break