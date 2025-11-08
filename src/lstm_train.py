import torch

import torch.nn as nn

import torch.optim as optim

from tqdm import tqdm

from rouge_score import rouge_scorer



# === Импорт вспомогательных функций и модели ===

from src.data_utils import prepare_dataloaders

from src.lstm_model import LSTMTextModel





# === Настройка устройства (GPU) ===

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





# === Функция вычисления ROUGE ===

def compute_rouge(pred_texts, ref_texts):


    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2"], use_stemmer=True)

    rouge1, rouge2 = [], []



    for pred, ref in zip(pred_texts, ref_texts):

        scores = scorer.score(ref, pred)

        rouge1.append(scores["rouge1"].fmeasure)

        rouge2.append(scores["rouge2"].fmeasure)



    return sum(rouge1)/len(rouge1), sum(rouge2)/len(rouge2)





# === Основная функция обучения модели ===

def train_model(model, train_loader, val_loader, vocab, inv_vocab,

                num_epochs=1, lr=0.001, device="cpu", save_path="lstm_model.pth"):

    """

    Обучает LSTM-модель на тренировочных данных, оценивает на валидации и сохраняет веса.

    """



    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)



    for epoch in range(num_epochs):

        model.train()

        total_loss = 0



        print(f"\n易 Эпоха {epoch+1}/{num_epochs}")



        # --- Цикл обучения ---

        for batch in tqdm(train_loader, desc="Обучение"):

            # batch — батч предложений (тензор [batch_size, seq_len])

            inputs = batch[:, :-1].to(device)   # всё, кроме последнего токена

            targets = batch[:, 1:].to(device)   # всё, кроме первого токена



            optimizer.zero_grad()

            outputs, _ = model(inputs)



            # Переформатируем для функции потерь: [batch*seq, vocab_size]

            loss = criterion(outputs.reshape(-1, len(vocab)), targets.reshape(-1))

            loss.backward()

            optimizer.step()

            total_loss += loss.item()



        avg_loss = total_loss / len(train_loader)

        print(f"Средняя функция потерь (train): {avg_loss:.4f}")



        # --- Валидация после каждой эпохи ---

        model.eval()

        val_loss = 0

        preds_text, refs_text = [], []



        with torch.no_grad():

            for batch in tqdm(val_loader, desc="Валидация"):

                inputs = batch[:, :-1].to(device)

                targets = batch[:, 1:].to(device)

                outputs, _ = model(inputs)



                loss = criterion(outputs.reshape(-1, len(vocab)), targets.reshape(-1))

                val_loss += loss.item()



                preds = torch.argmax(outputs, dim=2)

                preds_text += [" ".join([inv_vocab[t.item()] for t in seq]) for seq in preds]

                refs_text += [" ".join([inv_vocab[t.item()] for t in seq]) for seq in targets]



        rouge1, rouge2 = compute_rouge(preds_text, refs_text)

        print(f"Валидация: Loss={val_loss/len(val_loader):.4f} | ROUGE-1={rouge1:.4f} | ROUGE-2={rouge2:.4f}")



        # --- Пример автодополнения текста ---

        example_input = next(iter(val_loader))[0:1, :5].to(device)

        generated = model.generate(example_input, max_len=10)



        print("\nПример автодополнения:")

        print("Вход:", " ".join([inv_vocab[t.item()] for t in example_input[0]]))

        print("Сгенерировано:", " ".join([inv_vocab[t.item()] for t in generated[0]]))

        print("-" * 60)



    # === Сохранение модели ===

    torch.save(model.state_dict(), save_path)

    print(f"✅ Модель сохранена в {save_path}")





# === Точка входа ===

if __name__ == "__main__":

    # Загружаем данные и создаём модель

    train_loader, val_loader, test_loader, vocab, inv_vocab = prepare_dataloaders()



    model = LSTMTextModel(

        vocab_size=len(vocab),

        embed_dim=64,

        hidden_dim=128,

        num_layers=1

    ).to(device)



    # Запуск обучения

    train_model(

        model=model,

        train_loader=train_loader,

        val_loader=val_loader,

        vocab=vocab,

        inv_vocab=inv_vocab,

        num_epochs=1,

        lr=0.001,

        device=device,

        save_path="models/lstm_text_generator.pth"

    )



    print(" Обучение завершено успешно!")
    print("✅ Модель обучена и сохранена в models/lstm_text_generator.pth")

