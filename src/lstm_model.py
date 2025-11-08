import torch
import torch.nn as nn
import torch.nn.functional as F

# ===============================

#   Модель на основе LSTM

# ===============================


class LSTMTextModel(nn.Module):

    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_layers=1):

        """
        vocab_size — размер словаря (количество уникальных токенов)

        embed_dim — размер эмбеддинга

        hidden_dim — размер скрытого состояния LSTM

        num_layers — количество слоёв LSTM

        """

        super(LSTMTextModel, self).__init__()


        # Векторные представления токенов

        self.embedding = nn.Embedding(vocab_size, embed_dim)


        # Основная рекуррентная часть — LSTM

        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)


        # Линейный слой для предсказания следующего токена

        self.fc = nn.Linear(hidden_dim, vocab_size)


    def forward(self, input_seq, hidden=None):

        """

        input_seq: [batch_size, seq_len] — входная последовательность токенов

        hidden: скрытые состояния

        Возвращает:

            logits — предсказания по всем токенам (до softmax)

            hidden — новое скрытое состояние LSTM

        """

        embedded = self.embedding(input_seq)  # [batch, seq_len, embed_dim]

        lstm_out, hidden = self.lstm(embedded, hidden)  # [batch, seq_len, hidden_dim]

        logits = self.fc(lstm_out)  # [batch, seq_len, vocab_size]

        return logits, hidden


    @torch.no_grad()

    def generate(self, start_seq, next_tokens=10, temperature=1.0, device="cpu"):

        """

        Генерация последовательности токенов.

        start_seq: список токенов (начальная последовательность)

        next_tokens: сколько токенов нужно сгенерировать

        temperature: (чем больше, тем более случайные выборы)

        """

        self.eval()  # режим генерации (отключаем dropout и тд)

        generated = torch.tensor(start_seq, dtype=torch.long, device=device).unsqueeze(0)

        hidden = None


        for _ in range(next_tokens):

            logits, hidden = self.forward(generated, hidden)

            next_token_logits = logits[:, -1, :] / temperature  # берём последние логиты

            probs = F.softmax(next_token_logits, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)  # случайный выбор токена

            generated = torch.cat((generated, next_token), dim=1)



        return generated.squeeze(0).tolist()

