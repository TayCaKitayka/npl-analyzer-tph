!pip install transformers torch pandas numpy scikit-learn datasets

import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    TrainerCallback
)
from datasets import Dataset
import torch
from sklearn.preprocessing import LabelEncoder
import zipfile
import os
from google.colab import files
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("Проверка содержимого загруженного архива...")
# Распаковка архива с данными
!unzip -l /content/datafefu.zip
!unzip -q /content/datafefu.zip -d /content/data

print("\nПроверка содержимого файла train.txt...")
!head -n 5 /content/data/train.txt

# Инициализация токенизатора и модели
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
num_labels = 6  # количество эмоций

def debug_file_content(file_path):
    """
    Функция для отладки содержимого файла
    """
    logger.info(f"Анализ файла: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            logger.info("Первые 5 строк файла:")
            for i, line in enumerate(file):
                if i < 5:
                    logger.info(f"Строка {i+1}: {repr(line)}")
                else:
                    break
    except Exception as e:
        logger.error(f"Ошибка при чтении файла: {str(e)}")

def load_data(file_path):
    """
    Загружает данные из txt файла с форматом text;emotion
    с расширенной обработкой ошибок и отладкой
    """
    debug_file_content(file_path)

    texts = []
    emotions = []

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for i, line in enumerate(file, 1):
                line = line.strip()
                if not line:  # пропускаем пустые строки
                    continue

                # Пробуем разные разделители
                for separator in [';', '\t', ',']:
                    parts = line.split(separator)
                    if len(parts) >= 2:
                        text = parts[0]
                        emotion = parts[1]
                        texts.append(text)
                        emotions.append(emotion)
                        break
                else:  # если ни один разделитель не сработал
                    logger.warning(f"Предупреждение: строка {i} имеет неверный формат: {line}")

    except Exception as e:
        logger.error(f"Ошибка при чтении файла {file_path}: {str(e)}")
        raise

    if not texts or not emotions:
        raise ValueError(f"Не удалось загрузить данные из файла {file_path}. Файл пуст или имеет неверный формат.")

    logger.info(f"Успешно загружено {len(texts)} записей из {file_path}")
    return pd.DataFrame({'text': texts, 'emotion': emotions})

class CustomCallback(TrainerCallback):
    """Пользовательский callback для логирования процесса обучения"""
    def on_epoch_end(self, args, state, control, **kwargs):
        logger.info(f"Завершена эпоха {state.epoch}")
        if hasattr(state, 'best_metric'):
            logger.info(f"Лучшая метрика: {state.best_metric}")

# Функция для токенизации текста
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

# Подготовка данных для обучения
def prepare_datasets():
    logger.info("Загрузка датасетов...")
    # Загрузка всех трех датасетов из распакованного архива
    train_df = load_data('/content/data/train.txt')
    val_df = load_data('/content/data/val.txt')
    test_df = load_data('/content/data/test.txt')

    logger.info(f"Загружено записей: Train - {len(train_df)}, Val - {len(val_df)}, Test - {len(test_df)}")

    # Проверка уникальных эмоций
    logger.info("\nУникальные эмоции в датасете:")
    logger.info(train_df['emotion'].unique())

    # Кодирование меток
    le = LabelEncoder()
    # Обучаем LabelEncoder только на тренировочных данных
    train_df['label'] = le.fit_transform(train_df['emotion'])
    # Применяем тот же LabelEncoder к validation и test
    val_df['label'] = le.transform(val_df['emotion'])
    test_df['label'] = le.transform(test_df['emotion'])

    logger.info("Преобразование в датасеты...")
    # Создание датасетов
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)

    logger.info("Токенизация датасетов...")
    # Токенизация
    train_tokenized = train_dataset.map(tokenize_function, batched=True)
    val_tokenized = val_dataset.map(tokenize_function, batched=True)
    test_tokenized = test_dataset.map(tokenize_function, batched=True)

    return train_tokenized, val_tokenized, test_tokenized, le

def main():
    logger.info("Начало процесса обучения...")

    # Подготовка данных
    train_dataset, val_dataset, test_dataset, label_encoder = prepare_datasets()

    logger.info("Инициализация модели...")
    # Инициализация модели
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )

    # Настройка параметров обучения
    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_dir='./logs',
        logging_steps=100,
        report_to="none"  # Отключаем интеграцию с wandb
    )

    logger.info("Начало обучения...")
    # Инициализация тренера
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[CustomCallback]
    )

    # Обучение модели
    trainer.train()

    logger.info("Оценка на тестовом наборе...")
    # Оценка на тестовом наборе
    test_results = trainer.evaluate(test_dataset)
    logger.info(f"Test results: {test_results}")

    logger.info("Сохранение модели...")
    # Создание директории для сохранения модели
    os.makedirs("emotion_model", exist_ok=True)

    # Сохранение модели и токенизатора
    model.save_pretrained("./emotion_model")
    tokenizer.save_pretrained("./emotion_model")

    # Сохранение label encoder
    import pickle
    with open('./emotion_model/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)

    logger.info("Создание архива для скачивания...")
    # Создание архива с моделью для скачивания
    !zip -r emotion_model.zip emotion_model/

    logger.info("Скачивание архива...")
    # Скачивание архива с моделью
    files.download('emotion_model.zip')

    logger.info("Процесс обучения завершен!")

if __name__ == "__main__":
    main()