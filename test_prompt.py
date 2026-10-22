"""
Скрипт тестирования промпт-генератора через локальный Ollama.
Запуск: python test_prompt.py
"""

import os
from datetime import datetime
from prompt_generator import PromptGenerator
from ollama_client import OllamaClient


REQUIREMENTS = """
Функциональность: Корзина интернет-магазина.

1. Пользователь может добавить товар в корзину со страницы каталога или карточки товара.
2. В корзине отображается список товаров, количество, цена за единицу и общая стоимость.
3. Пользователь может изменить количество товара в пределах от 1 до 10.
4. Пользователь может удалить товар из корзины.
5. При изменении количества или удалении товара общая стоимость пересчитывается.
6. Если товара нет на складе, при попытке добавления в корзину выводится сообщение об ошибке.
7. Максимальное количество уникальных товаров в корзине — 20.
8. Количество одного товара в корзине не может превышать 10.
"""


def main() -> None:
    print("=" * 60)
    print("Шаг 1. Генерация промпта из требований")
    print("=" * 60)

    generator = PromptGenerator(REQUIREMENTS)
    prompt = generator.build_prompt()
    print(f"Промпт сгенерирован. Длина: {len(prompt)} символов.\n")

    print("=" * 60)
    print("Шаг 2. Проверка доступности Ollama")
    print("=" * 60)

    client = OllamaClient(model="qwen2.5:7b")
    if not client.is_available():
        print("❌ Ollama недоступен или модель qwen2.5:7b не загружена.")
        print("Запусти: docker exec -it ollama ollama pull qwen2.5:7b")
        return
    print("✅ Ollama доступен, модель загружена.\n")

    print("=" * 60)
    print("Шаг 3. Запрос к LLM (может занять 1-5 минут на CPU)")
    print("=" * 60)

    answer = client.generate(prompt)
    if answer is None:
        print("❌ Не удалось получить ответ от LLM.")
        return

    print(f"✅ Ответ получен. Длина: {len(answer)} символов.\n")

    # Сохраняем результаты
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    prompt_path = os.path.join("results", f"prompt_{timestamp}.md")
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(prompt)

    answer_path = os.path.join("results", f"llm_output_{timestamp}.md")
    with open(answer_path, "w", encoding="utf-8") as f:
        f.write(answer)

    print("=" * 60)
    print("Результаты сохранены:")
    print(f"  Промпт:  {prompt_path}")
    print(f"  Ответ LLM: {answer_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()