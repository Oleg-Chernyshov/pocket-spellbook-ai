# Pocket Spellbook AI

Сервис умного поиска заклинаний для [Pocket Spellbook](https://github.com/Oleg-Chernyshov/pocket-spellbook). Модель своя: TF-IDF (слова + символьные n-граммы) + TruncatedSVD (LSA). Без GPU и без внешних LLM.

## Запуск

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
python -m app.cli build --source file
uvicorn app.main:app --reload --port 8000
```

Через Docker:

```bash
copy .env.example .env
docker compose up --build
```

При старте сервис загружает `models/model.joblib`. Если файла нет, модель собирается автоматически: сначала из API `pocket-spellbook-server`, при недоступности API — из снимка `data/spells_raw.json`.

Пересборка модели:

```bash
python -m app.cli build --source auto
```

`--source api` — только NestJS, `--source file` — только снимок.

## API

- `GET /health` — готовность модели. Пока её нет, отвечает `503`.
- `GET /search?q=&language=ru&page=1&limit=20&level=&school=&source=&characterClass=` — умный поиск.

Формат ответа совпадает с `GET /spells` бэкенда NestJS, плюс поле `score` (косинусная близость 0..1).

Пример:

```
GET http://localhost:8000/search?q=вылечить%20союзника%20на%20расстоянии&language=ru
```

## Соответствие ТЗ

1. **Загрузчик данных.** `app/loader.py` забирает заклинания из API NestJS (`GET /spells` по языкам и классам) или из JSON-снимка и приводит их к `SpellRecord`.
2. **Обработка.** `app/preprocess.py` снимает HTML, нормализует регистр и `ё`, чинит названия школ, отбрасывает битые строки, стеммит ru/en, собирает поисковый документ.
3. **Обучение.** `app/train.py` обучает TF-IDF + SVD и сохраняет модель в `models/model.joblib`.
4. **Интеграция.** Сервис ходит в API `pocket-spellbook-server` и отдаёт поиск фронтенду.
5. **Веб-интерфейс инференса.** Переключатель «Умный поиск» в основном поиске клиента.

Пункты 6 и 7 (управление данными и визуализация) не входят в этот сервис.

## Процесс разработки

Репозиторий также используется, чтобы опробовать инструменты GitHub для управления проектом: issues, доску GitHub Projects, ветки по ролям, pull request'ы с ревью и CI. Роли исполнителей (`ai-dev1`, `qa-lead` и другие) условные — проект разрабатывает один автор.

