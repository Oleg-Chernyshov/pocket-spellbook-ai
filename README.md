# Pocket Spellbook AI

Сервис умного поиска заклинаний для [Pocket Spellbook](https://github.com/Oleg-Chernyshov/pocket-spellbook). Без GPU и без внешних LLM.

Поиск гибридный:

- **Лексическая часть** — своя модель TF-IDF (слова + символьные n-граммы) + TruncatedSVD (LSA), обучается на заклинаниях. Отвечает за точные названия, опечатки и формы слов.
- **Смысловая часть** — эмбеддинги [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small) (int8 ONNX, ~118 МБ, CPU). Понимает запросы своими словами: «поговорить с волком» → «Общение с животными», «заставить предмет светиться» → «Свет».

Итоговый `score = (1 − SEMANTIC_WEIGHT) · TF-IDF + SEMANTIC_WEIGHT · смысловая близость`. Смысловая близость считается от медианы по запросу. В выдачу попадают заклинания со `score` не ниже `MIN_SCORE`.

На наборе из 40 запросов «своими словами» (`tests/test_quality.py`) нужное заклинание попадает в топ-5 в 72% случаев против 60% у одного TF-IDF; запросы по ключевым словам находятся так же, как раньше. Вес подобран на этом же наборе, так что оценка оптимистичная.

Модель эмбеддингов скачивается с Hugging Face при первой сборке в `models/embeddings` (в Docker — volume `ai_models`). Версия закреплена коммитом в `EMBEDDING_REVISION`, поэтому результат воспроизводим. Если скачать не удалось, сервис работает только на TF-IDF, а `GET /health` показывает `"semantic": false`.

## Запуск

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
copy .env.example .env
python -m app.cli build --source file
uvicorn app.main:app --reload --port 8000
```

Через Docker:

```bash
copy .env.example .env
docker compose up --build
```

При старте сервис загружает `models/model.joblib`. Модель собирается заново, если файла нет, если она обучена другой версией scikit-learn/numpy или с другими настройками эмбеддингов (всё это записано в `models/model.meta.json`). Первая сборка занимает около 30 секунд: скачивание модели эмбеддингов (~135 МБ) и кодирование ~1000 описаний. Источник данных — сначала API `pocket-spellbook-server`, при недоступности API — снимок `data/spells_raw.json`.

Пересборка модели:

```bash
python -m app.cli build --source auto
```

`--source api` — только NestJS, `--source file` — только снимок.

Пересборка без перезапуска сервиса (доступна, только если задан `ADMIN_TOKEN`):

```bash
curl -X POST "http://localhost:8000/admin/rebuild?source=auto" -H "X-Admin-Token: <ADMIN_TOKEN>"
```

## Настройки

- `SPELLBOOK_API_URL` — адрес `pocket-spellbook-server`.
- `MIN_SCORE` — минимальный итоговый `score`, чтобы заклинание попало в выдачу, по умолчанию `0.35`. Выставлен на глаз: мусорные запросы дают 0–12 результатов вместо сотен, верный ответ остаётся в выдаче для 49 из 54 тестовых запросов, по коротким запросам вроде «лечение» остаётся 8–20 заклинаний.
- `EMBEDDING_MODEL` — модель эмбеддингов семейства E5; пустое значение выключает смысловой поиск.
- `SEMANTIC_WEIGHT` — вес смысловой части в итоговом `score` (0 — только TF-IDF), по умолчанию `0.5`.
- `CORS_ORIGINS` — адреса фронтенда через запятую, по умолчанию `http://localhost:8080`.
- `ADMIN_TOKEN` — токен для `POST /admin/rebuild`; пустой — эндпоинт выключен.

## API

- `GET /health` — готовность модели, число заклинаний, включён ли смысловой поиск (`semantic`), дата сборки и хеш данных. Пока модели нет, отвечает `503`.
- `POST /admin/rebuild?source=auto|api|file` — пересобрать модель и подменить её на лету. Заголовок `X-Admin-Token`.
- `GET /search?q=&language=ru&page=1&limit=20&level=&school=&source=&characterClass=` — умный поиск.

Формат ответа совпадает с `GET /spells` бэкенда NestJS, плюс поле `score` (гибридная оценка, не больше 1).

Пример:

```
GET http://localhost:8000/search?q=вылечить%20союзника%20на%20расстоянии&language=ru
```

## Соответствие ТЗ

1. **Загрузчик данных.** `app/loader.py` забирает заклинания из API NestJS (`GET /spells` по языкам и классам) или из JSON-снимка и приводит их к `SpellRecord`.
2. **Обработка.** `app/preprocess.py` снимает HTML, нормализует регистр и `ё`, чинит названия школ, отбрасывает битые строки, стеммит ru/en, собирает поисковый документ.
3. **Обучение.** `app/train.py` обучает TF-IDF + SVD, считает эмбеддинги описаний (`app/embeddings.py`) и сохраняет модель в `models/model.joblib`.
4. **Интеграция.** Сервис ходит в API `pocket-spellbook-server` и отдаёт поиск фронтенду.
5. **Веб-интерфейс инференса.** Переключатель «Умный поиск» в основном поиске клиента.

Пункты 6 и 7 (управление данными и визуализация) не входят в этот сервис.

## Процесс разработки

Репозиторий также используется, чтобы опробовать инструменты GitHub для управления проектом: issues, доску GitHub Projects, ветки по ролям, pull request'ы с ревью и CI. Роли исполнителей (`ai-dev1`, `qa-lead` и другие) условные — проект разрабатывает один автор.

