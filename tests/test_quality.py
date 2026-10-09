from app.search import SearchEngine

PARAPHRASES = {
    "ru": [
        ("подлечить друга издалека", "Исцеляющее слово"),
        ("быстро переместиться на несколько метров", "Туманный шаг"),
        ("дышать под водой", "Подводное дыхание"),
        ("не разбиться при падении с высоты", "Медленное падение"),
        ("поговорить с волком", "Общение с животными"),
        ("узнать свойства магического предмета", "Оценка"),
        ("вернуть к жизни только что погибшего", "Оживление"),
        ("усыпить врагов", "Сон"),
        ("сделать так чтобы враг считал меня другом", "Очаровать персону"),
        ("починить сломанную вещь", "Починка"),
        ("ударить молнией по линии", "Молния"),
        ("превратить врага в животное", "Перевоплощение"),
        ("отправить существо на другой план", "Изгнание"),
        ("понимать чужую речь", "Понимание языков"),
        ("ходить по стенам и потолку", "Паучья цепкость"),
        ("ускорить союзника", "Ускорение"),
        ("изменить свою внешность", "Маскировка"),
        ("открыть запертую дверь", "Стук"),
        ("снять болезнь или паралич", "Малое восстановление"),
        ("убить одним словом", "Слово Силы: смерть"),
        ("защититься от атаки реакцией", "Щит"),
        ("отменить чужое заклинание", "Контрзаклятье"),
        ("почувствовать магию рядом", "Обнаружение магии"),
        ("призвать зверька-помощника", "Призыв фамильяра"),
        ("заставить предмет светиться", "Свет"),
    ],
    "en": [
        ("heal a friend from afar", "Healing Word"),
        ("teleport a short distance", "Misty Step"),
        ("breathe underwater", "Water Breathing"),
        ("fall safely from a height", "Feather Fall"),
        ("talk to a wolf", "Speak with Animals"),
        ("learn what a magic item does", "Identify"),
        ("bring back someone who just died", "Revivify"),
        ("put enemies to sleep", "Sleep"),
        ("fix a broken object", "Mending"),
        ("turn an enemy into an animal", "Polymorph"),
        ("walk on walls and ceilings", "Spider Climb"),
        ("change my appearance", "Disguise Self"),
        ("open a locked door", "Knock"),
        ("cancel another caster's spell", "Counterspell"),
        ("get a small animal companion", "Find Familiar"),
    ],
}


def _hit_rate(engine: SearchEngine, k: int = 5) -> float:
    hits = total = 0
    for language, cases in PARAPHRASES.items():
        for query, expected in cases:
            names = [item.name for item in engine.search(query, language=language, limit=k).data]
            hits += expected in names
            total += 1
    return hits / total


def test_hybrid_search_understands_paraphrases_better(hybrid_engine: SearchEngine, lexical_engine: SearchEngine):
    lexical = _hit_rate(lexical_engine)
    hybrid = _hit_rate(hybrid_engine)
    assert hybrid >= 0.7, f"hybrid hit@5={hybrid:.2f}"
    assert hybrid > lexical, f"hybrid hit@5={hybrid:.2f}, lexical hit@5={lexical:.2f}"


def test_unrelated_query_returns_few_results(hybrid_engine: SearchEngine):
    assert hybrid_engine.search("купить хлеба в магазине", language="ru").pagination.total < 20
    assert hybrid_engine.search("stock market prices", language="en").pagination.total < 20


def test_short_query_still_returns_a_list(hybrid_engine: SearchEngine):
    assert hybrid_engine.search("лечение", language="ru").pagination.total >= 5


def test_embeddings_do_not_invent_results_for_nonsense(hybrid_engine: SearchEngine, lexical_engine: SearchEngine):
    assert lexical_engine.search("asdfgh", language="ru").pagination.total == 0
    assert hybrid_engine.search("asdfgh", language="ru").pagination.total == 0
