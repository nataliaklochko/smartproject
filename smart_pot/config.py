TOKEN = ""

ANSWERS = {
    "pos": "Да",
    "neg": "Нет",
    "unk": "Попробую ещё раз"
}

COMMANDS = {
    "start": """
        Привет!
        Каким должен быть твой идеальный светильник?
        Отправь фото и я помогу тебе с выбором в интернет магазине InHome360.ru
    """,
    "gen": "Отправь фото светильника и я помогу тебе с выбором в интернет магазине InHome360.ru",
    "search": "Есть кое-что для тебя...",
    "poll": "Нашёл ли ты что-нибудь подходящее?",
    "resp": "Есть ли похожие?",
    "pos": "Отлично!",
    "chosen": "Выбрано изображение №{}",
    "neg": "Очень жаль :( Ты можешь попробовать с другим фото, постараюсь помочь...",
    "unk": "Отправь другое фото, и я постараюсь помочь...",
    "num": "Cпасибо. Обращайся!:)",
    "next": "Выход"
}


POS = 0
NEG = 0
UNK = 0
nums = [str(i + 1) for i in range(10)]
