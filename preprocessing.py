# preprocessing.py
# Подготовка текстов и построение матрицы слов для NMF

import re
import string
from collections import Counter

import pymorphy3

from data import ARTICLES, TITLES


# Инициализация морфологического анализатора
_morph = pymorphy3.MorphAnalyzer()

# Русские стоп-слова (расширенный список)
STOP_WORDS = {
    "и", "в", "не", "на", "что", "с", "по", "для", "из", "как", "а", "то",
    "о", "он", "она", "они", "мы", "вы", "я", "к", "у", "за", "от", "до",
    "же", "бы", "ни", "ли", "будет", "это", "так", "только", "уже", "еще",
    "все", "его", "ее", "их", "но", "или", "если", "быть", "был", "была",
    "были", "со", "об", "под", "при", "над", "про", "чтобы", "после", "перед",
    "без", "нас", "вас", "ним", "ней", "них", "вам", "нам", "ему", "ей",
    "эту", "этот", "эта", "эти", "этого", "один", "два", "три", "четыре",
    "пять", "первый", "второй", "новый", "год", "день", "раз", "где", "кто",
    "когда", "почему", "который", "которая", "которые", "того", "такой",
}


def _normalize_word(word):
    """
    Приводит слово к нижнему регистру, удаляет пунктуацию,
    выполняет лемматизацию через pymorphy2.
    """
    word = word.lower().strip(string.punctuation + string.whitespace + "0123456789")
    if len(word) < 3:
        return None
    if word in STOP_WORDS:
        return None
    parsed = _morph.parse(word)
    if not parsed:
        return None
    lemma = parsed[0].normal_form
    if lemma in STOP_WORDS:
        return None
    return lemma


def separate_words(text):
    """
    Разбивает текст на слова, очищает и лемматизирует их.
    Возвращает список значимых слов.
    """
    # Удаляем HTML-теги (если вдруг есть) и лишние символы
    text = re.sub(r'<[^>]+>', ' ', text)
    # Оставляем только кириллицу, латиницу и апострофы внутри слов
    tokens = re.findall(r"[a-zA-Zа-яА-ЯёЁ]+(?:['’-][a-zA-Zа-яА-ЯёЁ]+)*", text)
    words = []
    for tok in tokens:
        norm = _normalize_word(tok)
        if norm:
            words.append(norm)
    return words


def build_word_matrix(min_freq=2, max_freq_ratio=0.6):
    """
    Строит матрицу документов-слов (word-document matrix) и список слов.

    Параметры:
      min_freq        — минимальное число документов, в которых должно
                        встречаться слово
      max_freq_ratio  — максимальная доля документов (0..1), иначе слово
                        считается слишком общим и удаляется

    Возвращает:
      matrix  — список списков (частоты слов по документам)
      wordvec — список отобранных слов (столбцы матрицы)
      titles  — список заголовков документов
    """
    allwords = Counter()
    articlewords = []

    for text in ARTICLES:
        words = separate_words(text)
        cnt = Counter(words)
        articlewords.append(cnt)
        for w in cnt:
            allwords[w] += 1

    n_docs = len(articlewords)
    # Отбираем слова по частоте (как в книге: >3 и <60%)
    if min_freq is None:
        min_freq = 2
    wordvec = [
        w for w, c in allwords.items()
        if c >= min_freq and c < n_docs * max_freq_ratio
    ]
    wordvec.sort()

    # Формируем матрицу
    matrix = []
    for cnt in articlewords:
        row = [(cnt.get(w, 0)) for w in wordvec]
        matrix.append(row)

    return matrix, wordvec, TITLES
