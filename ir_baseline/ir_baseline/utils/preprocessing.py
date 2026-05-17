"""
Text preprocessing utilities for IR baselines.
No pretrained models — only classical NLP tools.
"""

import re
import string
from typing import List

# ---------------------------------------------------------------------------
# Stop-words (standard English list, hard-coded — no NLTK download needed)
# ---------------------------------------------------------------------------
STOPWORDS = {
    "a","about","above","after","again","against","all","am","an","and","any",
    "are","aren't","as","at","be","because","been","before","being","below",
    "between","both","but","by","can't","cannot","could","couldn't","did",
    "didn't","do","does","doesn't","doing","don't","down","during","each",
    "few","for","from","further","get","got","had","hadn't","has","hasn't",
    "have","haven't","having","he","he'd","he'll","he's","her","here",
    "here's","hers","herself","him","himself","his","how","how's","i","i'd",
    "i'll","i'm","i've","if","in","into","is","isn't","it","it's","its",
    "itself","let's","me","more","most","mustn't","my","myself","no","nor",
    "not","of","off","on","once","only","or","other","ought","our","ours",
    "ourselves","out","over","own","same","shan't","she","she'd","she'll",
    "she's","should","shouldn't","so","some","such","than","that","that's",
    "the","their","theirs","them","themselves","then","there","there's",
    "these","they","they'd","they'll","they're","they've","this","those",
    "through","to","too","under","until","up","very","was","wasn't","we",
    "we'd","we'll","we're","we've","were","weren't","what","what's","when",
    "when's","where","where's","which","while","who","who's","whom","why",
    "why's","will","with","won't","would","wouldn't","you","you'd","you'll",
    "you're","you've","your","yours","yourself","yourselves",
}


def simple_tokenize(text: str) -> List[str]:
    """Lowercase, remove punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


def remove_stopwords(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def porter_stem(word: str) -> str:
    """
    Minimal Porter Stemmer (steps 1a, 1b, 1c, 2, 3, 4, 5).
    Pure Python — no external libraries.
    """
    if len(word) <= 2:
        return word

    def ends(w, s):
        return w.endswith(s)

    def measure(w):
        """Count VC sequences (Porter's 'm')."""
        stripped = re.sub(r"^[^aeiou]+", "", w)
        stripped = re.sub(r"[aeiou]+", "a", stripped)
        stripped = re.sub(r"[^a]", "b", stripped)
        return stripped.count("ab") + (1 if stripped.endswith("a") else 0)

    def has_vowel(w):
        return bool(re.search(r"[aeiou]", w))

    def ends_double_consonant(w):
        return (len(w) >= 2 and w[-1] == w[-2]
                and w[-1] not in "aeiou")

    def cvc(w):
        if len(w) < 3:
            return False
        return (w[-1] not in "aeiouwxy"
                and w[-2] in "aeiou"
                and w[-3] not in "aeiou")

    w = word

    # Step 1a
    if ends(w, "sses"):
        w = w[:-2]
    elif ends(w, "ies"):
        w = w[:-2]
    elif ends(w, "ss"):
        pass
    elif ends(w, "s"):
        w = w[:-1]

    # Step 1b
    if ends(w, "eed"):
        if measure(w[:-3]) > 0:
            w = w[:-1]
    elif ends(w, "ed"):
        if has_vowel(w[:-2]):
            w = w[:-2]
            if ends(w, "at") or ends(w, "bl") or ends(w, "iz"):
                w += "e"
            elif ends_double_consonant(w) and w[-1] not in "lsz":
                w = w[:-1]
            elif measure(w) == 1 and cvc(w):
                w += "e"
    elif ends(w, "ing"):
        if has_vowel(w[:-3]):
            w = w[:-3]
            if ends(w, "at") or ends(w, "bl") or ends(w, "iz"):
                w += "e"
            elif ends_double_consonant(w) and w[-1] not in "lsz":
                w = w[:-1]
            elif measure(w) == 1 and cvc(w):
                w += "e"

    # Step 1c
    if ends(w, "y") and has_vowel(w[:-1]):
        w = w[:-1] + "i"

    # Step 2
    step2_map = [
        ("ational", "ate"), ("tional", "tion"), ("enci", "ence"),
        ("anci", "ance"), ("izer", "ize"), ("abli", "able"),
        ("alli", "al"), ("entli", "ent"), ("eli", "e"),
        ("ousli", "ous"), ("ization", "ize"), ("ation", "ate"),
        ("ator", "ate"), ("alism", "al"), ("iveness", "ive"),
        ("fulness", "ful"), ("ousness", "ous"), ("aliti", "al"),
        ("iviti", "ive"), ("biliti", "ble"),
    ]
    for suf, rep in step2_map:
        if ends(w, suf) and measure(w[:-len(suf)]) > 0:
            w = w[:-len(suf)] + rep
            break

    # Step 3
    step3_map = [
        ("icate", "ic"), ("ative", ""), ("alize", "al"),
        ("iciti", "ic"), ("ical", "ic"), ("ful", ""), ("ness", ""),
    ]
    for suf, rep in step3_map:
        if ends(w, suf) and measure(w[:-len(suf)]) > 0:
            w = w[:-len(suf)] + rep
            break

    # Step 4
    step4_list = [
        "al","ance","ence","er","ic","able","ible","ant","ement",
        "ment","ent","ion","ou","ism","ate","iti","ous","ive","ize",
    ]
    for suf in step4_list:
        if ends(w, suf):
            stem = w[:-len(suf)]
            if measure(stem) > 1:
                if suf == "ion" and stem and stem[-1] in "st":
                    w = stem
                elif suf != "ion":
                    w = stem
                break

    # Step 5a
    if ends(w, "e"):
        a = w[:-1]
        if measure(a) > 1:
            w = a
        elif measure(a) == 1 and not cvc(a):
            w = a

    # Step 5b
    if ends_double_consonant(w) and ends(w, "l") and measure(w[:-1]) > 1:
        w = w[:-1]

    return w


def preprocess(text: str, stem: bool = True) -> List[str]:
    """Full pipeline: tokenize → remove stopwords → (optionally) stem."""
    tokens = simple_tokenize(text)
    tokens = remove_stopwords(tokens)
    if stem:
        tokens = [porter_stem(t) for t in tokens]
    return tokens
