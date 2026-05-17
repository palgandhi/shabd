# models/rule_based.py
import re
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Comprehensive mapping tables for character-level Roman-to-Devanagari translation
CONSONANTS = {
    'ksh': 'क्ष', 'kshh': 'क्ष', 'gya': 'ज्ञ',
    'kh': 'ख', 'gh': 'घ', 'ch': 'च', 'chh': 'छ', 'jh': 'झ', 'th': 'थ',
    'dh': 'ध', 'ph': 'फ', 'bh': 'भ', 'sh': 'श', 'zh': 'झ', 'ng': 'ङ',
    'tr': 'त्र', 'gy': 'ज्ञ',
    'k': 'क', 'g': 'ग', 'j': 'ज', 't': 'त', 'd': 'द', 'n': 'न',
    'p': 'प', 'b': 'ब', 'm': 'म', 'y': 'य', 'r': 'र', 'l': 'ल',
    'v': 'व', 'w': 'व', 's': 'स', 'h': 'ह', 'z': 'ज़', 'f': 'फ़',
    'c': 'क', 'q': 'क', 'x': 'क्स'
}

VOWELS_STANDALONE = {
    'aa': 'आ', 'ee': 'ई', 'oo': 'ऊ', 'ai': 'ऐ', 'au': 'औ', 'ae': 'ए',
    'a': 'अ', 'i': 'इ', 'u': 'उ', 'e': 'ए', 'o': 'ओ'
}

VOWEL_MATRAS = {
    'aa': 'ा', 'ee': 'ी', 'oo': 'ू', 'ai': 'ै', 'au': 'ौ', 'ae': 'े',
    'a': '',  # inherent vowel (empty matra representation)
    'i': 'ि', 'u': 'ु', 'e': 'े', 'o': 'ो'
}

HALANT = '्'

def transliterate_word(word):
    """
    Translates a single Romanized Hindi word to Devanagari script using phonetic rules.
    """
    if not word:
        return ""
    
    word = word.lower()
    result = ""
    i = 0
    
    while i < len(word):
        # 1. Try longest consonant match (up to 4 letters down to 1)
        cons = None
        for L in (4, 3, 2, 1):
            if i + L <= len(word):
                substring = word[i:i+L]
                if substring in CONSONANTS:
                    cons = substring
                    break
        
        if cons:
            result += CONSONANTS[cons]
            i += len(cons)
            
            # Look ahead for a vowel immediately following this consonant to apply its matra
            vowel = None
            for L in (2, 1):
                if i + L <= len(word):
                    substring = word[i:i+L]
                    if substring in VOWEL_MATRAS:
                        vowel = substring
                        break
            
            if vowel is not None:
                # HEURISTIC: In Hinglish, a trailing 'a' is almost always pronounced as 'aa' (ा)
                # e.g., 'ka' -> का, 'kya' -> क्या, 'mera' -> मेरा, 'kaisa' -> कैसा
                if vowel == 'a' and i + len(vowel) == len(word):
                    result += VOWEL_MATRAS['aa']
                else:
                    result += VOWEL_MATRAS[vowel]
                i += len(vowel)
            elif i < len(word) and word[i] in 'bcdfghjklmnpqrstvwxyz':
                # Consonant cluster (another consonant follows immediately), apply halant
                result += HALANT
        else:
            # 2. Standalone vowel (either at start of word or after another vowel)
            vowel = None
            for L in (2, 1):
                if i + L <= len(word):
                    substring = word[i:i+L]
                    if substring in VOWELS_STANDALONE:
                        vowel = substring
                        break
            
            if vowel:
                result += VOWELS_STANDALONE[vowel]
                i += len(vowel)
            else:
                # Fallback for unrecognized character (digit, non-latin letter, etc.)
                result += word[i]
                i += 1
                
    return result

def transliterate_sentence(sentence):
    """
    Splits a sentence, transliterates Roman words individually, and preserves spaces/symbols.
    """
    if not sentence:
        return ""
        
    # Split text into tokens: alphabetic runs, digit runs, space runs, or punctuation blocks
    tokens = re.findall(r"[a-zA-Z]+|[^\s\w]+|\s+|\d+", sentence)
    out = []
    for tok in tokens:
        if re.fullmatch(r"[a-zA-Z]+", tok):
            out.append(transliterate_word(tok))
        elif tok == '.':
            out.append('।')
        else:
            out.append(tok)   # Pass through punctuation, numbers, spaces
    return "".join(out)

if __name__ == "__main__":
    # Test cases to evaluate baseline mapping
    test_words = [
        "namaste", "pustak", "naam", "aaj", "mausam", "kaisa", "mera", "ghar", "dost", "shukriya"
    ]
    print("Testing word-level rule-based transliteration:")
    for w in test_words:
        print(f"  {w} -> {transliterate_word(w)}")
        
    test_sentences = [
        "Ye pustak ka naam kya hai?",
        "Aaj mausam bohot accha hai.",
        "Bharat ki rajdhani Delhi hai!"
    ]
    print("\nTesting sentence-level rule-based transliteration:")
    for s in test_sentences:
        print(f"  Input:  {s}")
        print(f"  Output: {transliterate_sentence(s)}")
