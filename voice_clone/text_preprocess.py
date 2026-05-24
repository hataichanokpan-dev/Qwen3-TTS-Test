"""
Thai text preprocessor for TTS.
Converts English words/abbreviations to Thai phonetic spelling.
Fixes pronunciation issues when using non-Thai-native TTS models.
"""

import re

# Curated dictionary: English → Thai phonetic spelling
# Organized by category for maintainability

# Abbreviations & acronyms
ABBREVIATIONS = {
    "AI": "เอไอ",
    "SDET": "เอสดีอีที",
    "MCP": "เอ็มซีพี",
    "USB": "ยูเอสบี",
    "S&P": "เอสแอนด์พี",
}

# Tech terms & proper nouns
TECH_TERMS = {
    # Companies / Brands
    "Google": "กูเกิล",
    "LinkedIn": "ลิงค์ดอิน",
    "Gartner": "การ์ทเนอร์",
    "HashiCorp": "ฮาชิคอร์ป",
    "Anthropic": "แอนโทรปิก",
    "Air Canada": "แอร์แคนาดา",
    "Promptfoo": "พรอมต์ฟู",

    # Tech concepts
    "chatbot": "แชทบอท",
    "Chatbot": "แชทบอท",
    "agent": "เอเจนต์",
    "Agent": "เอเจนต์",
    "agents": "เอเจนต์",
    "model": "โมเดล",
    "Model": "โมเดล",
    "harness": "ฮาร์เนส",
    "Harness": "ฮาร์เนส",
    "guardrails": "การ์ดเรล",
    "Guardrails": "การ์ดเรล",
    "monitoring": "มอนิเตอร์ริ่ง",
    "monitor": "มอนิเตอร์",
    "evaluation": "อีวาลูเอชัน",
    "evaluation.": "อีวาลูเอชัน",
    "portfolio": "พอร์ตโฟลิโอ",
    "deterministic": "ดีเทอร์มินิสติก",
    "probabilistic": "โพรบาบิลิสติก",

    # Business terms
    "enterprise": "เอ็นเทอร์ไพรซ์",
    "production": "โปรดักชัน",
    "production.": "โปรดักชัน",
    "demo": "เดโม",
    "remote": "รีโมท",
    "multinational": "มัลติเนชันแนล",
    "startup": "สตาร์ทอัพ",

    # Job titles
    "Engineer": "เอนจิเนียร์",
    "developer": "ดีเวลอเปอร์",
    "Developer": "ดีเวลอเปอร์",
}

# Numbers and symbols
SYMBOLS = {
    "42%": "สี่สิบสองเปอร์เซ็นต์",
    "17%": "สิบเจ็ดเปอร์เซ็นต์",
    "70%": "เจ็ดสิบเปอร์เซ็นต์",
    "30%": "สามสิบเปอร์เซ็นต์",
    "40%": "สี่สิบเปอร์เซ็นต์",
    "56%": "ห้าสิบหกเปอร์เซ็นต์",
    "2025": "สองพันยี่สิบห้า",
    "2026": "สองพันยี่สิบหก",
}

# Punctuation fixes for better TTS pacing
# Add spaces around Thai punctuation for clearer segmentation
def fix_punctuation(text):
    # Ensure spaces after sentences for breathing pauses
    text = re.sub(r'\.\s*', ' ', text)
    # Replace multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def preprocess_thai_text(text):
    """Preprocess Thai text for TTS synthesis.

    1. Replace English words with Thai phonetic spelling
    2. Fix abbreviations
    3. Normalize punctuation for pacing
    """
    result = text

    # Apply symbol replacements first (longer matches first)
    for eng, thai in sorted(SYMBOLS.items(), key=lambda x: -len(x[0])):
        result = result.replace(eng, thai)

    # Apply abbreviation replacements
    for eng, thai in ABBREVIATIONS.items():
        # Use word boundary matching to avoid partial replacements
        pattern = r'\b' + re.escape(eng) + r'\b'
        result = re.sub(pattern, thai, result)

    # Apply tech term replacements (longer phrases first)
    for eng, thai in sorted(TECH_TERMS.items(), key=lambda x: -len(x[0])):
        result = result.replace(eng, thai)

    # Fix punctuation
    result = fix_punctuation(result)

    return result
