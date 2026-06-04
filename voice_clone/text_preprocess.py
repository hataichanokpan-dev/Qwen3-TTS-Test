"""
Thai text preprocessor for TTS.
Converts English words/abbreviations to Thai phonetic spelling.
Fixes pronunciation issues when using non-Thai-native TTS models.
Uses pythainlp for normalization and number conversion.
"""

import re
from pythainlp.util import normalize as pythainlp_normalize
from pythainlp.util import num_to_thaiword

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

# Punctuation fixes for better TTS pacing
# Add spaces around Thai punctuation for clearer segmentation
def fix_punctuation(text):
    # Ensure spaces after sentences for breathing pauses
    text = re.sub(r'\.\s*', ' ', text)
    # Replace multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def convert_numbers(text):
    """Convert numeric digits in text to Thai words using pythainlp.

    Handles integers, decimals, percentages, and comma-separated numbers.
    """
    def replace_number(match):
        num_str = match.group(0)
        try:
            # Handle percentages
            if '%' in num_str:
                num_str_clean = num_str.replace('%', '').replace(',', '')
                num = float(num_str_clean)
                if num == int(num):
                    return num_to_thaiword(int(num)) + "เปอร์เซ็นต์"
                # Decimal percentage
                int_part = int(num)
                dec_part = num_str_clean.split('.')[1]
                return num_to_thaiword(int_part) + "จุด" + "".join(
                    num_to_thaiword(int(d)) for d in dec_part
                ) + "เปอร์เซ็นต์"
            # Handle decimals: 3.14 → "สามจุดหนึ่งสี่"
            if '.' in num_str:
                parts = num_str.replace(',', '').split('.')
                int_part = int(parts[0])
                dec_digits = parts[1]
                return num_to_thaiword(int_part) + "จุด" + "".join(
                    num_to_thaiword(int(d)) for d in dec_digits
                )
            # Handle integers (with optional commas)
            num = int(num_str.replace(',', ''))
            return num_to_thaiword(num)
        except (ValueError, TypeError):
            return num_str

    # Match numbers: optional minus, digits (with optional commas), optional decimal, optional %
    result = re.sub(
        r'-?\d+(?:,\d{3})*(?:\.\d+)?%?',
        replace_number,
        text
    )
    return result


def preprocess_thai_text(text):
    """Preprocess Thai text for TTS synthesis.

    1. Normalize Thai characters (pythainlp)
    2. Convert numbers to Thai words (pythainlp)
    3. Replace English words with Thai phonetic spelling
    4. Fix abbreviations
    5. Normalize punctuation for pacing
    """
    # Step 1: pythainlp normalization (handles irregular chars, repeated markers)
    result = pythainlp_normalize(text)

    # Step 2: Convert numbers to Thai words
    result = convert_numbers(result)

    # Step 3: Apply abbreviation replacements
    for eng, thai in ABBREVIATIONS.items():
        pattern = r'\b' + re.escape(eng) + r'\b'
        result = re.sub(pattern, thai, result)

    # Step 4: Apply tech term replacements (longer phrases first)
    for eng, thai in sorted(TECH_TERMS.items(), key=lambda x: -len(x[0])):
        result = result.replace(eng, thai)

    # Step 5: Fix punctuation
    result = fix_punctuation(result)

    return result
