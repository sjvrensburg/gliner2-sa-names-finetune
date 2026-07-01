"""Synthetic value generators for non-name PII entity types.

Backs the co-occurring entity labels required to match privacy-ext's DEFAULT_LABELS
schema (name, street address, email, phone_num, id_num, url, username), so training
examples can combine `name` with these other entity types in one input -- teaching the
model not to over-trigger `name` on everything in an address/label block.

All values are fabricated (no real PII); phone/ID formats follow SA conventions.
"""

import random
import string

SA_AREA_CODES = ["011", "012", "021", "031", "041", "051", "071", "072", "073", "082", "083"]
STREET_TYPES = ["Street", "Road", "Avenue", "Drive", "Close", "Crescent"]
EMAIL_DOMAINS = ["example.co.za", "webmail.co.za", "mailbox.co.za", "example.com"]
URL_DOMAINS = ["example.co.za", "example.com", "shop.example.co.za"]


def fake_phone() -> str:
    area = random.choice(SA_AREA_CODES)
    rest = "".join(random.choices(string.digits, k=7))
    return f"{area} {rest[:3]} {rest[3:]}"


def _slugify(name: str) -> str:
    return "".join(c for c in name.lower().replace(" ", ".") if c.isalnum() or c == ".")


def fake_email(name: str) -> str:
    return f"{_slugify(name)}@{random.choice(EMAIL_DOMAINS)}"


def fake_username(name: str) -> str:
    slug = _slugify(name).replace(".", "")
    return f"{slug}{random.randint(1, 999)}"


def fake_street_address(place_pool: list[str]) -> str:
    number = random.randint(1, 500)
    street_type = random.choice(STREET_TYPES)
    place = random.choice(place_pool) if place_pool else "Central"
    return f"{number} {place} {street_type}"


def fake_id_num() -> str:
    # SA ID number shape: YYMMDD SSSS C A Z (13 digits) -- format-plausible, not a real checksum.
    yy = random.randint(0, 99)
    mm = random.randint(1, 12)
    dd = random.randint(1, 28)
    seq = "".join(random.choices(string.digits, k=4))
    tail = "".join(random.choices(string.digits, k=3))
    return f"{yy:02d}{mm:02d}{dd:02d}{seq}{tail}"


def fake_url() -> str:
    path = "".join(random.choices(string.ascii_lowercase, k=6))
    return f"https://{random.choice(URL_DOMAINS)}/{path}"
