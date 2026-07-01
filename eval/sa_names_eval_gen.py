import json

NAMES = {
    "isiZulu": ["Nomvula Mahlangu", "Thabo Nkosi", "Sipho Zungu", "Bongani Ndlovu",
                "Lindiwe Khumalo", "Nokuthula Mkhize", "Mandla Buthelezi", "Zanele Cele"],
    "isiXhosa": ["Sibusiso Ngcobo", "Ayanda Xulu", "Anele Mgcina", "Lithemba Sonjica",
                 "Nomalanga Gqamana", "Buhle Mtshali", "Zukisa Dyantyi", "Loyiso Nqevu"],
    "Sesotho_Setswana": ["Palesa Mokoena", "Katlego Molefe", "Tshepo Rakgoale",
                         "Lerato Sekgobela", "Kabelo Mothibi", "Refilwe Mmusi"],
    "Tshivenda_Xitsonga": ["Rendani Mulaudzi", "Vhonani Netshitenzhe", "Hlengiwe Chauke",
                           "Ndivhuwo Ramulondi", "Rirhandzu Baloyi"],
    "Afrikaans": ["Pieter van der Merwe", "Annemarie Botha", "Willem Kruger",
                  "Susanna Fourie", "Hendrik Nieuwoudt"],
    "Western_control": ["John Smith", "Maria Garcia", "David Johnson", "Emma Wilson",
                        "Wei Chen", "Fatima Al-Sayed"],
}

CONTEXTS = {
    "cue_explicit":    "Hi, my name is {name} and I'd like to schedule an appointment for next week.",
    "billing":         "Invoice #4471 billed to {name} - payment overdue.",
    "address_attn":    "Attn: {name}, Unit 4B, Soweto. Please call before delivery.",
    "form_field":      "Full Name: {name}",
    "email_sig":       "Kind regards,\n{name}\nSenior Consultant",
    "narrative_no_cue":"{name} arrived at the office around 9am and left the documents at reception.",
    "third_person":    "The parcel was collected by {name} on Tuesday afternoon.",
}

rows = []
i = 0
for group, names in NAMES.items():
    for name in names:
        for ctx_key, template in CONTEXTS.items():
            text = template.format(name=name)
            start = text.find(name)
            rows.append({
                "id": i, "group": group, "context": ctx_key,
                "name": name, "text": text,
                "expected_start": start, "expected_end": start + len(name),
            })
            i += 1

from pathlib import Path

with open(Path(__file__).parent / "sa_names_eval.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"wrote {len(rows)} cases")
