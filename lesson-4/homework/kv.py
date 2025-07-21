import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

# 1. CSV fayldan URL larni o‘qish
df = pd.read_csv(r"D:\maab\ml\lesson-4\homework\olx_uylar_all_pages.csv", encoding="utf-8")
urls = df['url'].dropna().tolist()  # NaN bo‘lsa olib tashlaymiz

headers = {"User-Agent": "Mozilla/5.0"}

all_rows = []

# 2. Har bir URL bo‘yicha ma'lumot yig‘ish
for i, ad_url in enumerate(urls):
    try:
        print(f"{i+1}/{len(urls)}: Yuklanmoqda - {ad_url}")
        response = requests.get(ad_url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')

        chips = soup.find_all("p", class_="css-1los5bp")

        info = {}
        for chip in chips:
            text = chip.get_text(strip=True)
            if ':' in text:
                key, value = text.split(':', 1)
                info[key.strip()] = value.strip()
            else:
                info["seller_type"] = text.strip()

        desc_block = soup.find("div", {"data-cy": "ad_description"})
        description = desc_block.get_text(strip=True) if desc_block else None

        location = soup.find("div", class_="css-1cju8pu")
        city = location.get_text(strip=True) if location else None

        row = {
            "seller_type": info.get("seller_type"),
            "house_type": info.get("Turarjoy turi"),
            "rooms": info.get("Xonalar soni"),
            "area_m2": info.get("Umumiy maydon"),
            "floor": info.get("Qavati"),
            "floors_total": info.get("Uy qavatliligi"),
            "build_type": info.get("Qurilish turi"),
            "planning": info.get("Rejasi"),
            "bathroom": info.get("Sanuzel"),
            "furnished": info.get("Mebelli"),
            "includes": info.get("Kvartirada bor"),
            "nearby": info.get("Yaqinida joylashgan"),
            "commission": info.get("Vositachilik haqqi"),
            "description": description,
            "city": city,
            "url": ad_url
        }

        all_rows.append(row)
        time.sleep(1)  # Bloklanmaslik uchun kutish

    except Exception as e:
        print(f"⚠️ Xatolik {i+1}-e'londa: {e}")
        continue

# 3. Barcha natijalarni DataFramega aylantirish
df_detailed = pd.DataFrame(all_rows)

# 4. CSVga yozish
df_detailed.to_csv("olx_uy_detailed_dataset.csv", index=False, encoding="utf-8")
print("✅ Barcha ma’lumotlar muvaffaqiyatli CSV faylga yozildi.")
