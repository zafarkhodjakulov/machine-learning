import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

headers = {
    "User-Agent": "Mozilla/5.0"
}

# Fayl nomi
csv_file = "olx_uylar_all_pages.csv"

# Faylni boshlanishida header bilan yoziladi
first_page = True

for page in range(1, 27):
    url = f"https://www.olx.uz/oz/nedvizhimost/kvartiry/prodazha/?page={page}"
    print("Yuklanayotgan sahifa:", url)

    main_response = requests.get(url, headers=headers)
    main_soup = BeautifulSoup(main_response.text, 'html.parser')

    ads = main_soup.find_all("div", class_="css-1sw7q4x")
    results = []

    for ad in ads:
        try:
            a_tag = ad.find("a", href=True)
            if not a_tag:
                continue
            ad_url = "https://www.olx.uz" + a_tag["href"]

            ad_resp = requests.get(ad_url, headers=headers)
            ad_soup = BeautifulSoup(ad_resp.text, "html.parser")

            title = ad_soup.find("h1").get_text(strip=True) if ad_soup.find("h1") else None
            price = ad_soup.find("h3").get_text(strip=True) if ad_soup.find("h3") else None
            desc_block = ad_soup.find("div", {"data-cy": "ad_description"})
            description = desc_block.get_text(strip=True) if desc_block else None

            results.append({
                "title": title,
                "price": price,
                "description": description,
                "url": ad_url
            })
            time.sleep(1)

        except Exception as e:
            print("Xatolik:", e)
            continue

    df = pd.DataFrame(results)

    df.to_csv(
        csv_file,
        mode='w' if first_page else 'a',  
        header=first_page,                 
        index=False,
        encoding="utf-8"
    )

    first_page = False
    print(f"{page}-sahifa yuklandi ✅")
