import requests
from bs4 import BeautifulSoup
import csv
import time

BASE_URL = "http://books.toscrape.com/catalogue/page-{}.html"
START_URL = "http://books.toscrape.com/catalogue/page-1.html"
OUTPUT_FILE = "books.csv"

# Columns: title, price, stock, rating, category, link
all_books = []

# Mapping rating text to numbers
RATING_MAP = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5
}

page = 1
while True:
    url = BASE_URL.format(page)
    print(f"Scraping page {page}: {url}")
    response = requests.get(url)
    if response.status_code != 200:
        print("No more pages or error occurred. Stopping.")
        break

    soup = BeautifulSoup(response.text, "lxml")
    books = soup.select("article.product_pod")

    if not books:
        break  # no more books

    for book in books:
        title_el = book.select_one("h3 a")
        title = title_el["title"] if title_el else ""

        link = "http://books.toscrape.com/catalogue/" + title_el["href"] if title_el else ""

        price_el = book.select_one(".price_color")
        price = price_el.get_text(strip=True).replace("£", "") if price_el else ""

        stock_el = book.select_one(".availability")
        stock = stock_el.get_text(strip=True) if stock_el else ""

        rating_class = book.select_one("p.star-rating")["class"][1] if book.select_one("p.star-rating") else ""
        rating = RATING_MAP.get(rating_class, 0)

        # Category requires fetching the book page
        category = ""
        try:
            book_page = requests.get(link)
            book_soup = BeautifulSoup(book_page.text, "lxml")
            breadcrumb = book_soup.select("ul.breadcrumb li a")
            if len(breadcrumb) >= 3:
                category = breadcrumb[2].get_text(strip=True)
        except:
            pass

        all_books.append({
            "title": title,
            "price": price,
            "stock": stock,
            "rating": rating,
            "category": category,
            "link": link
        })

    page += 1
    time.sleep(1)  # polite delay

# Save CSV
keys = ["title", "price", "stock", "rating", "category", "link"]
with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=keys)
    writer.writeheader()
    for book in all_books:
        writer.writerow(book)

print(f"Scraped {len(all_books)} books and saved to {OUTPUT_FILE}")
