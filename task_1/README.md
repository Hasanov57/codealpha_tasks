# CodeAlpha Web Scraping Task — Books Dataset

**Project:** Web Scraping Book Data  
**Source:** [https://books.toscrape.com](https://books.toscrape.com)  
**Author:** Samir Hasanov  

## Overview
This project scrapes book information from an open e-commerce website.  
Collected fields:
- Title  
- Price  
- Availability
- Rating
- Category
- Link

The script uses `requests` to fetch HTML and `BeautifulSoup` to parse it,  
then saves data to `books.csv`.
