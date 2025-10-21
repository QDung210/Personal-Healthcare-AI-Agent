import asyncio
import json
from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig
from bs4 import BeautifulSoup

START_URL = "https://nhathuoclongchau.com.vn/bai-viet/phong-chua-benh/kien-thuc-y-khoa"

async def get_article_urls(crawler, num_clicks):
    js = f"""
        for(let i=0;i<{num_clicks};i++){{
            document.querySelector('div.inline-flex.cursor-pointer')?.click();
            await new Promise(r=>setTimeout(r,2000));
        }}
    """ if num_clicks > 0 else ""
    
    result = await crawler.arun(
        url=START_URL,
        config=CrawlerRunConfig(js_code=[js] if js else None, wait_for="css:.omd\\:bg-white")
    )
    soup = BeautifulSoup(result.html, "html.parser")
    articles = soup.select("div.omd\\:bg-white a[href*='/bai-viet/']")
    
    return {f"https://nhathuoclongchau.com.vn{a['href']}" for a in articles if a.get('href', '').startswith('/bai-viet/')}

async def crawl_article(crawler, url):
    result = await crawler.arun(url=url, config=CrawlerRunConfig(wait_for="css:h1.text-heading2"))
    soup = BeautifulSoup(result.html, "html.parser")
    h1 = soup.find("h1", class_=lambda x: x and "text-heading2" in x)
    article = soup.find("div", {"data-theme-element": "article"})
    
    return {
        "url": url,
        "title": h1.text.strip() if h1 else "No title",
        "content": article.get_text(separator="\n", strip=True) if article else "No content"
    }

async def main():
    async with AsyncWebCrawler(config=BrowserConfig(headless=True, verbose=False)) as crawler:
        with open("ntlc_crawl.json", "w", encoding="utf-8") as f:
            f.write("[\n")
        
        crawled_urls = set()
        clicks = 0
        
        while True:
            all_urls = await get_article_urls(crawler, clicks)
            new_urls = all_urls - crawled_urls
            
            if not new_urls:
                break
            
            sem = asyncio.Semaphore(10)
            async def crawl_with_sem(url):
                async with sem:
                    data = await crawl_article(crawler, url)
                    with open("ntlc_crawl.json", "a", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                        f.write(",\n")
                    return url
            
            results = await asyncio.gather(*[crawl_with_sem(url) for url in new_urls])
            crawled_urls.update(results)
            print(f"Đã crawl được {len(crawled_urls)} bài")
            clicks += 1
        
        with open("ntlc_crawl.json", "rb+") as f:
            f.seek(-2, 2)
            f.truncate()
            f.write(b"\n]")

if __name__ == "__main__":
    asyncio.run(main())