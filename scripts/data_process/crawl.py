import argparse
import json
import os
import random
import time
from urllib.parse import quote

import pypinyin
import requests
from bs4 import BeautifulSoup

# ---------- 配置常量 ----------
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.8,en-US;q=0.5,en;q=0.3",
}
TIMEOUT = 10
RETRY_TIMES = 3
DOWNLOAD_DELAY = 1
PAGE_DELAY = 2
PER_PAGE = 35


def chinese_to_pinyin(text: str) -> str:
    """
    将中文文本转换为拼音，非中文部分保留
    """
    result = []
    for char in text:
        if "一" <= char <= "鿿":
            pys = pypinyin.pinyin(char, style=pypinyin.NORMAL)
            if pys:
                result.append(pys[0][0])
        else:
            result.append(char)
    return "".join(result)


def has_chinese(text: str) -> bool:
    """
    检查是否包含中文字符
    """
    return any("一" <= c <= "鿿" for c in text)


def download_image(url, save_path):
    """
    下载单张图片，返回是否成功
    """
    for attempt in range(RETRY_TIMES):
        try:
            headers = HEADERS.copy()
            headers["Referer"] = url
            resp = requests.get(url, headers=headers, timeout=TIMEOUT, stream=True)
            if resp.status_code == 200:
                # 根据 Content-Type 确定扩展名
                content_type = resp.headers.get("content-type", "").lower()
                if "jpeg" in content_type or "jpg" in content_type:
                    ext = ".jpg"
                elif "png" in content_type:
                    ext = ".png"
                elif "webp" in content_type:
                    ext = ".webp"
                else:
                    ext = ".jpg"
                actual_path = save_path.rsplit(".", 1)[0] + ext
                with open(actual_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True
        except Exception as e:
            print(f"Download attempt {attempt+1}/{RETRY_TIMES} failed: {e}")
        time.sleep(1)
    return False


def search_bing_images(keyword, first, count):
    """
    获取必应图片搜索结果中的原始图片URL列表
    """
    encoded_keyword = quote(keyword)
    url = f"https://www.bing.com/images/search?q={encoded_keyword}&first={first}&count={count}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        print(f"Failed to fetch search page: {e}")
        return []

    image_urls = []
    # 提取 a.iusc 标签中的 murl
    for tag in soup.find_all("a", class_="iusc"):
        m_attr = tag.get("m")
        if not m_attr:
            continue
        try:
            data = json.loads(m_attr)
            murl = data.get("murl")
            if murl and murl.startswith("http"):
                image_urls.append(murl)
        except json.JSONDecodeError:
            continue

    # 降级方案：从 img 标签获取
    if not image_urls:
        for img in soup.select("img.mimg, img.img"):
            img_url = img.get("data-src") or img.get("src")
            if img_url and img_url.startswith("http") and "data:image" not in img_url:
                if "w=60" not in img_url and "w=100" not in img_url:
                    image_urls.append(img_url)

    # 去重
    unique = []
    for u in image_urls:
        if u not in unique:
            unique.append(u)
    return unique


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download images from Bing Image Search")
    parser.add_argument("name", type=str, help="Search keyword (e.g., '尖嘴钳')")
    parser.add_argument("--num", type=int, default=50, help="Number of images to download")
    parser.add_argument("--output", type=str, default="./data/downloads", help="Output directory (default: ../data/downloads)")
    args = parser.parse_args()

    keyword = args.name
    target_num = args.num
    save_dir = args.output

    os.makedirs(save_dir, exist_ok=True)

    print(f"Searching and downloading {target_num} images for '{keyword}'")
    print(f"Save directory: {save_dir}\n")

    downloaded = 0
    page = 0
    seen_urls = set()

    while downloaded < target_num:
        first = page * PER_PAGE
        print(f"Fetching page {page+1} (start index {first})...")
        urls = search_bing_images(keyword, first, PER_PAGE)
        if not urls:
            print("No more images found.")
            break

        new_urls = [u for u in urls if u not in seen_urls]
        if not new_urls:
            print("All images on this page already seen, moving to next page.")
            page += 1
            time.sleep(PAGE_DELAY)
            continue

        print(f"Found {len(new_urls)} new image URLs on this page, downloading...")
        for idx, img_url in enumerate(new_urls):
            if downloaded >= target_num:
                break
            # 文件名：name_序号.jpg
            if has_chinese(keyword):
                keyword_pinyin = chinese_to_pinyin(keyword)
                keyword_pinyin = keyword_pinyin.replace(" ", "")
                file_name = f"{keyword_pinyin}_{downloaded+1}.jpg"
            else:
                file_name = f"{keyword}_{downloaded+1}.jpg"
            save_path = os.path.join(save_dir, file_name)
            print(f"Downloading [{downloaded+1}/{target_num}]: {img_url[:80]}...")
            success = download_image(img_url, save_path)
            if success:
                downloaded += 1
                seen_urls.add(img_url)
                print(f"  Saved to: {file_name}")
            else:
                print(f"  Failed to download, skipped.")
            time.sleep(DOWNLOAD_DELAY + random.uniform(0, 0.8))

        page += 1
        time.sleep(PAGE_DELAY)

    print(f"\nDownload completed. Successfully downloaded {downloaded} images to '{save_dir}'")
