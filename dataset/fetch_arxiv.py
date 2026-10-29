"""
dataset/fetch_arxiv.py -- arXiv API'den 2025-2026 cs.AI / cs.CL / cs.LG özetlerini çeker.

Tasarım kararları (raporda "Veri seti" bölümünde anlatılıyor):
  * AY AY çekiyoruz: Tek bir "en yeniler" sorgusu sadece son birkaç haftayı
    getirirdi (bu kategorilerde ayda ~6-7 bin makale var). Her aydan eşit
    sayıda alarak 22 aylık dönemi temsil eden bir korpus elde ediyoruz.
  * Rate limit: arXiv "3 saniyede en fazla 1 istek" istiyor. Biz 3.5 sn
    bekliyoruz ve hata gelirse artan sürelerle (backoff) tekrar deniyoruz.
  * Önbellek: Her sayfa data/raw/<ay>_p<sayfa>.xml olarak kaydedilir. Script
    tekrar çalıştırılırsa var olan sayfalar ATLANIR -> API'ye gereksiz yük
    bindirmeyiz ve kesintide kaldığımız yerden devam ederiz.
  * Ölçekleme: --pages 1 ile ~6.600 ham kayıt (geliştirme), --pages 2 ile
    ~13.200 ham kayıt (ölçekleme testi). İkinci çalıştırma sadece 2. sayfaları indirir.

Kullanım:
    py -3.14 dataset/fetch_arxiv.py --pages 1
    py -3.14 dataset/fetch_arxiv.py --pages 2
"""

import argparse
import calendar
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA_DIR, RAW_DIR  # noqa: E402

API_URL = "http://export.arxiv.org/api/query"
CATEGORY_QUERY = "(cat:cs.AI OR cat:cs.CL OR cat:cs.LG)"
PAGE_SIZE = 300          # tek istekte kayıt sayısı; arXiv büyük sayfalarda sık hata veriyor
DELAY_S = 3.5            # istekler arası bekleme (arXiv kuralı: >= 3 sn)
MAX_RETRIES = 6

# Atom XML ad alanları (namespace): ElementTree etiketleri bunlarla arar.
NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
}


def month_list(start=(2025, 1), end=(2026, 10)):
    """Ocak 2025'ten Ekim 2026'ya kadar (yıl, ay) çiftlerini üretir."""
    y, m = start
    while (y, m) <= end:
        yield y, m
        m += 1
        if m == 13:
            y, m = y + 1, 1


def build_url(year, month, start):
    """Belirli bir ay ve sayfa için arXiv API URL'sini oluşturur."""
    last_day = calendar.monthrange(year, month)[1]
    date_q = f"submittedDate:[{year}{month:02d}010000 TO {year}{month:02d}{last_day}2359]"
    params = {
        "search_query": f"{CATEGORY_QUERY} AND {date_q}",
        "start": start,
        "max_results": PAGE_SIZE,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    return API_URL + "?" + urllib.parse.urlencode(params)


def fetch_page(url):
    """
    Bir sayfayı indirir; hata veya boş yanıt gelirse bekleyip tekrar dener.

    Bu iş tamamen I/O-bound: zamanın neredeyse tamamı ağ beklemesi. Yine de
    paralel (asyncio/thread) indirme YAPMIYORUZ, çünkü darboğaz bizim CPU'muz
    değil arXiv'in izin verdiği istek hızı. Kurallara uymak burada performanstan önce gelir.
    """
    req = urllib.request.Request(url, headers={"User-Agent": "YZM513-course-project/1.0"})
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = resp.read()
            root = ET.fromstring(body)
            n_entries = len(root.findall("atom:entry", NS))
            if n_entries > 0:
                return body, n_entries
            # arXiv bazen yük altında 200 + boş liste döndürüyor: tekrar dene.
            print(f"    boş yanıt, tekrar denenecek ({attempt + 1}/{MAX_RETRIES})")
        except (urllib.error.URLError, ET.ParseError, TimeoutError) as e:
            print(f"    hata: {e} ({attempt + 1}/{MAX_RETRIES})")
        time.sleep(DELAY_S * (2 ** attempt))   # üstel geri çekilme (backoff)
    raise RuntimeError(f"Sayfa indirilemedi: {url}")


def parse_entries(xml_bytes):
    """Atom XML'den makale kayıtlarını (id, başlık, özet, tarih, kategoriler) çıkarır."""
    root = ET.fromstring(xml_bytes)
    out = []
    for e in root.findall("atom:entry", NS):
        prim = e.find("arxiv:primary_category", NS)
        out.append({
            # ".../abs/2504.13875v2" -> "2504.13875" (sürüm eki atılır, tekilleştirme için)
            "id": e.findtext("atom:id", "", NS).rsplit("/abs/", 1)[-1].split("v")[0],
            "title": e.findtext("atom:title", "", NS),
            "abstract": e.findtext("atom:summary", "", NS),
            "published": e.findtext("atom:published", "", NS),
            "primary_category": prim.get("term") if prim is not None else "",
            "categories": [c.get("term") for c in e.findall("atom:category", NS)],
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=1, help="her ay için indirilecek sayfa sayısı")
    args = ap.parse_args()

    months = list(month_list())
    print(f"{len(months)} ay x {args.pages} sayfa x {PAGE_SIZE} kayıt hedefleniyor")

    for (y, m) in months:
        for p in range(args.pages):
            path = RAW_DIR / f"{y}-{m:02d}_p{p}.xml"
            if path.exists():                      # önbellekte var -> API'ye gitme
                continue
            url = build_url(y, m, p * PAGE_SIZE)
            body, n = fetch_page(url)
            path.write_bytes(body)
            print(f"  {y}-{m:02d} sayfa {p}: {n} kayıt -> {path.name}")
            time.sleep(DELAY_S)                    # rate limit'e saygı

    # Tüm önbellek sayfalarını tek bir JSONL'ye birleştir (ön işlemenin girdisi).
    records = []
    for path in sorted(RAW_DIR.glob("*.xml")):
        for r in parse_entries(path.read_bytes()):
            r["source_page"] = path.stem
            records.append(r)
    out = DATA_DIR / "arxiv_raw.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Toplam ham kayıt: {len(records)} -> {out}")


if __name__ == "__main__":
    main()
