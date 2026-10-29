"""
dataset/preprocess.py -- Ham arXiv kayıtlarını temizler, korpus/sorgu olarak ayırır,
istatistikleri raporlanabilir biçimde kaydeder.

Ön işleme adımları (her adımdan sonra kalan kayıt sayısı dataset_stats.json'a yazılır):
  1. Tekilleştirme: aynı arXiv id (sürüm eki atılmış) bir kez tutulur.
  2. Tarih filtresi: yayın tarihi 2025-01-01 .. 2026-12-31 dışındakiler atılır
     (proje şartı: 2025-2026 verisi).
  3. Metin temizliği: satır sonları ve çoklu boşluklar tek boşluğa; LaTeX
     komutları (\\mathbf, \\emph ...) ve $ { } karakterleri silinir. Matematik
     ifadesinin *içeriği* korunur (ör. "$O(n^2)$" -> "O(n^2)").
  4. Uzunluk filtresi: 50 kelimeden kısa özetler atılır (proje şartı: ort. >= 50 token).
  5. Karıştırma (seed=42): korpusun "ilk n" alt kümesi her zaman tüm 22 ayı
     temsil eden rastgele bir örneklem olsun diye. Ölçekleme testi buna dayanıyor.
  6. Sorgu ayırma: karışık listenin ilk 100 makalesi KORPUSTAN ÇIKARILIR ve
     başlıkları sorgu olarak kullanılır. Böylece sorgu kendi özetini bulamaz;
     gerçek bir "bu konuda benzer makaleler" araması simüle edilir.

Kullanım:  py -3.14 dataset/preprocess.py [--max-docs 20000]
"""

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA_DIR, EMBED_MODEL, SEED, load_jsonl, save_json  # noqa: E402

N_QUERIES = 100
MIN_WORDS = 50

# Önceden derlenmiş regex'ler: her kayıtta yeniden derlemek gereksiz maliyet olur.
RE_LATEX_CMD = re.compile(r"\\[a-zA-Z]+\*?")    # \mathbf, \emph, \textit ...
RE_LATEX_CHARS = re.compile(r"[${}\\]")         # kalan $ { } \ karakterleri
RE_SPACES = re.compile(r"\s+")


def clean_text(s):
    """Tek bir metni temizler (adım 3)."""
    s = RE_LATEX_CMD.sub("", s)
    s = RE_LATEX_CHARS.sub("", s)
    s = RE_SPACES.sub(" ", s)
    return s.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-docs", type=int, default=20000, help="korpus üst sınırı (proje: <= 20K)")
    args = ap.parse_args()

    raw = load_jsonl(DATA_DIR / "arxiv_raw.jsonl")
    steps = [{"adim": "ham kayit (API)", "kalan": len(raw)}]

    # 1. Tekilleştirme
    seen, uniq = set(), []
    for r in raw:
        if r["id"] not in seen:
            seen.add(r["id"])
            uniq.append(r)
    steps.append({"adim": "tekillestirme (arXiv id)", "kalan": len(uniq)})

    # 2. Tarih filtresi
    dated = [r for r in uniq if r["published"][:4] in ("2025", "2026")]
    steps.append({"adim": "tarih filtresi (2025-2026)", "kalan": len(dated)})

    # 3. Temizlik
    for r in dated:
        r["title"] = clean_text(r["title"])
        r["abstract"] = clean_text(r["abstract"])

    # 4. Uzunluk filtresi
    long_enough = [r for r in dated if len(r["abstract"].split()) >= MIN_WORDS]
    steps.append({"adim": f"uzunluk filtresi (>= {MIN_WORDS} kelime)", "kalan": len(long_enough)})

    # 5. Karıştırma (deterministik)
    random.Random(SEED).shuffle(long_enough)

    # 6. Sorgu / korpus ayrımı
    queries = long_enough[:N_QUERIES]
    corpus = long_enough[N_QUERIES:N_QUERIES + args.max_docs]
    steps.append({"adim": f"sorgu ayirma ({N_QUERIES} makale korpustan cikarildi)", "kalan": len(corpus)})

    keep = ("id", "title", "abstract", "published", "primary_category", "categories")
    with open(DATA_DIR / "corpus.jsonl", "w", encoding="utf-8") as f:
        for r in corpus:
            f.write(json.dumps({k: r[k] for k in keep}, ensure_ascii=False) + "\n")
    with open(DATA_DIR / "queries.jsonl", "w", encoding="utf-8") as f:
        for r in queries:
            d = {k: r[k] for k in keep}
            d["query"] = r["title"]           # arama metni = başlık
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # ---------------- İstatistikler ----------------
    words = np.array([len(r["abstract"].split()) for r in corpus])
    q_words = np.array([len(q["title"].split()) for q in queries])   # sorgu metni = başlık

    # Embedding modelinin kendi tokenizer'ı ile (WordPiece) token sayısı:
    # MiniLM en fazla 256 token okur; daha uzun özetler kırpılır. Bunu raporda
    # dürüstçe belirtmek için kırpılan oranı ölçüyoruz.
    tok_stats = {}
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(EMBED_MODEL)
        lens = np.array([len(t) for t in tok([r["abstract"] for r in corpus],
                                             add_special_tokens=True)["input_ids"]])
        tok_stats = {
            "mean": float(lens.mean()), "median": float(np.median(lens)),
            "min": int(lens.min()), "max": int(lens.max()),
            "frac_truncated_over_256": float((lens > 256).mean()),
        }
    except Exception as e:                    # tokenizer indirilemezse istatistik atlanır
        tok_stats = {"error": str(e)}

    months = Counter(r["published"][:7] for r in corpus)
    prim = Counter(r["primary_category"] for r in corpus)
    in_target = Counter(c for r in corpus for c in set(r["categories"])
                        if c in ("cs.AI", "cs.CL", "cs.LG"))

    stats = {
        "kaynak": "arXiv API (export.arxiv.org/api/query), kategoriler cs.AI / cs.CL / cs.LG, 2025-01 .. 2026-10",
        "n_corpus": len(corpus),
        "n_queries": len(queries),
        "preprocessing_steps": steps,
        "abstract_words": {"mean": float(words.mean()), "median": float(np.median(words)),
                           "min": int(words.min()), "max": int(words.max()),
                           "std": float(words.std())},
        "abstract_wordpiece_tokens": tok_stats,
        "query_words": {"mean": float(q_words.mean()), "min": int(q_words.min()),
                        "max": int(q_words.max())},
        "months": dict(sorted(months.items())),
        "primary_category_top10": dict(prim.most_common(10)),
        "target_category_coverage": dict(in_target),
        "samples_corpus": [{k: r[k] for k in ("id", "title", "published", "primary_category")}
                           | {"abstract": r["abstract"][:400] + "..."} for r in corpus[:3]],
        "samples_queries": [q["title"] for q in queries[:5]],
    }
    save_json(stats, "dataset_stats.json")
    print(json.dumps({k: stats[k] for k in ("n_corpus", "n_queries", "preprocessing_steps",
                                            "abstract_words", "abstract_wordpiece_tokens")},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
