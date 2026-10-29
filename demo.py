"""
demo.py -- Canlı demo: arXiv 2025-26 özetlerinde anlamsal benzerlik araması.

Kullanım:
    py -3.14 demo.py                          # etkileşimli: sorgu yaz, Enter
    py -3.14 demo.py --query "LLM agents for code repair"
    py -3.14 demo.py --compare                # aynı sorguyu naive baseline ile de çalıştır
    py -3.14 demo.py --svd 256                # boyut indirgemeli mod

Ekranda iki süre ayrı gösterilir:
  * embedding süresi: MiniLM'in metni vektöre çevirmesi (FAQ gereği benchmark dışı)
  * arama süresi:     bizim optimize ettiğimiz kısım (BLAS + Numba top-k)
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Canlı demoda ekran sade kalsın: Hugging Face / transformers'ın zararsız uyarılarını
# (kimliksiz istek, ağırlık yükleme raporu, ilerleme çubuğu) transformers import
# edilmeden ÖNCE susturuyoruz. Model önbellekteyse çevrimdışı çalışıyoruz: sunumda
# internet olmasa da demo açılır ve açılışta ağ isteği beklenmez.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
_hf_cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
if (_hf_cache / "models--sentence-transformers--all-MiniLM-L6-v2").exists():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "phase4_final"))
sys.path.insert(0, str(ROOT / "phase1_baseline"))

import numpy as np  # noqa: E402

from common import TOP_K, load_corpus, load_embeddings  # noqa: E402
from search_engine import SimilaritySearchEngine  # noqa: E402


def show(results_idx, results_sc, corpus):
    """Top-k sonuçları başlık, tarih, kategori ve skorla yazdırır."""
    for rank, (i, s) in enumerate(zip(results_idx, results_sc), 1):
        d = corpus[i]
        print(f"  {rank}. [{s:.3f}] {d['title'][:95]}")
        print(f"       arXiv:{d['id']}  {d['published'][:10]}  {d['primary_category']}")


def run_query(engine, corpus, text, compare=False):
    """Tek bir metin sorgusunu çalıştırır ve süreleri gösterir."""
    idx, sc, t_emb, t_search = engine.search_text([text], TOP_K)
    # İlk çağrı soğuk olabilir (önbellek, BLAS thread havuzu); gösterilen süre için
    # aynı aramayı birkaç kez tekrar edip medyanı alıyoruz (benchmark kuralıyla tutarlı).
    Qn = engine.prepare_queries(engine.encode([text]))
    ts = []
    for _ in range(7):
        t0 = time.perf_counter()
        engine.search_prepared(Qn, TOP_K)
        ts.append(time.perf_counter() - t0)
    t_med = float(np.median(ts))
    print(f"\nSorgu: \"{text}\"")
    print(f"  embedding: {t_emb * 1e3:.1f} ms | arama ({engine.n_docs:,} doküman x {engine.dim} boyut): "
          f"{t_med * 1e3:.3f} ms (medyan, 7 tekrar)")
    show(idx[0], sc[0], corpus)

    if compare:
        from naive_search import cosine_similarity_python, search_naive  # noqa: F401
        C, _ = load_embeddings()
        q = engine.encode([text])[0].tolist()
        t0 = time.perf_counter()
        base = search_naive([q], C.tolist(), TOP_K)
        t_base = time.perf_counter() - t0
        msg = f"  naive saf Python: {t_base * 1e3:.0f} ms  -> {t_base / t_med:,.0f}x daha yavaş"
        if engine.svd is None:            # SVD modunda uzaylar farklı; birebir eşitlik beklenmez
            same = [i for i, _ in base[0]] == list(idx[0])
            msg += f" | aynı top-{TOP_K}: {same}"
        print(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", type=str, default=None)
    ap.add_argument("--compare", action="store_true", help="naive baseline ile karşılaştır")
    ap.add_argument("--svd", type=int, default=None, help="TruncatedSVD bileşen sayısı (ops.)")
    args = ap.parse_args()

    print("İndeks yükleniyor...")
    corpus = load_corpus()
    C, _ = load_embeddings()
    engine = SimilaritySearchEngine(C, svd_k=args.svd)
    print(f"Hazır: {engine.n_docs:,} makale, {engine.dim} boyut, {engine.threads} thread, "
          f"indeksleme {engine.index_time_s * 1e3:.0f} ms")
    print("MiniLM modeli yükleniyor (ilk seferde birkaç saniye)...")
    engine.encode(["warmup"])                 # modeli ve BLAS'ı ısıt: demo sırasında bekleme olmasın
    engine.search(engine.encode(["warmup"]))

    if args.query:
        run_query(engine, corpus, args.query, args.compare)
        return
    print("\nSorgu yazın (çıkmak için boş satır):")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            break
        run_query(engine, corpus, text, args.compare)


if __name__ == "__main__":
    main()
