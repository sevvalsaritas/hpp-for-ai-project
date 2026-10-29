"""
Phase 1: Naive Baseline Search (starter dosyası, TODO'lar dolduruldu)

Hikayenin başlangıç noktası: BİLEREK yavaş, saf Python bir arama.
  * Veri: MiniLM embedding'leri Python listesine (list of list of float) çevrilir.
  * Her sorgu x her doküman için eleman eleman cosine benzerliği hesaplanır.
  * Tüm skorlar sıralanır ve ilk k alınır.

Bu dosyayı profilleyip (profiling_template.py) darboğazları bulacağız.
Kavram: Hafta 1 -- Python yorumlayıcısının maliyeti (her + ve * bir bytecode,
her float bir heap nesnesi, her liste erişimi bir pointer takibi).
"""

import heapq
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_utils import benchmark, strip_results  # noqa: E402
from common import REF_N_DOCS, REF_N_QUERIES, TOP_K, load_embeddings, save_json  # noqa: E402


def cosine_similarity_python(a, b):
    """
    İki vektörün cosine benzerliğini SAF Python ile hesaplar (NumPy yok).

    Bilinçli verimsizlikler (profilde göreceğimiz):
      - Her çağrıda HEM sorgunun HEM dokümanın normu yeniden hesaplanıyor.
        Sorgu normu 19.671 kez, her doküman normu 100 kez boşa hesaplanır.
      - Döngünün her adımı yorumlayıcıda ~10 bytecode: indeksleme, çarpma,
        toplama, yeni float nesnesi oluşturma... SIMD/vektör birimi kullanılmaz.
    Kavram: GIL + yorumlayıcı yükü -> CPU'nun teorik FLOP kapasitesinin ~%0.1'i.
    """
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for i in range(len(a)):          # 384 adımlık iç döngü: zamanın büyük kısmı burada
        dot += a[i] * b[i]
        norm_a += a[i] * a[i]
        norm_b += b[i] * b[i]
    if norm_a == 0.0 or norm_b == 0.0:   # sıfır vektörde 0'a bölmeyi engelle
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def search_naive(queries, corpus, k=TOP_K):
    """
    Her sorgu için korpustaki en benzer k dokümanı bulur (hocanın iskeleti).

    Karmaşıklık: O(Q * N * d) benzerlik + O(Q * N log N) sıralama.
    Sıralama: k=5 için 19.671 elemanın TAMAMINI sıralamak gereksiz; ayrıca
    key=lambda her karşılaştırmada bir Python fonksiyon çağrısı demek.

    Args:
        queries: list of lists -- sorgu vektörleri
        corpus:  list of lists -- doküman vektörleri
    Returns:
        her sorgu için [(doc_index, similarity_score), ...] (azalan skor)
    """
    results = []
    for q_idx, query in enumerate(queries):
        scores = []
        for d_idx, doc in enumerate(corpus):
            sim = cosine_similarity_python(query, doc)
            scores.append((d_idx, sim))           # her skor için yeni bir tuple nesnesi

        # Skora göre azalan sırala, ilk k'yı al (tam sıralama: O(N log N))
        scores.sort(key=lambda x: x[1], reverse=True)
        results.append(scores[:k])

    return results


def search_python_optimized(queries, corpus, k=TOP_K):
    """
    Profilden sonra SAF PYTHON içinde yapılabilecek algoritmik iyileştirmeler.

    1. Normlar bir kez önceden hesaplanır (gereksiz 2/3 işi atar).
    2. Tam sıralama yerine heapq.nlargest: O(N log k).
    3. zip ile indekssiz iterasyon (a[i] erişimi yerine).
    Amaç: "algoritmik düzeltme ne kadar kazandırır, yorumlayıcı yükü ne kadar
    kalır?" sorusunu ölçmek. (Cevap: ~3x; asıl kazanç vektörizasyonda.)
    """
    doc_norms = [math.sqrt(sum(x * x for x in d)) or 1.0 for d in corpus]
    results = []
    for query in queries:
        qn = math.sqrt(sum(x * x for x in query)) or 1.0
        scores = ((sum(a * b for a, b in zip(query, doc)) / (qn * dn), d_idx)
                  for d_idx, (doc, dn) in enumerate(zip(corpus, doc_norms)))
        top = heapq.nlargest(k, scores)
        results.append([(d_idx, s) for s, d_idx in top])
    return results


def load_python_lists(n_docs=REF_N_DOCS, n_queries=REF_N_QUERIES):
    """Embedding'leri yükler ve Python listesine çevirir (baseline'ın veri yapısı)."""
    C, Q = load_embeddings(n_docs)
    if n_queries:
        Q = Q[:n_queries]
    # tolist(): float32 -> Python float (64-bit, kutulanmış nesne). Her sayı
    # artık ayrı bir heap nesnesi; ardışık bellekte DEĞİL. (Veri erişim darboğazı)
    return Q.tolist(), C.tolist()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    # Varsayılan = standart iş yükü (100 sorgu x tüm korpus). Tek ölçüm ~45-60 s
    # sürer ama baseline'ın final sistemle AYNI iş üzerinde ölçülmesi gerekir.
    ap.add_argument("--n-docs", type=int, default=REF_N_DOCS)
    ap.add_argument("--n-queries", type=int, default=REF_N_QUERIES)
    ap.add_argument("--runs", type=int, default=5)
    args = ap.parse_args()

    print("Embedding'ler yükleniyor...")
    queries, corpus = load_python_lists(args.n_docs, args.n_queries)
    print(f"Korpus: {len(corpus)} doküman x {len(corpus[0])} boyut")
    print(f"Sorgu:  {len(queries)}")

    # Saf Python'da JIT yok; 1 warmup (cache/allocator ısınması) + 5 ölçüm yeterli
    # ve tek ölçüm onlarca saniye sürdüğü için 3 warmup gereksiz zaman kaybı.
    print("\nNaive baseline çalışıyor (warmup=1, runs=%d)..." % args.runs)
    res = benchmark(search_naive, queries, corpus, TOP_K, warmup=1, runs=args.runs)
    print(f"Baseline medyan: {res['median']:.3f} s | sorgu başına {res['median'] / len(queries) * 1000:.1f} ms")
    print(f"İlk sorgu top-5: {res['results'][0]}")

    print("\nSaf Python + algoritmik düzeltmeler...")
    res_opt = benchmark(search_python_optimized, queries, corpus, TOP_K, warmup=1, runs=args.runs)
    print(f"Optimize Python medyan: {res_opt['median']:.3f} s "
          f"({res['median'] / res_opt['median']:.2f}x)")

    save_json({
        "n_docs": len(corpus), "n_queries": len(queries), "dim": len(corpus[0]), "k": TOP_K,
        "naive": strip_results(res),
        "python_optimized": strip_results(res_opt),
        "per_query_ms_naive": res["median"] / len(queries) * 1000,
        "baseline_results_first3": res["results"][:3],
    }, "phase1_baseline.json")
