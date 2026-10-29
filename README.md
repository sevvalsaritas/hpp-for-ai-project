# Yüksek Başarımlı Benzerlik Arama Motoru (YZM 513 · Proje 1)

**Şevval Özlem Sarıtaş** · İstanbul Medeniyet Üniversitesi · Güz 2026
Kod deposu: https://github.com/sevvalsaritas/hpp-for-ai-project

arXiv'deki 2025-2026 cs.AI / cs.CL / cs.LG özetlerinden oluşan ~20K dokümanlık bir korpusta
top-5 anlamsal benzerlik araması. Hocanın starter kodu üzerine inşa edildi: saf Python baseline →
profil → BLAS / Numba → paralellik → final sistem. PDF rapor ayrıca teslim edilmiştir; tüm
figürler `figures/`, ham ölçümler `results/` altındadır.

## Kurulum

Windows + Python 3.14 üzerinde test edildi. GPU gerekmez.

```powershell
py -3.14 -m pip install -r requirements.txt
```

## Hızlı başlangıç

```powershell
py -3.14 demo.py                                   # etkileşimli canlı demo
py -3.14 demo.py --query "LLM agents for automated code repair" --compare
py -3.14 demo.py --svd 256                         # boyut indirgemeli mod
```

`--compare` aynı sorguyu naive saf Python baseline ile de çalıştırır, hızlanmayı gösterir ve iki sonucun
aynı olduğunu doğrular.

## Her şeyi yeniden üretme

```powershell
py -3.14 run_all.py               # veri -> tüm benchmark'lar -> figürler
py -3.14 run_all.py --skip-data   # veri hazırsa
py -3.14 run_all.py --only phase3 # tek bir adım
```

Adımlar ve yaklaşık süreler (Core Ultra 7 255H):

| Adım | Script | Çıktı |
|---|---|---|
| Veri çekme (önbellekli, ~4 dk ilk sefer) | `dataset/fetch_arxiv.py --pages 3` | `data/raw/*.xml`, `data/arxiv_raw.jsonl` |
| Ön işleme + istatistik | `dataset/preprocess.py` | `data/corpus.jsonl`, `data/queries.jsonl`, `results/dataset_stats.json` |
| Embedding (~8 dk, benchmark dışı) | `dataset/embed.py` | `data/*_embeddings.npy`, `data/tfidf_*.npz` |
| Phase 1: profil + baseline | `phase1_baseline/profiling_template.py`, `naive_search.py` | Şekil 1, `results/phase1_baseline.json` |
| Phase 2A: sparse vs dense | `phase2_matrix_ops/sparse_dense_compare.py` | Şekil 2 |
| Phase 2B: SVD / PCA | `phase2_matrix_ops/svd_power_iteration.py` | Şekil 3 |
| Phase 2C: Python / NumPy / Numba | `phase2_matrix_ops/vectorise_vs_loop.py` | Şekil 4 |
| Phase 3: paralellik | `phase3_parallelism/run_phase3.py` | Şekil 5 |
| Phase 4: final, ölçekleme, doğruluk | `phase4_final/run_phase4.py` | Şekil 6, 7 |
| Bonus: ANN, roofline | `bonus/ann_compare.py`, `bonus/roofline.py` | Ek A, B |

Arama süreleri gürültüye duyarlıdır. Ölçüm sırasında laptop şarja takılı olsun ve ağır uygulamaları
kapatın.

## Klasör yapısı

```
benchmark_utils.py      hocanın yardımcıları + medyan, auto_runs, eşitlik-duyarlı doğrulama
common.py               yollar, veri yükleme, thread ayarı (configure_threads), top-k
plotstyle.py            figürlerin ortak stili
dataset/                arXiv çekme, ön işleme, embedding
phase1_baseline/        naive saf Python arama + cProfile / line_profiler
phase2_matrix_ops/      sparse/dense çekirdekleri, kendi güç iterasyonu SVD'miz, 4'lü vektörizasyon
phase3_parallelism/     BLAS, Numba, hibrit, ThreadPool, ProcessPool(+shm), Dask, Ray, asyncio
phase4_final/           SimilaritySearchEngine + ölçekleme / doğruluk / tüm sürümler
bonus/                  kendi LSH ve IVF uygulamalarımız, roofline
demo.py                 canlı demo
results/                ham benchmark verisi (CSV/JSON, tüm tekrarlar)
figures/                rapor figürleri (kodla üretilir)
```

## Benchmark kuralları

- Her ölçüm warmup + en az 5 tekrar içerir ve **medyan** raporlanır.
- 50 ms'den kısa işlerde tekrar sayısı 25'e çıkarılır (Windows'ta thread uyandırma gürültüsü).
- BLAS ve Numba **8 thread'e** sabitlenir (`common.configure_threads`; değiştirmek için
  `set HPP_THREADS=16`). Gerekçe, raporun 4.1. bölümündeki taramadır: bu hibrit işlemcide 16 thread daha yavaştır.
- Thread ayarları ölçüm *dışında* yapılır.
- Her optimize sürüm baseline ile 1e-5 toleransta doğrulanır.

## Veri seti notu

`data/` klasörü büyük olduğu için depoda yer almayabilir. `run_all.py` veriyi arXiv API'den
yeniden üretir. API'nin "3 saniyede 1 istek" kuralına uyulur ve her sayfa önbelleğe alınır.
