"""
common.py -- Tüm aşamaların paylaştığı yardımcılar (yollar, veri yükleme, top-k).

Neden ayrı bir dosya?
    Her phase dosyası aynı veriyi (korpus, sorgular, embedding'ler) yüklüyor.
    Yükleme mantığını tek yerde tutarsak bütün benchmark'lar *aynı* veriyle
    çalışır; bu da karşılaştırmaların adil olmasının ön koşuludur.
"""

import json
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Yollar: dosyanın kendi konumundan hesaplanır, böylece scriptler hangi
# klasörden çalıştırılırsa çalıştırılsın aynı dosyaları bulur.
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"            # arXiv API'den gelen ham XML sayfaları (önbellek)
RESULTS_DIR = ROOT / "results"        # ham benchmark sonuçları (CSV/JSON)
FIGURES_DIR = ROOT / "figures"        # rapordaki PNG figürler

for _d in (DATA_DIR, RAW_DIR, RESULTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Proje genelinde sabitler -- tek yerden değiştirilebilsin diye burada.
TOP_K = 5             # her sorgu için döndürülen sonuç sayısı
SEED = 42             # tekrarlanabilirlik için tüm rastgelelik bu tohumla başlar
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # 384 boyutlu dense embedding

# STANDART İŞ YÜKÜ: 100 sorgu x tüm korpus (19.671 doküman).
# Fazlar arası karşılaştırmanın anlamlı olması için her ana ölçüm bu iş yüküyle
# yapılır. Bundan sapan deneyler (ölçekleme eğrileri, sorgu sayısını değişken
# olarak tarayan deneyler, kavram gösterimleri) gerekçesini kendi dosyasında yazar.
REF_N_DOCS = None     # None = korpusun tamamı
REF_N_QUERIES = 100


DEFAULT_THREADS = 8   # Phase 2 ölçümlerinin sabit thread sayısı (gerekçe aşağıda)


def configure_threads(n=None):
    """
    BLAS (OpenBLAS) ve Numba thread sayısını SABİTLER. Tüm benchmark scriptleri başta çağırır.

    Neden gerekli? Bu laptop hibrit bir işlemciye sahip (Core Ultra 7 255H:
    6 performans + 8 verimlilik + 2 düşük güç çekirdeği). OpenBLAS varsayılan
    olarak 16 thread açar ve işi eşit böler; hesap en yavaş (LP-E) çekirdeği
    beklediği için 100 sorguluk arama 8 thread'e göre ~7 kat YAVAŞLIYOR (Phase 3
    thread taraması, Amdahl + yük dengesizliği). 8 thread, ölçtüğümüz tüm
    boyutlarda en iyiye yakın ve kararlı sonuç verdi.
    Ortam değişkeniyle değiştirilebilir:  set HPP_THREADS=16
    """
    import os
    n = int(n or os.environ.get("HPP_THREADS", DEFAULT_THREADS))
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=n)        # bağlam yöneticisi olmadan: süreç boyunca geçerli
    try:
        import numba
        numba.set_num_threads(min(n, numba.config.NUMBA_NUM_THREADS))
    except ImportError:
        pass
    return n


def scaling_sizes(n_total):
    """
    Ölçekleme / bellek testleri için korpus boyutlarını seçer.

    Gerekçe: Noktalar yaklaşık 2x aralıklı (geometrik) seçilir: 1K, 2.5K, 5K,
    10K ve eldeki toplam veri. Geometrik aralık log-log grafikte eşit aralıklı
    noktalar verir; böylece eğimden (1 = doğrusal, 2 = karesel) ölçekleme
    davranışı doğrudan okunur. Hocanın önerdiği 1K/5K/10K/20K ile de uyumludur.
    """
    base = [1000, 2500, 5000, 10000, 20000]
    # Toplama çok yakın bir boyutu atla (ör. 10.000 ve 10.050 neredeyse aynı nokta olur).
    sizes = [s for s in base if s < 0.8 * n_total]
    sizes.append(n_total)            # tüm veri her zaman son nokta
    return sizes


# --------------------------------------------------------------------------
# Veri yükleme
# --------------------------------------------------------------------------
def load_jsonl(path):
    """JSON Lines dosyasını (her satır bir kayıt) liste olarak okur."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_corpus(n=None):
    """
    Temizlenmiş korpusu yükler. `n` verilirse ilk n doküman döner.

    Korpus ön işlemede rastgele karıştırıldığı için "ilk n" alt kümesi
    tüm zaman aralığını temsil eden rastgele bir örneklemdir; ölçekleme
    testinde 1K/2.5K/5K... alt kümeleri bu yüzden adil karşılaştırılabilir.
    """
    docs = load_jsonl(DATA_DIR / "corpus.jsonl")
    return docs[:n] if n else docs


def load_queries():
    """Korpusta OLMAYAN makalelerin başlıklarından oluşan sorgu kümesini yükler."""
    return load_jsonl(DATA_DIR / "queries.jsonl")


def load_embeddings(n=None, dtype=np.float32):
    """
    Dense (MiniLM, 384 boyut) embedding'leri yükler.

    float32 kullanıyoruz: float64'e göre yarı bellek = yarı bellek bant
    genişliği ihtiyacı. Benzerlik araması bant genişliği sınırlı bir iş
    olduğundan bu doğrudan hız demektir.
    Dönüş: (corpus_emb [n x d], query_emb [q x d])
    """
    C = np.load(DATA_DIR / "corpus_embeddings.npy").astype(dtype, copy=False)
    Q = np.load(DATA_DIR / "query_embeddings.npy").astype(dtype, copy=False)
    if n:
        C = C[:n]
    # C-contiguous (satır-öncelikli) yerleşim: BLAS ve Numba döngüleri satırları
    # ardışık okuduğunda önbellek (cache) dostu olur.
    return np.ascontiguousarray(C), np.ascontiguousarray(Q)


def load_tfidf(n=None):
    """Sparse TF-IDF matrislerini (CSR formatında) yükler."""
    from scipy import sparse
    X = sparse.load_npz(DATA_DIR / "tfidf_corpus.npz").tocsr()
    Qs = sparse.load_npz(DATA_DIR / "tfidf_queries.npz").tocsr()
    if n:
        X = X[:n]
    return X, Qs


def normalize_rows(X):
    """
    Her satırı birim uzunluğa (L2 norm = 1) getirir.

    Neden? Normlar 1 olunca cosine(a, b) = a · b olur. Böylece arama tek bir
    matris çarpımına (Q @ C.T) indirgenir ve BLAS'a devredilebilir.
    Normlar arama anında değil, indeksleme sırasında BİR KEZ hesaplanır.
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0          # sıfır vektörde 0'a bölmeyi engelle
    return X / norms


# --------------------------------------------------------------------------
# Top-k seçimi
# --------------------------------------------------------------------------
def topk_from_scores(scores, k=TOP_K):
    """
    Skor matrisinden (q x n) her satır için en yüksek k sonucu seçer.

    Tam sıralama O(n log n) iken argpartition ortalama O(n)'dir: önce k
    adayı bölümleyip sadece o k elemanı sıralıyoruz. n=10.000, k=5 için
    bu, sıralama maliyetini neredeyse yok eder (Phase 1'deki 2. darboğaz).
    Dönüş: her sorgu için [(doc_idx, score), ...] listesi (azalan skor).
    """
    scores = np.atleast_2d(scores)
    k = min(k, scores.shape[1])
    # -scores: argpartition küçükten büyüğe çalışır, biz en büyükleri istiyoruz.
    idx = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    part = np.take_along_axis(scores, idx, axis=1)
    order = np.argsort(-part, axis=1, kind="stable")
    idx = np.take_along_axis(idx, order, axis=1)
    part = np.take_along_axis(part, order, axis=1)
    return [[(int(i), float(s)) for i, s in zip(ri, rs)] for ri, rs in zip(idx, part)]


# --------------------------------------------------------------------------
# Sonuç kaydetme
# --------------------------------------------------------------------------
def save_json(obj, name):
    """results/ altına JSON kaydeder (NumPy tiplerini otomatik çevirir)."""
    path = RESULTS_DIR / name

    def _conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_conv)
    print(f"Kaydedildi: {path}")
    return path


def load_json(name):
    """results/ altındaki bir JSON sonucunu okur."""
    with open(RESULTS_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def hardware_info():
    """Raporun 'donanım/yazılım ortamı' bölümü için sistem bilgisini toplar."""
    import os
    import platform
    import sys
    info = {
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "cpu": platform.processor(),
        "logical_cores": os.cpu_count(),
        "numpy": np.__version__,
    }
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 1024**3, 1)
        info["physical_cores"] = psutil.cpu_count(logical=False)
    except ImportError:
        pass
    for mod in ("scipy", "sklearn", "numba", "dask", "ray"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = "yok"
    try:
        from threadpoolctl import threadpool_info
        info["blas"] = [(d.get("internal_api"), d.get("num_threads")) for d in threadpool_info()]
    except ImportError:
        pass
    return info
