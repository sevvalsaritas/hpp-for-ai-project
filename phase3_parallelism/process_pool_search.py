"""
Phase 3: ProcessPoolExecutor Stratejisi (starter dosyası, tamamlandı)

Ne zaman doğru seçim? CPU-bound SAF PYTHON (GIL thread'leri bloklar; süreçlerin
her birinin kendi yorumlayıcısı ve kendi GIL'i vardır).
Bedeli: süreçler belleği paylaşmaz -> veri pickle ile kopyalanır. Windows'ta
süreçler "spawn" ile başlar (sıfırdan Python + import) -> başlatma yükü yüksek.

Bu dosyada üç sürüm var:
  * search_processpool_bad  : her sorgu ayrı görev + korpus HER görevde pickle (starter)
  * search_processpool_good : sorgular parçalara bölünür, korpus parça başına bir kez (starter)
  * search_processpool_shm  : NumPy korpus PAYLAŞILAN BELLEKTE (multiprocessing.shared_memory);
                              görevlere sadece sorgu parçası gider. Korpus hiç kopyalanmaz.
Not (Windows): worker fonksiyonları modül seviyesinde olmalı (spawn ile
çocuk süreç bu modülü import edip fonksiyonu adıyla bulur).
"""

import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from search_kernels import chunk_ranges, search_block  # noqa: E402


# --------------------------------------------------------------------------
# Saf Python sürümleri (hocanın iskeleti): GIL'i aşmanın tek yolu süreçler
# --------------------------------------------------------------------------

def search_batch_python(query_batch, corpus_list, k=5):
    """Bir sorgu grubunu saf Python ile arar. Ayrı bir süreçte çalışır."""
    results = []
    for query in query_batch:
        scores = []
        for d_idx, doc in enumerate(corpus_list):
            dot = 0.0
            norm_q = 0.0
            norm_d = 0.0
            for i in range(len(query)):
                dot += query[i] * doc[i]
                norm_q += query[i] * query[i]
                norm_d += doc[i] * doc[i]
            sim = dot / (norm_q ** 0.5 * norm_d ** 0.5) if norm_q > 0 and norm_d > 0 else 0.0
            scores.append((d_idx, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        results.append(scores[:k])
    return results


def search_processpool_bad(queries_list, corpus_list, k=5, max_workers=4, executor=None):
    """
    KÖTÜ: her sorgu ayrı görev ve korpusun tamamı her görevde yeniden pickle'lanır.
    1.000 doküman x 384 float ~ 3 MB pickle; 20 sorgu = 60 MB serileştirme + kopya.
    """
    ex = executor or ProcessPoolExecutor(max_workers=max_workers)
    futures = [ex.submit(search_batch_python, [q], corpus_list, k) for q in queries_list]
    results = [f.result()[0] for f in futures]
    if executor is None:
        ex.shutdown()
    return results


def search_processpool_good(queries_list, corpus_list, k=5, max_workers=4, executor=None):
    """
    İYİ: sorgular max_workers parçaya bölünür -> korpus sadece max_workers kez gider.
    Görev sayısı az = pickle ve süreçler arası iletişim (IPC) yükü az.
    """
    batch_size = (len(queries_list) + max_workers - 1) // max_workers
    batches = [queries_list[i:i + batch_size] for i in range(0, len(queries_list), batch_size)]
    ex = executor or ProcessPoolExecutor(max_workers=max_workers)
    futures = [ex.submit(search_batch_python, b, corpus_list, k) for b in batches]
    results = []
    for f in futures:
        results.extend(f.result())
    if executor is None:
        ex.shutdown()
    return results


_PY = {}   # saf Python GIL deneyi için worker başına korpus (list of lists)


def init_python_corpus(corpus_list):
    """
    Worker başlarken korpusu BİR KEZ alır (initializer). Böylece zamanlanan
    görevlere sadece sorgular gider ve ölçtüğümüz şey serileştirme değil
    saf Python hesabı olur -- GIL deneyini adil kılan şey bu.
    """
    _PY["C"] = corpus_list


def python_search_preloaded(query_batch, k=5):
    """Worker içinde, önceden yüklenmiş korpusla saf Python arama."""
    return search_batch_python(query_batch, _PY["C"], k)


def search_processpool_preloaded(queries_list, executor, n_workers, k=5):
    """Sorguları n_workers parçaya bölüp korpusu önceden yüklenmiş worker'lara dağıtır."""
    bs = (len(queries_list) + n_workers - 1) // n_workers
    futures = [executor.submit(python_search_preloaded, queries_list[i:i + bs], k)
               for i in range(0, len(queries_list), bs)]
    out = []
    for f in futures:
        out.extend(f.result())
    return out


# --------------------------------------------------------------------------
# NumPy + paylaşılan bellek sürümü
# --------------------------------------------------------------------------

_WORKER = {}   # her çocuk süreçte bir kez doldurulan küresel durum (korpus görünümü)


def _init_worker(shm_name, shape, dtype_str):
    """
    Her worker süreci BAŞLARKEN bir kez çalışır (initializer).
    Paylaşılan bellek bloğuna bağlanır ve onu kopyalamadan bir NumPy dizisi
    olarak görür. BLAS thread'lerini 1'e sabitler: N süreç x 1 BLAS thread
    = N çekirdek (aşırı abonelik yok).
    """
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    shm = shared_memory.SharedMemory(name=shm_name)
    _WORKER["shm"] = shm                       # referansı tut: GC bloğu kapatmasın
    _WORKER["C"] = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf)


def _worker_search(q_chunk, k):
    """Worker içinde: korpus paylaşılan bellekten okunur, sadece sorgu parçası pickle ile gelir."""
    return search_block(q_chunk, _WORKER["C"], k)


class SharedCorpusPool:
    """
    Korpusu paylaşılan belleğe bir kez koyan ve kalıcı bir süreç havuzu tutan sınıf.

    Neden kalıcı havuz? Windows'ta süreç başlatma (spawn + numpy import)
    yüzlerce ms sürer. Her aramada havuz kurmak bu yükü her seferinde öderdi.
    Gerçek bir arama servisi de havuzu bir kez kurar. Başlatma süresini ayrıca ölçüyoruz.
    """

    def __init__(self, corpus, max_workers):
        self.shm = shared_memory.SharedMemory(create=True, size=corpus.nbytes)
        view = np.ndarray(corpus.shape, dtype=corpus.dtype, buffer=self.shm.buf)
        view[:] = corpus                         # korpusu bir kez paylaşılan belleğe kopyala
        self.max_workers = max_workers
        self.ex = ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker,
                                      initargs=(self.shm.name, corpus.shape, corpus.dtype.str))
        # Tüm worker'ları şimdi başlat (ilk aramaya başlatma süresi binmesin).
        list(self.ex.map(_noop, range(max_workers * 2)))

    def search(self, queries, k=5):
        """Sorguları worker sayısı kadar parçaya böl, paralel ara, birleştir."""
        parts = chunk_ranges(len(queries), self.max_workers)
        futures = [self.ex.submit(_worker_search, queries[a:b], k) for a, b in parts]
        res = [f.result() for f in futures]
        return np.vstack([r[0] for r in res]), np.vstack([r[1] for r in res])

    def close(self):
        """Havuzu kapat ve paylaşılan bellek bloğunu serbest bırak (sızıntı olmasın)."""
        self.ex.shutdown()
        self.shm.close()
        self.shm.unlink()


def _noop(x):
    """Havuz ısınması için boş görev."""
    return x


def search_processpool_numpy(queries, corpus, k=5, max_workers=4):
    """Tek seferlik kullanım: havuzu kur, ara, kapat (başlatma yükü DAHİL)."""
    pool = SharedCorpusPool(corpus, max_workers)
    try:
        return pool.search(queries, k)
    finally:
        pool.close()


if __name__ == "__main__":
    print("Bu strateji phase3_parallelism/run_phase3.py içinden ölçülür.")
