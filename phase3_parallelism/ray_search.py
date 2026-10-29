"""
Phase 3: Ray (Tasks / Actors) ve Dask Stratejileri (starter dosyası, tamamlandı)

Ray: birçok bağımsız görevi dağıtmak veya durum tutan worker'lar (Actor) için.
     ray.put() korpusu paylaşılan nesne deposuna (object store) BİR KEZ koyar;
     görevler onu kopyalamadan (zero-copy, salt okunur) okur.
Dask: Bellekten büyük veri / out-of-core hesap için. Burada dask.delayed ile
     sorgu parçalarını threaded scheduler'da çalıştırıyoruz.

Starter'dan farkımız: starter her aramada ray.init()/ray.shutdown() çağırıyordu.
Ray başlatması Windows'ta saniyeler sürer; bu yükü her ölçüme katmak haksız
olur. Ray bir kez başlatılır (başlatma süresi ayrıca raporlanır).
Ayrıca görev sayısı = worker sayısı (parça başına görev), sorgu başına değil.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from search_kernels import chunk_ranges, search_block  # noqa: E402

_RAY = {}   # ray.remote fonksiyon/sınıf tanımları (ray import'u isteğe bağlı olsun diye tembel)


def ray_start(num_cpus):
    """Ray'i bir kez başlatır; başlatma süresini döndürür."""
    import ray
    t0 = time.perf_counter()
    ray.init(num_cpus=num_cpus, include_dashboard=False, log_to_driver=False,
             ignore_reinit_error=True)

    @ray.remote(num_cpus=1)
    def search_task(q_chunk, corpus, k):
        # corpus: ray.put ile konulmuş nesne referansı argüman olarak geçince Ray
        # onu otomatik çözer; NumPy dizileri nesne deposundan sıfır kopya okunur.
        # Ray, num_cpus=1 görevlerde OMP/BLAS thread sayısını 1'e ayarlar.
        return search_block(q_chunk, corpus, k)

    @ray.remote(num_cpus=1)
    class SearchActor:
        """Durumlu worker: korpusu bir kez alır, sonra sadece sorgu bekler (model servisi gibi)."""

        def __init__(self, corpus):
            self.corpus = corpus
            self.query_count = 0

        def search(self, q_chunk, k):
            self.query_count += len(q_chunk)
            return search_block(q_chunk, self.corpus, k)

        def get_stats(self):
            return {"queries_served": self.query_count}

    _RAY.update(task=search_task, actor_cls=SearchActor)
    return time.perf_counter() - t0


def ray_stop():
    """Ray'i kapatır."""
    import ray
    ray.shutdown()


def ray_put_corpus(corpus):
    """Korpusu nesne deposuna bir kez koyar (her görevde yeniden göndermemek için)."""
    import ray
    return ray.put(corpus)


def search_ray_tasks(queries, corpus_ref, k=5, num_workers=4):
    """Stateless görevler: sorgular num_workers parçaya bölünür, her parça bir görev."""
    import ray
    futures = [_RAY["task"].remote(queries[a:b], corpus_ref, k)
               for a, b in chunk_ranges(len(queries), num_workers)]
    res = ray.get(futures)
    return np.vstack([r[0] for r in res]), np.vstack([r[1] for r in res])


def make_actors(corpus_ref, num_actors):
    """num_actors adet durumlu arama worker'ı oluşturur (her biri korpus referansını tutar)."""
    import ray
    actors = [_RAY["actor_cls"].remote(corpus_ref) for _ in range(num_actors)]
    ray.get([a.get_stats.remote() for a in actors])      # hepsinin hazır olmasını bekle
    return actors


def search_ray_actors(queries, actors, k=5):
    """Sorgu parçaları actor'lere sırayla (round-robin) dağıtılır."""
    import ray
    parts = chunk_ranges(len(queries), len(actors))
    futures = [actors[i % len(actors)].search.remote(queries[a:b], k)
               for i, (a, b) in enumerate(parts)]
    res = ray.get(futures)
    return np.vstack([r[0] for r in res]), np.vstack([r[1] for r in res])


def search_dask(queries, corpus, k=5, num_workers=4):
    """
    Dask: her sorgu parçası bir dask.delayed görevi; threaded scheduler çalıştırır.
    BLAS GIL'i bıraktığı için thread'ler paralel ilerler (ThreadPool ile aynı mantık,
    üstüne görev grafiği yükü). Korpus bir kez delayed nesne yapılır -> grafikte tek düğüm.
    Dask'ın asıl gücü (out-of-core, chunk'lı diziler) korpus RAM'e sığmayınca ortaya çıkar.
    """
    import dask
    corpus_d = dask.delayed(corpus, pure=True)
    tasks = [dask.delayed(search_block)(queries[a:b], corpus_d, k)
             for a, b in chunk_ranges(len(queries), num_workers)]
    res = dask.compute(*tasks, scheduler="threads", num_workers=num_workers)
    return np.vstack([r[0] for r in res]), np.vstack([r[1] for r in res])


def search_dask_array(queries, corpus, k=5, chunk_docs=2000):
    """
    Dask Array sürümü (out-of-core senaryosu): korpus chunk_docs'luk bloklara
    bölünür, her blok için Q @ blok.T hesaplanır. Korpus diskte (np.memmap /
    zarr) olsaydı aynı kod RAM'den büyük veride çalışırdı.
    """
    import dask.array as da
    q = da.from_array(queries, chunks=(queries.shape[0], queries.shape[1]))
    c = da.from_array(corpus, chunks=(chunk_docs, corpus.shape[1]))
    S = (q @ c.T).compute(scheduler="threads")
    k = min(k, S.shape[1])
    idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
    sc = np.take_along_axis(S, idx, axis=1)
    order = np.argsort(-sc, axis=1, kind="stable")
    return np.take_along_axis(idx, order, axis=1), np.take_along_axis(sc, order, axis=1)


if __name__ == "__main__":
    print("Bu strateji phase3_parallelism/run_phase3.py içinden ölçülür.")
