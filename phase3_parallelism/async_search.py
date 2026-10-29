"""
Phase 3: Async I/O Stratejisi (starter dosyası, tamamlandı)

Ne zaman doğru seçim? Çok sayıda eşzamanlı I/O (ör. 100 sorgu için harici bir
embedding API'sine istek). asyncio TEK thread'dir: beklerken (await) başka
görevlere geçer; CPU işini hızlandırmaz.

Senaryo: kullanıcı sorguları METİN olarak geliyor; her biri önce uzak bir
embedding servisine gidiyor (ağ gecikmesi asyncio.sleep ile simüle), sonra
korpusta aranıyor. Gerçekçi olsun diye servis, sorgunun önceden hesaplanmış
gerçek MiniLM vektörünü döndürüyor.

Deneyler:
  serial      : sorgular sırayla (her gecikme tek tek beklenir)
  concurrent  : asyncio.gather -> tüm istekler aynı anda havada
  bounded     : Semaphore ile en fazla M eşzamanlı istek (API rate limit'e saygı)
  to_thread   : CPU araması asyncio.to_thread ile thread'e itilir -> olay döngüsü bloklanmaz
  cpu_only    : gecikme 0 (embedding önbellekte) -> async'in hiçbir kazancı kalmamalı
"""

import asyncio
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from search_kernels import search_block  # noqa: E402


async def get_embedding_async(q_idx, Q, delay):
    """Uzak embedding API çağrısını simüle eder: ağ gecikmesi + vektör dönüşü."""
    await asyncio.sleep(delay)        # await: bu görev beklerken olay döngüsü diğerlerini çalıştırır
    return Q[q_idx:q_idx + 1]


async def search_api_async(q_idx, Q, corpus, k, delay, offload=False):
    """
    Tam asenkron hat: (1) embedding I/O, (2) CPU benzerlik araması.
    offload=False iken (2) olay döngüsünü BLOKLAR: o sırada başka hiçbir görev ilerleyemez.
    """
    q = await get_embedding_async(q_idx, Q, delay)
    if offload:
        idx, sc = await asyncio.to_thread(search_block, q, corpus, k)
    else:
        idx, sc = search_block(q, corpus, k)
    return idx[0], sc[0]


async def search_all_serial(Q, corpus, k=5, delay=0.01):
    """Sorguları tek tek işler: toplam süre ~ n_sorgu x gecikme."""
    return [await search_api_async(i, Q, corpus, k, delay) for i in range(len(Q))]


async def search_all_concurrent(Q, corpus, k=5, delay=0.01, offload=False):
    """Tüm sorgular aynı anda: toplam süre ~ 1 x gecikme + toplam CPU süresi."""
    return await asyncio.gather(*[search_api_async(i, Q, corpus, k, delay, offload)
                                  for i in range(len(Q))])


async def search_all_bounded(Q, corpus, k=5, delay=0.01, max_concurrent=10):
    """
    Sınırlı eşzamanlılık: aynı anda en fazla max_concurrent istek.
    Gerçek API'ler rate limit uygular (arXiv'den veri çekerken bizim yaptığımız gibi);
    sınırsız gather API'yi boğar ve 429 hataları alır.
    Beklenen süre ~ ceil(n / M) x gecikme.
    """
    sem = asyncio.Semaphore(max_concurrent)

    async def bounded(i):
        async with sem:
            return await search_api_async(i, Q, corpus, k, delay)

    return await asyncio.gather(*[bounded(i) for i in range(len(Q))])


def to_arrays(results):
    """[(idx, sc), ...] -> (idx dizisi, skor dizisi) (doğruluk kontrolü için)."""
    return np.vstack([r[0] for r in results]), np.vstack([r[1] for r in results])


def run(coro_fn, *args, **kwargs):
    """Senkron benchmark() fonksiyonundan çağrılabilsin diye asyncio.run sarmalayıcısı."""
    return asyncio.run(coro_fn(*args, **kwargs))


if __name__ == "__main__":
    print("Bu strateji phase3_parallelism/run_phase3.py içinden ölçülür.")
