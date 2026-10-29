"""
phase3_parallelism/search_kernels.py -- Tüm paralellik stratejilerinin paylaştığı arama çekirdekleri.

Her strateji aynı işi farklı şekilde dağıtır; işin KENDİSİ burada tek yerde
tanımlı, böylece ölçtüğümüz fark sadece "dağıtma yöntemi" olur.

Arama işi (Phase 2'nin en iyi sürümü):
    S = Qn @ Cn.T      (normalize vektörler -> cosine; BLAS sgemm)
    top-k = argpartition + küçük sıralama
"""

import numpy as np
from numba import njit, prange


def search_block(Qn, Cn, k):
    """
    Bir sorgu bloğu için top-k (indeks, skor) dizilerini döndürür.

    Dizi (array) döndürüyoruz, Python tuple listesi değil: süreçler/düğümler
    arası taşımada (pickle) diziler tek bir bellek bloğu olarak çok daha ucuz gider.
    Dönüş: idx (q x k, int64), scores (q x k, float32) -- azalan skor sırası
    """
    S = Qn @ Cn.T                                          # q x N skor matrisi (BLAS)
    k = min(k, S.shape[1])
    idx = np.argpartition(-S, k - 1, axis=1)[:, :k]        # O(N): sadece k adayı ayır
    sc = np.take_along_axis(S, idx, axis=1)
    order = np.argsort(-sc, axis=1, kind="stable")         # sadece k elemanı sırala
    return np.take_along_axis(idx, order, axis=1), np.take_along_axis(sc, order, axis=1)


def to_result_list(idx, scores):
    """(idx, scores) dizilerini verify_results'ın beklediği [(i, s), ...] biçimine çevirir."""
    return [[(int(i), float(s)) for i, s in zip(ri, rs)] for ri, rs in zip(idx, scores)]


def chunk_ranges(n, n_chunks):
    """[0, n) aralığını yaklaşık eşit n_chunks parçaya böler (yük dengesi)."""
    n_chunks = max(1, min(n_chunks, n))
    bounds = np.linspace(0, n, n_chunks + 1).astype(int)
    return [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]


@njit(parallel=True, cache=True)
def topk_numba(S, k):
    """
    Skor matrisinin her satırı için top-k, satırlar prange ile paralel.

    Neden? Profil: 2.000 x 6.490 skor matrisinde np.argpartition TEK THREAD
    çalışıyor ve 66 ms sürüyor; BLAS matris çarpımı ise 8 thread'de 28 ms.
    Yani hesabın paralel kısmını hızlandırdıkça seri top-k toplam sürenin
    çoğunu oluşturdu (Amdahl yasası). Bu çekirdek seri kısmı paralelleştirir:
    her thread kendi satırlarını tek geçişte tarar, k elemanlık sıralı bir
    dizi tutar (k=5 için ekleme sıralaması O(N)). Ölçülen: 66 ms -> 6 ms.
    """
    nq, n = S.shape
    out_idx = np.empty((nq, k), dtype=np.int64)
    out_sc = np.empty((nq, k), dtype=S.dtype)
    for i in prange(nq):
        best_s = np.full(k, -np.inf, dtype=S.dtype)        # azalan sıralı en iyi k skor
        best_i = np.full(k, -1, dtype=np.int64)
        for j in range(n):
            s = S[i, j]
            if s > best_s[k - 1]:                           # sadece en kötüden iyiyse işlem yap
                p = k - 1
                while p > 0 and best_s[p - 1] < s:          # yerini bul (eşitlikte küçük indeks önde)
                    best_s[p] = best_s[p - 1]
                    best_i[p] = best_i[p - 1]
                    p -= 1
                best_s[p] = s
                best_i[p] = j
        out_idx[i] = best_i
        out_sc[i] = best_s
    return out_idx, out_sc


QUERY_BLOCK = 512   # bir blokta işlenen sorgu sayısı (ölçüm: 32..1024 arasında en iyisi)


def search_hybrid(Qn, Cn, k, block=QUERY_BLOCK):
    """
    Hibrit (final sistem): matris çarpımı BLAS'ta (en iyi GEMM), top-k Numba'da (paralel).
    Her işi en iyi yapan araca veriyoruz; ikisi de GIL'i bırakır.

    Sorgu bloklama (uygulama düzeyinde cache blocking):
      2.000 sorgu x 19.671 doküman skor matrisi 157 MB. Her çağrıda bu kadar
      YENİ bellek ayırmak Windows'ta sayfa hatalarına (ilk dokunuşta sıfırlanan
      sayfalar; ölçtüğümüz ~1 ms/MB) ve RAM'e gidip gelen trafiğe yol açıyordu.
      Sorguları `block`'luk parçalar halinde işleyip TEK bir tamponu yeniden
      kullanıyoruz (512 x 19.671 x 4 B = 40 MB, bir kez ayrılır).
      Ön ölçüm (ayrı bir script, 2.000 sorgu, 19.671 doküman, 8 thread): bloksuz 92 ms -> blok 512
      ile 74 ms. Blok 32-128'de BLAS'a verilen matris küçüldüğü için verim
      düşüyor (101-140 ms). Kalan sürenin çoğu matris çarpımının kendisi
      (~30 GFLOP): artık hesap sınırlıyız (roofline, Ek B).
    """
    nq = Qn.shape[0]
    k = min(k, Cn.shape[0])
    if nq <= block:                                        # küçük iş: tek çağrı
        return topk_numba(Qn @ Cn.T, k)
    out_idx = np.empty((nq, k), dtype=np.int64)
    out_sc = np.empty((nq, k), dtype=np.float32)
    buf = np.empty((block, Cn.shape[0]), dtype=np.float32)  # bir kez ayrılan, tekrar kullanılan tampon
    CT = Cn.T                                              # kopya değil, görünüm (BLAS transpoz bayrağı)
    for a in range(0, nq, block):
        b = min(a + block, nq)
        S = buf[:b - a]                                    # ilk satırlar: ardışık (C-contiguous)
        np.matmul(Qn[a:b], CT, out=S)                      # sonuç mevcut tampona yazılır, yeni ayırma yok
        out_idx[a:b], out_sc[a:b] = topk_numba(S, k)
    return out_idx, out_sc


@njit(parallel=True, fastmath=True, cache=True)
def search_numba_fused(Qn, Cn, k):
    """
    Numba prange stratejisi: benzerlik + top-k TEK geçişte ("kaynaştırılmış", fused).

    NumPy sürümü önce q x N skor matrisinin tamamını belleğe yazar, sonra
    argpartition onu tekrar okur -> 2 kez bellek trafiği. Burada her thread bir
    sorgu için skorları hesaplarken k elemanlık küçük bir diziyle en iyileri
    tutar; büyük ara matris HİÇ oluşmaz. Bellek bant genişliği tasarrufu.

    Paralellik: prange sorgular üzerinde; her thread kendi out satırına yazar.
    fastmath: iç toplamın SIMD ile vektörleşmesine izin verir.
    """
    nq, d = Qn.shape
    n = Cn.shape[0]
    out_idx = np.empty((nq, k), dtype=np.int64)
    out_sc = np.empty((nq, k), dtype=np.float32)
    for i in prange(nq):
        best_s = np.full(k, -np.inf, dtype=np.float32)    # azalan sıralı en iyi k skor
        best_i = np.full(k, -1, dtype=np.int64)
        for j in range(n):
            s = np.float32(0.0)
            for t in range(d):
                s += Qn[i, t] * Cn[j, t]
            if s > best_s[k - 1]:                          # en kötü adaydan iyiyse ekle
                p = k - 1
                while p > 0 and best_s[p - 1] < s:         # eklemeli sıralama (k küçük)
                    best_s[p] = best_s[p - 1]
                    best_i[p] = best_i[p - 1]
                    p -= 1
                best_s[p] = s
                best_i[p] = j
        out_idx[i] = best_i
        out_sc[i] = best_s
    return out_idx, out_sc
