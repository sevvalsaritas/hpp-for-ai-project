"""
Phase 3: Tüm paralellik stratejilerini AYNI iş yükünde ölçer ve Figür 5'i çizer.

İş yükleri:
  * online : STANDART İŞ YÜKÜ, 100 sorgu x tüm korpus (diğer fazlarla doğrudan karşılaştırılır)
  * batch  : 2.000 sorgu (korpustaki ilk 2.000 makale -> "her makale için benzer
             makaleleri bul" toplu işi; öneri sistemi / yakın-kopya tespiti gibi).
    Neden ikisi? 100 sorgu BLAS ile ~ms sürer; paralelliğin sabit yükleri
    (thread/süreç başlatma, IPC) bu kadar kısa işte baskın gelir. Toplu iş,
    stratejilerin gerçek ölçeklenmesini görmemizi sağlar.

Stratejiler (worker sayısı 1..16 taranır):
  1. blas     : NumPy, BLAS iç thread sayısı = w           (örtük paralellik)
  2. numba    : kaynaştırılmış prange çekirdeği, w thread   (özel döngü)
  3. thread   : ThreadPoolExecutor, w thread, BLAS=1        (GIL serbest)
  4. process  : ProcessPool + paylaşılan bellek, w süreç    (GIL yok, IPC var)
  5. dask     : dask.delayed, threaded scheduler, w thread
  6. ray      : Ray task'ları (w parça) ve actor'ler (w actor)
Ek deneyler: saf Python'da thread vs process (GIL gösterimi), pickle maliyeti
(kötü vs iyi), asyncio (I/O gecikmesi).
"""

import contextlib
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import numba  # noqa: E402

from benchmark_utils import benchmark, save_rows_csv, verify_results_tie_aware  # noqa: E402
from common import (FIGURES_DIR, RESULTS_DIR, TOP_K, configure_threads, load_embeddings,  # noqa: E402
                    load_json, normalize_rows, save_json)
import plotstyle  # noqa: E402
from plotstyle import C, METHOD_COLORS, plt  # noqa: E402
from search_kernels import (search_block, search_hybrid, search_numba_fused,  # noqa: E402
                            to_result_list)
import async_search as asy  # noqa: E402
from process_pool_search import (SharedCorpusPool, init_python_corpus,  # noqa: E402
                                 search_processpool_bad, search_processpool_good,
                                 search_processpool_preloaded)
from ray_search import (make_actors, ray_put_corpus, ray_start, ray_stop,  # noqa: E402
                        search_dask, search_ray_actors, search_ray_tasks)
from thread_pool_search import (search_threadpool, search_threadpool_python)  # noqa: E402

WORKERS = [1, 2, 4, 6, 8, 12, 16]
AMDAHL_MAX_W = 8       # Amdahl uydurmasında kullanılan en büyük worker sayısı (gerekçe plot_parallel'de)
N_BATCH = 2000
WARMUP, RUNS = 2, 5
GIL_SETUP = {"n_queries": 64, "n_docs": 2000}   # kavram gösterimi; gerekçe run_gil_demo'da


def check(name, out, ref):
    """Strateji sonucunu referans (tek çağrı BLAS) ile karşılaştırır; tolerans 1e-5."""
    ok, diff, mism = verify_results_tie_aware(to_result_list(*out), to_result_list(*ref), tol=1e-5)
    if not ok:
        print(f"  !! {name}: DOĞRULUK HATASI (max fark {diff:.2e})")
    return ok, diff


def blas_threads(n):
    """BLAS thread sayısını ayarlayan bağlam yöneticisi (ölçüm DIŞINDA girilir)."""
    return threadpool_limits(n)


def sweep(name, fn_for_w, Qw, ref, workers=WORKERS, ctx=None):
    """
    Bir stratejiyi her worker sayısı için ölçer.
    fn_for_w(w) -> argümansız çağrılabilir fonksiyon (o w ayarıyla arama yapan).
    ctx(w)     -> ölçümün TAMAMINI saran ayar (ör. BLAS thread limiti).

    Ölçüm hatası dersi: threadpool_limits() her girişte yüklü tüm DLL'leri
    tarar (~10-20 ms). İlk sürümümüzde bunu zamanlanan fonksiyonun içinde
    çağırıyorduk ve 5 ms'lik aramayı 22 ms ölçüyorduk. Ayar artık ölçüm
    dışında, bir kez yapılıyor; gerçek bir servis de ayarı başlangıçta bir kez yapar.
    """
    rows = []
    for w in workers:
        f = fn_for_w(w)
        with (ctx(w) if ctx else contextlib.nullcontext()):
            b = benchmark(f, warmup=WARMUP, runs=RUNS, auto_runs=True)
        ok, diff = check(f"{name} w={w}", b["results"], ref)
        rows.append({"strategy": name, "workers": w, "n_queries": len(Qw), "time_s": b["median"],
                     "raw_times": b["raw_times"], "correct": ok, "max_diff": diff})
        print(f"  {name:13s} w={w:2d}: {b['median'] * 1e3:9.2f} ms  doğru={ok}")
    return rows


def run_numpy_strategies(Qw, Cn, ref):
    """BLAS, Numba, ThreadPool, Dask: hepsi aynı süreçte, farklı thread modelleri."""
    rows = []
    # BLAS iç thread sayısı = w (tek çağrı, örtük paralellik)
    rows += sweep("blas", lambda w: (lambda: search_block(Qw, Cn, TOP_K)), Qw, ref, ctx=blas_threads)

    def numba_threads(w):
        numba.set_num_threads(w)                # Numba prange thread sayısı = w (ucuz çağrı)
        return contextlib.nullcontext()
    rows += sweep("numba", lambda w: (lambda: search_numba_fused(Qw, Cn, TOP_K)), Qw, ref,
                  ctx=numba_threads)

    # Hibrit: BLAS (w thread) + Numba top-k (w thread). İkisi sırayla çalışır, çakışmaz.
    @contextlib.contextmanager
    def both(w):
        numba.set_num_threads(w)
        with threadpool_limits(w):
            yield
    rows += sweep("hybrid", lambda w: (lambda: search_hybrid(Qw, Cn, TOP_K)), Qw, ref, ctx=both)

    # ThreadPool / Dask: w Python thread'i x 1 BLAS thread (aşırı abonelik yok)
    rows += sweep("thread", lambda w: (lambda: search_threadpool(Qw, Cn, TOP_K, w)), Qw, ref,
                  ctx=lambda w: blas_threads(1))
    rows += sweep("dask", lambda w: (lambda: search_dask(Qw, Cn, TOP_K, w)), Qw, ref,
                  ctx=lambda w: blas_threads(1))
    numba.set_num_threads(configure_threads())
    return rows


def run_process_strategy(Qw, Cn, ref):
    """ProcessPool + paylaşılan bellek. Havuz her w için bir kez kurulur; kurulum süresi ayrı."""
    rows = []
    for w in WORKERS:
        t0 = time.perf_counter()
        pool = SharedCorpusPool(Cn, w)
        startup = time.perf_counter() - t0
        try:
            r = sweep("process", lambda _w: (lambda: pool.search(Qw, TOP_K)), Qw, ref, [w])
            r[0]["startup_s"] = startup
            rows += r
        finally:
            pool.close()
        print(f"     (havuz kurulumu {startup:.2f} s)")
    return rows


def run_ray_strategy(Qw, Cn, ref):
    """Ray task'ları ve actor'leri. Ray bir kez başlatılır."""
    rows = []
    try:
        init_s = ray_start(num_cpus=max(WORKERS))
    except Exception as e:                      # Ray kurulamazsa/başlamazsa raporda belirtilir
        print(f"Ray başlatılamadı: {e}")
        return rows, {"error": str(e)}
    info = {"init_s": init_s}
    try:
        t0 = time.perf_counter()
        cref = ray_put_corpus(Cn)
        info["put_s"] = time.perf_counter() - t0
        rows += sweep("ray_tasks", lambda w: (lambda: search_ray_tasks(Qw, cref, TOP_K, w)), Qw, ref)
        for w in WORKERS:
            actors = make_actors(cref, w)
            rows += sweep("ray_actors", lambda _w: (lambda: search_ray_actors(Qw, actors, TOP_K)),
                          Qw, ref, [w])
            del actors
    finally:
        ray_stop()
    return rows, info


def run_gil_demo(Qn, Cn, n_q=GIL_SETUP["n_queries"], n_d=GIL_SETUP["n_docs"]):
    """
    Saf Python işinde thread vs process: GIL'in etkisini doğrudan gösterir.
    Aynı saf Python arama; thread'lerde hızlanma yok, süreçlerde var.
    Süreç tarafında korpus worker başlangıcında bir kez yüklenir (initializer);
    ilk denememizde korpusu her çağrıda pickle'lıyorduk ve 0.3 s'lik işte
    serileştirme maliyeti hesabı gölgeliyordu. İş yükü de ~1-2 s'ye büyütüldü.

    Neden standart iş yükü (100 x 19.671) DEĞİL? Bu bir performans ölçümü değil,
    kavram gösterimi: soru "thread'ler saf Python'u hızlandırır mı?" ve cevap
    iş büyüklüğünden bağımsız. Standart iş yükünde tek ölçüm ~45 s; thread'ler
    hızlanmadığı için 4 worker ayarı x 6 ölçüm sadece thread tarafında ~20 dk sürerdi.
    """
    q_list = Qn[:n_q].tolist()
    c_list = Cn[:n_d].tolist()
    rows = []
    for w in [1, 2, 4, 8]:
        b = benchmark(search_threadpool_python, q_list, c_list, TOP_K, w, warmup=1, runs=RUNS)
        rows.append({"strategy": "python_thread", "workers": w, "time_s": b["median"],
                     "raw_times": b["raw_times"]})
        with ProcessPoolExecutor(max_workers=w, initializer=init_python_corpus,
                                 initargs=(c_list,)) as ex:
            list(ex.map(abs, range(w * 2)))     # süreçleri önceden başlat (ısınma)
            b = benchmark(search_processpool_preloaded, q_list, ex, w, TOP_K,
                          warmup=1, runs=RUNS)
        rows.append({"strategy": "python_process", "workers": w, "time_s": b["median"],
                     "raw_times": b["raw_times"]})
        print(f"  saf Python w={w}: thread={rows[-2]['time_s']:.3f}s  process={rows[-1]['time_s']:.3f}s")
    return rows


def run_pickle_demo(Qn, Cn, n_q=16, n_d=2000, w=4):
    """
    Starter'daki kötü (korpus her sorguda pickle) vs iyi (parça başına) yaklaşım.
    Küçük iş yükü bilinçli: kötü yaklaşım her sorguda korpusu serileştirir; tüm
    korpusla (19.671 x 384 Python float'ı, ~190 MB pickle) 100 sorgu dakikalar sürer.
    Gösterilen şey oran (kötü / iyi), mutlak süre değil.
    """
    q_list = Qn[:n_q].tolist()
    c_list = Cn[:n_d].tolist()
    out = {}
    with ProcessPoolExecutor(max_workers=w) as ex:
        list(ex.map(abs, range(w * 2)))
        for name, fn in [("bad_pickle_per_query", search_processpool_bad),
                         ("good_batched", search_processpool_good)]:
            b = benchmark(fn, q_list, c_list, TOP_K, w, ex, warmup=1, runs=RUNS)
            out[name] = {"time_s": b["median"], "raw_times": b["raw_times"]}
            print(f"  pickle demo {name}: {b['median']:.3f} s")
    import pickle
    out["corpus_pickle_mb"] = len(pickle.dumps(c_list)) / 1024**2
    out["setup"] = {"n_queries": n_q, "n_docs": n_d, "workers": w}
    return out


def run_async_demo(Qn, Cn):
    """asyncio: farklı ağ gecikmelerinde serial / concurrent / bounded / offload."""
    rows = []
    Q = Qn[:100]
    for delay in [0.0, 0.01, 0.05]:
        for name, fn, kw in [("serial", asy.search_all_serial, {}),
                             ("concurrent", asy.search_all_concurrent, {}),
                             ("bounded_10", asy.search_all_bounded, {"max_concurrent": 10}),
                             ("concurrent_to_thread", asy.search_all_concurrent, {"offload": True})]:
            b = benchmark(asy.run, fn, Q, Cn, TOP_K, delay, warmup=1, runs=RUNS, **kw)
            rows.append({"delay_s": delay, "mode": name, "time_s": b["median"],
                         "raw_times": b["raw_times"]})
            print(f"  async gecikme={delay:.2f}s {name:22s}: {b['median'] * 1e3:8.1f} ms")
    return rows


def amdahl_fit(workers, speedups):
    """
    Ölçülen hızlanmalara Amdahl yasasını en küçük kareler ile uydurur:
        S(n) = 1 / ((1 - P) + P / n)   ->   1/S = 1 - P * (1 - 1/n)
    1/S, (1 - 1/n)'ye göre doğrusaldır; eğimden paralel oran P bulunur.
    """
    x = 1 - 1 / np.asarray(workers, float)
    y = 1 - 1 / np.asarray(speedups, float)        # = P * x
    P = float(np.clip(np.dot(x, y) / np.dot(x, x), 0, 1))
    return P


def plot_parallel(rows, gil_rows, save_path):
    """Figür 5: (a) NumPy tabanlı stratejiler hızlanma vs worker, (b) saf Python GIL gösterimi."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1.45, 1]})
    batch = [r for r in rows if r["workload"] == "batch"]
    t1 = next(r["time_s"] for r in batch if r["strategy"] == "blas" and r["workers"] == 1)

    ax = axes[0]
    styles = [("hybrid", "Hibrit: BLAS + Numba top-k", C["red"]),
              ("blas", "NumPy/BLAS thread", METHOD_COLORS["blas"]),
              ("numba", "Numba prange (fused)", METHOD_COLORS["numba_parallel"]),
              ("thread", "ThreadPool (BLAS=1)", METHOD_COLORS["thread"]),
              ("process", "ProcessPool + shm", METHOD_COLORS["process"]),
              ("dask", "Dask (threads)", METHOD_COLORS["dask"]),
              ("ray_tasks", "Ray tasks", METHOD_COLORS["ray"])]
    ws = np.array(WORKERS)
    ax.plot(ws, ws, color=plotstyle.MUTED, linestyle=":", linewidth=1.2, label="ideal (doğrusal)")
    fits = {}
    for key, label, col in styles:
        pts = sorted((r["workers"], t1 / r["time_s"]) for r in batch if r["strategy"] == key)
        if not pts:
            continue
        w_, s_ = zip(*pts)
        # Amdahl uydurması stratejinin KENDİ 1-worker süresine göre yapılır:
        # own1 / t_w = own1 * s / t1  (s = t1 / t_w)
        # Sadece w <= 8 kullanılır: 12-16 worker'da yavaş E/LP-E çekirdekler ve
        # (hibritte) BLAS + Numba thread havuzlarının çakışması Amdahl'ın
        # varsaymadığı "eklenen worker yavaşlatır" etkisi yaratıyor.
        own1 = next(r["time_s"] for r in batch if r["strategy"] == key and r["workers"] == 1)
        fit_pts = [(w, own1 * s / t1) for w, s in zip(w_, s_) if w <= AMDAHL_MAX_W]
        fits[key] = amdahl_fit(*zip(*fit_pts))
        ax.plot(w_, s_, "o-", color=col, label=f"{label}  (P≈{fits[key]:.2f})")
    # En iyi stratejinin Amdahl eğrisi
    best = max(fits, key=lambda k: max(t1 / r["time_s"] for r in batch if r["strategy"] == k))
    own1 = next(r["time_s"] for r in batch if r["strategy"] == best and r["workers"] == 1)
    wr = np.linspace(1, 16, 100)
    ax.plot(wr, (t1 / own1) / ((1 - fits[best]) + fits[best] / wr), "--", color=plotstyle.INK2,
            linewidth=1.1, label=f"Amdahl uydurma ({best}, w≤{AMDAHL_MAX_W}, P={fits[best]:.2f})")
    ax.axvspan(14, 16.5, color=plotstyle.GRID, alpha=0.5, linewidth=0)
    ax.text(15.25, 15.0, "LP-E\nçekirdekler", ha="center", fontsize=7.5, color=plotstyle.MUTED)
    ax.set_xlabel("Worker / thread sayısı")
    ax.set_ylabel("Hızlanma (1-thread BLAS'a göre)")
    ax.set_title(f"(a) {N_BATCH} sorgu toplu arama: hızlanma vs worker")
    ax.set_xticks(WORKERS)
    ax.set_xlim(0.5, 16.5)
    ax.legend(fontsize=7.8, loc="upper left")

    ax = axes[1]
    for key, label, col in [("python_thread", "Saf Python + ThreadPool", METHOD_COLORS["thread"]),
                            ("python_process", "Saf Python + ProcessPool", METHOD_COLORS["process"])]:
        pts = sorted((r["workers"], r["time_s"]) for r in gil_rows if r["strategy"] == key)
        w_, t_ = zip(*pts)
        ax.plot(w_, [t_[0] / t for t in t_], "o-", color=col, label=label)
    ax.plot([1, 8], [1, 8], color=plotstyle.MUTED, linestyle=":", linewidth=1.2, label="ideal")
    ax.set_xlabel("Worker sayısı")
    ax.set_ylabel("Hızlanma (1 worker'a göre)")
    ax.set_title("(b) GIL: saf Python işte thread vs süreç")
    ax.set_xticks([1, 2, 4, 8])
    ax.legend(fontsize=8.5)

    plt.tight_layout()
    plotstyle.save(fig, save_path)
    return fits


def main():
    configure_threads()
    C_all, Q = load_embeddings()
    Cn = np.ascontiguousarray(normalize_rows(C_all).astype(np.float32))
    Qn = np.ascontiguousarray(normalize_rows(Q).astype(np.float32))
    Qb = np.ascontiguousarray(Cn[:N_BATCH])
    print(f"Korpus {Cn.shape}, online sorgu {Qn.shape[0]}, toplu sorgu {Qb.shape[0]}, "
          f"çekirdek {os.cpu_count()}")

    # Referans sonuçlar (doğruluk için): tek çağrı BLAS
    ref_b = search_block(Qb, Cn, TOP_K)
    ref_o = search_block(Qn, Cn, TOP_K)

    rows = []
    for wl, Qw, ref in [("batch", Qb, ref_b), ("online", Qn, ref_o)]:
        print(f"\n===== İş yükü: {wl} ({len(Qw)} sorgu) =====")
        r = run_numpy_strategies(Qw, Cn, ref)
        r += run_process_strategy(Qw, Cn, ref)
        rr, ray_info = run_ray_strategy(Qw, Cn, ref)
        r += rr
        for x in r:
            x["workload"] = wl
        rows += r

    print("\n===== GIL gösterimi (saf Python) =====")
    gil_rows = run_gil_demo(Qn, Cn)
    print("\n===== Pickle maliyeti =====")
    pickle_demo = run_pickle_demo(Qn, Cn)
    print("\n===== asyncio =====")
    async_rows = run_async_demo(Qn, Cn)

    baseline_est = baseline_estimate(Cn.shape[0], Qn.shape[0])
    fits = plot_parallel(rows, gil_rows, FIGURES_DIR / "fig5_parallel_scaling.png")
    save_rows_csv(rows, RESULTS_DIR / "phase3_parallel.csv")
    save_rows_csv(gil_rows, RESULTS_DIR / "phase3_gil_demo.csv")
    save_rows_csv(async_rows, RESULTS_DIR / "phase3_async.csv")
    save_json({"n_docs": Cn.shape[0], "n_batch": N_BATCH, "workers": WORKERS,
               "rows": rows, "gil_demo": gil_rows, "gil_setup": GIL_SETUP, "pickle_demo": pickle_demo,
               "async": async_rows, "amdahl_P": fits, "ray": ray_info,
               "naive_baseline_extrapolated_s": baseline_est}, "phase3_parallel.json")


def baseline_estimate(n_docs, n_online):
    """
    Phase 1 baseline'ından saf Python süresi. Online iş yükü standart iş yüküyle
    aynı olduğundan bu doğrudan ölçülen değerdir; toplu iş (2.000 sorgu) için
    sorgu x doküman başına maliyetle doğrusal ekstrapolasyon yapılır.
    """
    try:
        p1 = load_json("phase1_baseline.json")
    except FileNotFoundError:
        return {}
    per_pair = p1["naive"]["median"] / (p1["n_queries"] * p1["n_docs"])
    return {"batch": per_pair * N_BATCH * n_docs, "online": per_pair * n_online * n_docs}


def replot():
    """Kaydedilmiş sonuçlardan Figür 5'i yeniden çizer, Amdahl P'lerini ve baseline tahminini günceller (ölçüm yapmaz)."""
    d = load_json("phase3_parallel.json")
    n_online = next(r["n_queries"] for r in d["rows"] if r["workload"] == "online")
    d["naive_baseline_extrapolated_s"] = baseline_estimate(d["n_docs"], n_online)
    d["amdahl_P"] = plot_parallel(d["rows"], d["gil_demo"], FIGURES_DIR / "fig5_parallel_scaling.png")
    d["amdahl_max_workers"] = AMDAHL_MAX_W
    d["gil_setup"] = GIL_SETUP
    save_json(d, "phase3_parallel.json")


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        replot()
    else:
        main()
