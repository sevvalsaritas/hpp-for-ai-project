"""
Phase 1: Profiling (starter dosyası, tamamlandı)

"Bir şeyi düzeltmeden önce neyin bozuk olduğunu bilmelisin."
  1. cProfile    -> hangi FONKSİYON ne kadar zaman alıyor?
  2. line_profiler -> fonksiyonun içinde hangi SATIR ne kadar zaman alıyor?
  3. Figür 1     -> ikisini tek bir görselde birleştirir (figures/fig1_profiling.png)

Çıktılar: results/profile_cprofile.txt, results/profile_line.txt, results/profile_summary.json
Not: Profiler'ların kendisi yavaşlatır (cProfile her fonksiyon çağrısını,
line_profiler her satırı izler). Yine de profili baseline ile AYNI standart iş
yükünde (100 sorgu x tüm korpus) alıyoruz; böylece profildeki oranlar doğrudan
Phase 1 baseline süresine uygulanabilir (cProfile ~2 dk, line_profiler ~10 dk).
"""

import cProfile
import io
import pstats
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # ekransız (dosyaya) çizim -- Windows'ta pencere açmasın
import matplotlib.pyplot as plt
from line_profiler import LineProfiler

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import FIGURES_DIR, RESULTS_DIR, TOP_K, save_json  # noqa: E402
from naive_search import cosine_similarity_python, load_python_lists, search_naive  # noqa: E402


def profile_with_cprofile(queries, corpus, k=TOP_K):
    """
    cProfile ile fonksiyon düzeyinde profil.
    tottime = fonksiyonun KENDİ içinde geçen süre; cumtime = çağırdıklarıyla birlikte.
    Darboğazı bulmak için tottime'a bakarız.
    """
    pr = cProfile.Profile()
    pr.enable()
    search_naive(queries, corpus, k)
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
    ps.print_stats(15)
    text = s.getvalue()
    (RESULTS_DIR / "profile_cprofile.txt").write_text(text, encoding="utf-8")
    print(text)

    # Grafik ve JSON için (fonksiyon adı, çağrı sayısı, tottime) çıkar.
    rows = []
    total = ps.total_tt
    for (fname, line, func), (cc, nc, tt, ct, _) in ps.stats.items():
        label = func if func.startswith("<") else f"{func}()"
        rows.append({"function": label, "ncalls": nc, "tottime": tt, "cumtime": ct})
    rows.sort(key=lambda r: r["tottime"], reverse=True)
    return rows, total


def profile_with_line_profiler(queries, corpus, k=TOP_K):
    """
    line_profiler ile satır düzeyinde profil (cosine fonksiyonu + arama döngüsü).
    Programatik kullanım: kernprof gerekmez, LineProfiler nesnesine fonksiyonları ekliyoruz.
    """
    lp = LineProfiler()
    lp.add_function(cosine_similarity_python)
    lp.add_function(search_naive)
    lp.runcall(search_naive, queries, corpus, k)

    s = io.StringIO()
    lp.print_stats(stream=s, output_unit=1e-3)   # milisaniye cinsinden
    text = s.getvalue()
    (RESULTS_DIR / "profile_line.txt").write_text(text, encoding="utf-8")
    print(text)

    # cosine fonksiyonunun satır bazında süreleri (Figür 1 sağ panel için)
    stats = lp.get_stats()
    line_rows = {}
    for (fname, start, func), timings in stats.timings.items():
        src = Path(fname).read_text(encoding="utf-8").splitlines()
        # Satır yorumlarını (# ...) etiketten atıyoruz ki grafik okunaklı olsun.
        rows = [{"line": ln, "hits": hits, "time_s": t * stats.unit,
                 "code": src[ln - 1].split("#")[0].strip()} for ln, hits, t in timings]
        line_rows[func] = rows
    return line_rows


def plot_profile(func_rows, total, line_rows, save_path):
    """Figür 1: sol = cProfile tottime payları, sağ = cosine içindeki satır payları."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1, 1.15]})

    top = func_rows[:5]
    labels = [r["function"].replace("<method ", "<").replace(" of 'list' objects>", ">")[:42] for r in top]
    pct = [100 * r["tottime"] / total for r in top]
    ax = axes[0]
    bars = ax.barh(labels[::-1], pct[::-1], color="#3b6ea5")
    for b, r in zip(bars, top[::-1]):
        ax.text(b.get_width() + 1, b.get_y() + b.get_height() / 2,
                f"{b.get_width():.1f}%  ({r['ncalls']:,} çağrı)", va="center", fontsize=9)
    ax.set_xlim(0, 115)
    ax.set_xlabel("Toplam sürenin yüzdesi (tottime)")
    ax.set_title("cProfile: fonksiyon bazında süre", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    rows = line_rows.get("cosine_similarity_python", [])
    tot_line = sum(r["time_s"] for r in rows) or 1
    ax = axes[1]
    labels = [f"L{r['line']}: {r['code'][:38]}" for r in rows]
    pct = [100 * r["time_s"] / tot_line for r in rows]
    bars = ax.barh(labels[::-1], pct[::-1], color="#c0504d")
    for b in bars:
        ax.text(b.get_width() + 0.5, b.get_y() + b.get_height() / 2,
                f"{b.get_width():.1f}%", va="center", fontsize=9)
    ax.set_xlim(0, max(pct) * 1.25 if pct else 1)
    ax.set_xlabel("cosine_similarity_python içindeki sürenin yüzdesi")
    ax.set_title("line_profiler: satır bazında süre", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.tick_params(axis="y", labelsize=8.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Kaydedildi: {save_path}")


def profile_with_pyspy():
    """py-spy ile alev grafiği (flame graph) almak için terminal komutu (opsiyonel)."""
    print("Flame graph için (opsiyonel):")
    print("  py-spy record -o figures/flamegraph.svg -- py -3.14 phase1_baseline/naive_search.py --runs 1")


if __name__ == "__main__":
    queries, corpus = load_python_lists()          # standart iş yükü
    print(f"Profil iş yükü: {len(queries)} sorgu x {len(corpus)} doküman")

    func_rows, total = profile_with_cprofile(queries, corpus)
    line_rows = profile_with_line_profiler(queries, corpus)

    plot_profile(func_rows, total, line_rows, FIGURES_DIR / "fig1_profiling.png")
    save_json({"n_queries": len(queries), "n_docs": len(corpus),
               "cprofile_top": func_rows[:10], "cprofile_total_s": total,
               "line_profile": line_rows}, "profile_summary.json")
    profile_with_pyspy()
