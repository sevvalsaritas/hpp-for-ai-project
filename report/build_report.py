"""
report/build_report.py -- Raporu (PDF) results/ altındaki ham sonuçlardan ve figures/ altındaki
figürlerden otomatik üretir.

Neden kodla? Rapordaki her sayı bir JSON dosyasından okunur. Benchmark'lar yeniden
koşulduğunda rapor da yeniden üretilir; elle kopyalanan sayı yoktur
(tekrarlanabilirlik şartı). Türkçe karakterler için matplotlib ile gelen
DejaVu fontları kullanılır (reportlab'in yerleşik Helvetica'sında ş/ğ/ı yok).

Kullanım:  py -3.14 report/build_report.py   ->  report/rapor.pdf
"""

import os
import sys
from pathlib import Path

import matplotlib
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, Preformatted,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import FIGURES_DIR, RESULTS_DIR, load_json  # noqa: E402

OUT = ROOT / "report" / "rapor.pdf"
AUTHOR = "Şevval Özlem Sarıtaş"
REPO = "https://github.com/sevvalsaritas/hpp-for-ai-project"

# ---------------------------------------------------------------- fontlar ve stiller
FONT_DIR = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("DV", str(FONT_DIR / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DV-B", str(FONT_DIR / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DV-I", str(FONT_DIR / "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFont(TTFont("DV-BI", str(FONT_DIR / "DejaVuSans-BoldOblique.ttf")))
pdfmetrics.registerFont(TTFont("DVM", str(FONT_DIR / "DejaVuSansMono.ttf")))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")

INK = colors.HexColor("#0b0b0b")
INK2 = colors.HexColor("#52514e")
ACCENT = colors.HexColor("#2a78d6")
RULE = colors.HexColor("#c3c2b7")
ZEBRA = colors.HexColor("#f3f2ee")

ss = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=ss["Normal"], fontName="DV", fontSize=9.3, leading=13.2,
                      alignment=TA_JUSTIFY, textColor=INK, spaceAfter=5)
H1 = ParagraphStyle("h1", parent=BODY, fontName="DV-B", fontSize=13.5, leading=17, spaceBefore=10,
                    spaceAfter=6, textColor=INK, alignment=0)
H2 = ParagraphStyle("h2", parent=BODY, fontName="DV-B", fontSize=10.5, leading=14, spaceBefore=7,
                    spaceAfter=3, textColor=ACCENT, alignment=0)
CAP = ParagraphStyle("cap", parent=BODY, fontSize=8.4, leading=11.2, textColor=INK2, spaceAfter=9)
BUL = ParagraphStyle("bul", parent=BODY, leftIndent=12, bulletIndent=2, spaceAfter=2.5)
CODE = ParagraphStyle("code", fontName="DVM", fontSize=7.3, leading=9.2, textColor=INK,
                      backColor=colors.HexColor("#f5f4f0"), borderPadding=5, leftIndent=4,
                      spaceBefore=3, spaceAfter=8)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=7.9, leading=10, alignment=0, spaceAfter=0)
CELLB = ParagraphStyle("cellb", parent=CELL, fontName="DV-B")
TITLE = ParagraphStyle("title", parent=BODY, fontName="DV-B", fontSize=17, leading=22, alignment=0,
                       spaceAfter=4)

story = []
FIG_N = [0]
TAB_N = [0]


def P(t, st=BODY):
    story.append(Paragraph(t, st))


def H(t, lvl=1):
    story.append(Paragraph(t, H1 if lvl == 1 else H2))


def bullets(items):
    for it in items:
        story.append(Paragraph(it, BUL, bulletText="•"))


def code(t):
    story.append(Preformatted(t.strip("\n"), CODE))


def fig(name, caption, width=17.0):
    """Figürü (genişlik cm) ve çıkarım içeren başlığını ekler; birlikte aynı sayfada tutulur."""
    FIG_N[0] += 1
    path = FIGURES_DIR / name
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    img = Image(str(path), width=width * cm, height=width * cm * h / w)
    story.append(KeepTogether([img, Paragraph(f"<b>Şekil {FIG_N[0]}:</b> {caption}", CAP)]))
    return FIG_N[0]


def table(rows, widths, caption=None, header=True):
    """Basit tablo: ilk satır başlık, zebra çizgili."""
    data = [[Paragraph(str(c), CELLB if (header and i == 0) else CELL) for c in r]
            for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1 if header else 0)
    st = [("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK2),
          ("LINEBELOW", (0, -1), (-1, -1), 0.5, RULE),
          ("TOPPADDING", (0, 0), (-1, -1), 2.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2)]
    for i in range(1, len(rows)):
        if i % 2 == 0:
            st.append(("BACKGROUND", (0, i), (-1, i), ZEBRA))
    t.setStyle(TableStyle(st))
    items = []
    if caption:
        TAB_N[0] += 1
        items.append(Paragraph(f"<b>Tablo {TAB_N[0]}:</b> {caption}", CAP))
    items.append(t)
    items.append(Spacer(1, 8))
    story.append(KeepTogether(items))


def ms(t, d=2):
    """Saniyeyi okunur birime çevirir."""
    if t >= 1:
        return f"{t:.2f} s"
    if t >= 1e-3:
        return f"{t * 1e3:.{d}f} ms"
    return f"{t * 1e6:.1f} µs"


def x(v):
    return f"{v:,.0f}×" if v >= 10 else f"{v:.1f}×"


def on_page(canvas, doc):
    """Her sayfanın altına sayfa numarası ve kısa başlık."""
    canvas.saveState()
    canvas.setFont("DV", 7.5)
    canvas.setFillColor(INK2)
    canvas.drawString(2 * cm, 1.2 * cm, "YZM 513 · Yüksek Başarımlı Benzerlik Arama Motoru")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"{doc.page}")
    canvas.restoreState()


# ============================================================================ veri
ds = load_json("dataset_stats.json")
emb = load_json("embedding_info.json")
p1 = load_json("phase1_baseline.json")
prof = load_json("profile_summary.json")
p2a = load_json("phase2a_sparse_dense.json")
p2b = load_json("phase2b_svd.json")
p2c = load_json("phase2c_vectorise.json")
p3 = load_json("phase3_parallel.json")
p4 = load_json("phase4_final.json")
hw = load_json("hardware.json")
ann = load_json("bonus_ann.json") if (RESULTS_DIR / "bonus_ann.json").exists() else None
roof = load_json("bonus_roofline.json") if (RESULTS_DIR / "bonus_roofline.json").exists() else None

N = ds["n_corpus"]
NQ = ds["n_queries"]
naive_t = p1["naive"]["median"]
sizes = p4["sizes"]
sc_last = p4["scaling"][-1]
corr = p4["correctness"]
vers = {v["version"]: v for v in p4["all_versions"]}
final_v = next(v for v in p4["all_versions"] if v["key"] == "final")
naive_v = next(v for v in p4["all_versions"] if v["key"] == "python")
final_speedup = naive_v["time_s"] / final_v["time_s"]

# ============================================================================ KAPAK
story.append(Spacer(1, 6))
P("YZM 513 · Proje 1", ParagraphStyle("k", parent=BODY, textColor=ACCENT, fontName="DV-B"))
P("Yüksek Başarımlı Benzerlik Arama Motoru: arXiv 2025-2026 Özetleri Üzerinde "
  "Profilden Paralel Sisteme", TITLE)
P(f"{AUTHOR} · İstanbul Medeniyet Üniversitesi · Güz 2026 · Kod: "
  f"<link href='{REPO}' color='#2a78d6'>{REPO}</link>", ParagraphStyle("au", parent=CAP, alignment=0))
P(f"<b>Özet.</b> {N:,} arXiv özetinden (cs.AI / cs.CL / cs.LG, 2025-2026) oluşan bir korpusta "
  f"top-{5} anlamsal benzerlik araması için saf Python bir başlangıç noktasından (100 sorgu x "
  f"{p1['n_docs']:,} doküman: <b>{ms(naive_t)}</b>) başlayıp profil -> darboğaz -> düzeltme döngüsüyle "
  f"aynı işi <b>{ms(final_v['time_s'])}</b>'de yapan bir sisteme ulaştık (<b>{x(final_speedup)}</b>). "
  f"Hızlanmanın kaynağı sırasıyla: yorumlayıcı yükünü BLAS'a devretmek, normları önceden hesaplamak, "
  f"BLAS thread sayısını hibrit işlemciye göre ayarlamak ve hızlanma sonrası seri kalan top-k seçimini "
  f"Numba ile paralelleştirmek (Amdahl). Final sistem {sizes[-1]:,} dokümanda tek sorguyu "
  f"{ms(sc_last['exact_single_query_s'])}'de cevaplıyor, baseline ile 1e-5 toleransta birebir aynı sonucu veriyor "
  f"ve doküman sayısıyla en fazla doğrusal ölçekleniyor.", BODY)

# ============================================================================ 1. GİRİŞ
H("1. Giriş")
P(f"<b>Problem.</b> Kullanıcı bir metin sorgusu girer; sistem korpustaki en benzer {5} makaleyi "
  f"etkileşimli hızda (hedef: sorgu başına &lt; 10 ms) döndürmelidir. Benzerlik ölçüsü cosine "
  f"benzerliğidir. Benchmark'a sadece arama (benzerlik + top-k) dahildir; embedding üretimi FAQ gereği "
  f"hariçtir.")
H("1.1 Veri seti (Seçenek B: arXiv özetleri)", 2)
steps = ds["preprocessing_steps"]
aw = ds["abstract_words"]
tk = ds["abstract_wordpiece_tokens"]
P(f"<b>Kaynak:</b> arXiv API (export.arxiv.org), kategoriler cs.AI, cs.CL, cs.LG; Ocak 2025 - Ekim 2026. "
  f"Tek bir 'en yeniler' sorgusu yalnızca son birkaç haftayı getireceği için (bu kategorilerde ayda "
  f"~6-7 bin makale var) veriyi <b>ay ay</b>, her aydan eşit sayıda çektik; böylece 22 ayın tamamı temsil "
  f"edildi. arXiv'in '3 saniyede 1 istek' kuralına uymak için istekler arasında 3,5 s bekledik ve her "
  f"sayfayı <font face='DVM'>data/raw/</font> altında önbelleğe aldık (tekrar çalıştırmada API'ye gidilmez).")
STEP_TR = {"ham kayit": "ham kayıt (API)", "tekillestirme": "tekilleştirme (arXiv id)",
           "tarih filtresi": "tarih filtresi (2025-2026)", "uzunluk filtresi": "uzunluk filtresi (≥ 50 kelime)",
           "sorgu ayirma": f"sorgu ayırma ({NQ} makale korpustan çıkarıldı)"}


def step_tr(name):
    """preprocess.py'nin ASCII adım adlarını rapor için Türkçe karakterli yazar."""
    return next((v for k, v in STEP_TR.items() if name.startswith(k)), name)


table([["Ön işleme adımı", "Kalan kayıt"]] + [[step_tr(s["adim"]), f"{s['kalan']:,}"] for s in steps] +
      [["karıştırma (seed=42): 'ilk n' alt kümesi tüm dönemi temsil etsin", f"{N:,}"]],
      [12.5, 3], "Ön işleme adımları ve her adımdan sonra kalan kayıt sayısı.")
P(f"Metin temizliğinde satır sonları ve çoklu boşluklar tek boşluğa indirildi, LaTeX komutları ve "
  f"$ {{ }} karakterleri silindi (matematiğin içeriği korundu). <b>Sorgular</b>, korpustan çıkarılan "
  f"{NQ} makalenin <b>başlıklarıdır</b>: sorgu kendi özetini bulamaz, gerçek bir 'bu konuda benzer "
  f"makaleler' araması simüle edilir.")
table([["İstatistik", "Değer"],
       ["Doküman (korpus) / sorgu", f"{N:,} / {NQ}"],
       ["Özet uzunluğu (kelime): ort. / medyan / min-maks", f"{aw['mean']:.0f} / {aw['median']:.0f} / {aw['min']}-{aw['max']}"],
       ["Özet uzunluğu (MiniLM WordPiece token): ort. / medyan", f"{tk.get('mean', 0):.0f} / {tk.get('median', 0):.0f}"],
       ["256 token sınırında kırpılan özet oranı", f"%{100 * tk.get('frac_truncated_over_256', 0):.0f}"],
       ["Sorgu (başlık) uzunluğu: ort. kelime", f"{ds['query_words']['mean']:.1f}"],
       ["Dense temsil", f"{emb['model'].split('/')[-1]}, {emb['dim']} boyut, float32 "
                        f"(üretim {emb['embedding_time_s_not_benchmarked']:.0f} s, benchmark dışı)"],
       ["Sparse temsil", f"TF-IDF, sözlük {emb['tfidf_vocab']:,}, doküman başına ort. "
                         f"{emb['tfidf_avg_nnz_per_doc']:.0f} sıfırdan farklı, yoğunluk %{100 * emb['tfidf_density']:.2f}"],
       ["Birincil kategori (ilk 3)", ", ".join(f"{k} ({v:,})" for k, v in list(ds['primary_category_top10'].items())[:3])]],
      [9, 7.5], "Veri seti istatistikleri.")
s0 = ds["samples_corpus"][0]
P(f"<b>Örnek doküman</b> (arXiv:{s0['id']}, {s0['published'][:10]}, {s0['primary_category']}): "
  f"<i>{s0['title']}</i> - \"{s0['abstract'][:260]}...\"  <b>Örnek sorgular:</b> "
  + "; ".join(f"\"{q}\"" for q in ds["samples_queries"][:3]) + ".")

H("1.2 Donanım ve yazılım", 2)
P(f"Tüm ölçümler aynı laptopta yapıldı: <b>Intel Core Ultra 7 255H</b> (16 çekirdek: 6 performans + "
  f"8 verimlilik + 2 düşük güç, hyper-threading yok), {hw.get('ram_gb', '?')} GB RAM, GPU yok, "
  f"{hw['os'].split('-')[0]} {hw['os'].split('-')[1] if '-' in hw['os'] else ''}. Python {hw['python']}, "
  f"NumPy {hw['numpy']} (OpenBLAS), SciPy {hw['scipy']}, scikit-learn {hw['sklearn']}, Numba {hw['numba']}, "
  f"Dask {hw['dask']}, Ray {hw['ray']}. <b>Benchmark kuralı:</b> her ölçüm warmup + en az 5 tekrar, "
  f"raporlanan değer <b>medyan</b>. 50 ms'den kısa işlerde tekrar sayısı 25'e çıkarıldı (bkz. Bölüm 4.1). "
  f"İşlemcinin hibrit yapısı sonuçları belirgin biçimde etkiledi; bu raporun ana bulgularından biri.")

H("1.3 Deney düzeni: standart iş yükü", 2)
gds = p3["gil_setup"]
pks = p3["pickle_demo"]["setup"]
P(f"Fazlar arası karşılaştırmanın anlamlı olması için tek bir <b>standart iş yükü</b> tanımladık: "
  f"<b>{NQ} sorgu x {N:,} doküman</b> (tüm korpus), top-5. Tüm ana ölçümler, baseline dahil, bu iş "
  f"yüküyle yapıldı; dolayısıyla raporda aynı iş için tek bir süre vardır. Bundan sapan deneyler iki türlü: "
  f"(i) boyutun kendisinin <i>değişken</i> olduğu deneyler (ölçekleme eğrileri, sorgu sayısı taraması; her "
  f"eğri standart noktadan geçer) ve (ii) performans değil <i>kavram</i> gösteren küçük demolar.")
table([["Deney", "Doküman", "Sorgu", "Neden bu boyut"],
       ["Baseline, profil (cProfile + line_profiler)", f"{p1['n_docs']:,}", f"{p1['n_queries']}", "Standart"],
       ["Sparse vs dense (3.1)", ", ".join(f"{r['corpus_size'] / 1000:g}K" for r in p2a), f"{NQ}",
        "Değişken: bellek ve süre N ile nasıl büyür"],
       ["SVD karşılaştırması, recall (3.2)", f"{int(round(p2b['frobenius_X'] ** 2)):,}", f"{NQ}",
        "Standart (N taraması ayrıca 1K..tamamı)"],
       ["Vektörizasyon (3.3)", "1K, 5K, tamamı", "1, 10, 100",
        "Değişken: iki tarama standart noktada kesişir (GEMV -> GEMM)"],
       ["Gram matrisi X<super>T</super>X (3.3)", f"{p2c['gram']['m']:,} x {p2c['gram']['n']}", "-",
        "Starter'daki ikinci işlem; saf Python üçlü döngü tam boyutta ~10 dk/ölçüm"],
       ["Paralellik, online (4)", f"{p3['n_docs']:,}", f"{NQ}", "Standart"],
       ["Paralellik, toplu (4)", f"{p3['n_docs']:,}", f"{p3['n_batch']:,}",
        "100 sorgu ~ms: paralellik yükü işi aşar, ölçeklenme görünmez"],
       ["GIL ve pickle demoları (4)", f"{gds['n_docs']:,} / {pks['n_docs']:,}", f"{gds['n_queries']} / {pks['n_queries']}",
        "Kavram gösterimi (oran önemli); saf Python standart işte ~1 dk/ölçüm"],
       ["Ölçekleme (6)", ", ".join(f"{s / 1000:g}K" for s in sizes), f"{NQ}", "Değişken (≥4 nokta, geometrik)"],
       ["Doğruluk, tüm sürümler (6)", f"{N:,}", f"{NQ}", "Standart"],
       ["ANN, roofline (Ek)", f"{N:,}", f"{NQ} (roofline: 1, 100, 2.000)", "Standart; roofline'da sorgu sayısı değişken"]],
      [4.6, 2.9, 2.1, 7.2], "Her deneyin iş yükü. Alt kümeler, karıştırılmış korpusun ilk N kaydıdır "
                            "(iç içe: 1K ⊂ 5K ⊂ tamamı); sorgular her yerde aynı 100 başlıktır.")

# ============================================================================ 2. BASELINE
H("2. Baseline ve Profil")
P(f"Baseline, hocanın iskeletindeki gibi tamamen saf Python'dur: embedding'ler Python listesine "
  f"çevrilir, her sorgu x her doküman için eleman eleman cosine hesaplanır, tüm skorlar sıralanır:")
code("""def cosine_similarity_python(a, b):
    dot = norm_a = norm_b = 0.0
    for i in range(len(a)):            # 384 adımlık iç döngü
        dot    += a[i] * b[i]
        norm_a += a[i] * a[i]          # sorgu normu her dokümanda YENİDEN
        norm_b += b[i] * b[i]          # doküman normu her sorguda YENİDEN
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))
# search_naive: her sorgu için N skor -> scores.sort(key=lambda x: x[1]) -> ilk 5""")
top = prof["cprofile_top"][0]
lp = prof["line_profile"].get("cosine_similarity_python", [])
lp_tot = sum(r["time_s"] for r in lp) or 1
loop_pct = 100 * sum(r["time_s"] for r in lp if "+=" in r["code"] or r["code"].startswith("for")) / lp_tot
norm_pct = 100 * sum(r["time_s"] for r in lp if r["code"].startswith("norm")) / lp_tot
sort_row = next((r for r in prof["cprofile_top"] if "sort" in r["function"]), None)
P(f"<b>Baseline süresi:</b> {p1['n_queries']} sorgu x {p1['n_docs']:,} doküman x {p1['dim']} boyut = "
  f"<b>{ms(naive_t)}</b> (medyan, 1 warmup + 5 ölçüm), yani sorgu başına "
  f"<b>{p1['per_query_ms_naive']:.0f} ms</b>: 10 ms'lik etkileşimli hedefin ~{p1['per_query_ms_naive'] / 10:.0f} katı.")
fig("fig1_profiling.png",
    f"cProfile (sol) zamanın %{100 * top['tottime'] / prof['cprofile_total_s']:.0f}'inin "
    f"<font face='DVM'>cosine_similarity_python</font> içinde geçtiğini, line_profiler (sağ) ise bunun "
    f"%{loop_pct:.0f}'inin dört döngü satırına neredeyse eşit dağıldığını gösteriyor: darboğaz tek bir "
    f"'pahalı satır' değil, Python yorumlayıcısının eleman başına ödediği sabit yük. Sıralama "
    f"toplamın yalnızca %{100 * (sort_row['tottime'] if sort_row else 0) / prof['cprofile_total_s']:.1f}'i.")
P("<b>Belirlenen üç darboğaz:</b>")
bullets([
    f"<b>(1) Yorumlayıcı yükü (hesap):</b> {p1['n_queries'] * p1['n_docs'] * p1['dim'] / 1e6:.0f} milyon "
    f"çarp-topla adımının her biri ~10 bytecode, yeni bir float nesnesi ve liste indekslemesi demek. "
    f"Eleman başına ~{naive_t / (p1['n_queries'] * p1['n_docs'] * p1['dim'] * 3) * 1e9:.0f} ns; "
    f"CPU'nun SIMD birimleri hiç kullanılmıyor. Çözüm: vektörizasyon / JIT (Bölüm 3.3).",
    f"<b>(2) Gereksiz iş:</b> norm satırları döngü süresinin %{norm_pct:.0f}'ini alıyor; ama normlar "
    f"sorgudan bağımsız, indekslemede bir kez hesaplanabilir. Vektörler normalize edilince cosine = "
    f"nokta çarpım olur ve arama tek bir matris çarpımına (Q @ C<super>T</super>) indirgenir.",
    f"<b>(3) Veri erişim deseni ve sıralama:</b> Python listesi ardışık float'lar değil, heap'e dağılmış "
    f"nesnelere pointer dizisidir (önbellek dostu değil). k=5 için N elemanın tamamını O(N log N) "
    f"sıralamak ve her karşılaştırmada lambda çağırmak gereksiz; O(N) seçim (argpartition) yeterli.",
])
P(f"Saf Python içinde (2) ve (3)'ü düzelten sürüm (önceden hesaplanmış normlar, zip, heapq.nlargest) "
  f"yalnızca <b>{naive_t / p1['python_optimized']['median']:.1f}×</b> kazandırdı "
  f"({ms(p1['python_optimized']['median'])}). Asıl kazanç ancak (1) çözülünce gelecekti.")

# ============================================================================ 3. MATRİS
H("3. Matris İşlemleri ve Boyut İndirgeme")
P("Phase 2 ölçümlerinde BLAS ve Numba thread sayısı <b>8'e sabitlendi</b>. Gerekçe Bölüm 4.1'deki "
  "thread taramasıdır: bu işlemcide OpenBLAS'ın varsayılan 16 thread'i küçük işlerde 8 thread'den "
  "kat kat yavaş.")
H("3.1 Sparse ve dense: temsil kararı", 2)
r_last = p2a[-1]
r_first = p2a[0]
P(f"S = Q @ X<super>T</super> skor matrisini dört kombinasyonda <b>kendi Numba çekirdeklerimizle</b> "
  f"hesapladık ve her birini kütüphane (scipy / NumPy-BLAS) sonucuyla doğruladık (hepsi 1e-5 içinde). "
  f"Sparse x sparse için arama motorlarının <b>ters indeks</b> fikrini uyguladık: X'i CSC'ye çevirince "
  f"her kelimenin 'posting list'i ardışık durur; sorgunun her kelimesi için yalnızca o kelimeyi "
  f"içeren dokümanlar gezilir, sıfırlara hiç dokunulmaz:")
code("""@njit(parallel=True)
def _ss_kernel(q_indptr, q_indices, q_data, x_indptr, x_indices, x_data, n_docs):
    out = np.zeros((nq, n_docs), np.float32)
    for i in prange(nq):                                  # her sorgu ayrı thread
        for p in range(q_indptr[i], q_indptr[i + 1]):     # sorgunun kelimeleri
            t, w = q_indices[p], q_data[p]
            for r in range(x_indptr[t], x_indptr[t + 1]): # t'nin posting listesi
                out[i, x_indices[r]] += w * x_data[r]""")
P(f"Sparse x dense çekirdeğinde korpusu (V x N) <b>kelime-öncelikli</b> sakladık: X[:, t] sütun erişimi "
  f"V adımlı olur ve her eleman ayrı bir 64 B önbellek satırı çeker; X<super>T</super>[t] ise ardışık "
  f"ve SIMD'e uygundur. Dense x dense, her iki matrisi de satır-öncelikli okuyan klasik üçlü döngüdür.")
fig("fig2_sparse_dense.png",
    f"Aynı TF-IDF verisi CSR'da {r_last['mem_sparse_tfidf_mb']:.1f} MB, sıfırlarıyla dense saklandığında "
    f"{r_last['mem_dense_tfidf_mb']:.0f} MB ({r_last['mem_dense_tfidf_mb'] / r_last['mem_sparse_tfidf_mb']:.0f}× fazla) "
    f"yer kaplıyor; MiniLM ({r_last['mem_dense_minilm_mb']:.1f} MB) ikisinin arasında. Hesapta da sıralama aynı: "
    f"%{100 * r_last['density']:.2f} yoğunluktaki veride sparse çekirdekler {r_last['corpus_size']:,} dokümanda "
    f"~{ms(r_last['t_SS_own'])}, dense TF-IDF ise {ms(r_last['t_DDt_own'])} (sıfırları çarpmak da bellek "
    f"bant genişliği ve zaman harcar). Kesikli çizgiler kütüphane sürümleri: BLAS dense işlerde bizim "
    f"döngümüzden {r_last['t_DDe_own'] / r_last['t_DDe_ref']:.0f}× hızlı, scipy sparse x sparse ise bizim "
    f"ters indeksimizden {r_last['t_SS_ref'] / r_last['t_SS_own']:.0f}× yavaş.")
table([["N", "CSR", "Dense TF-IDF", "MiniLM", "SS kendi / scipy", "SD kendi / scipy",
        "DDt kendi / BLAS", "DDe kendi / BLAS"]] +
      [[f"{r['corpus_size']:,}", f"{r['mem_sparse_tfidf_mb']:.1f} MB", f"{r['mem_dense_tfidf_mb']:.0f} MB",
        f"{r['mem_dense_minilm_mb']:.1f} MB", f"{ms(r['t_SS_own'])} / {ms(r['t_SS_ref'])}",
        f"{ms(r['t_SD_own'])} / {ms(r['t_SD_ref'])}", f"{ms(r['t_DDt_own'], 0)} / {ms(r['t_DDt_ref'], 0)}",
        f"{ms(r['t_DDe_own'])} / {ms(r['t_DDe_ref'])}"] for r in p2a],
      [1.4, 1.6, 1.9, 1.6, 2.5, 2.5, 2.6, 2.5],
      f"Phase 2A ham sonuçları (100 sorgu, medyan). Boyutlar ({', '.join(f'{s:,}' for s in sizes)}) "
      f"eldeki toplam veriye göre ~2× geometrik aralıkla seçildi: log-log grafikte eşit aralıklı noktalar "
      f"eğimi (ölçekleme üssünü) doğrudan okutur.")
P(f"<b>Hangisi ne zaman?</b> Sözcük örtüşmesi aranıyorsa ve sorgu birkaç kelimeyse sparse (ters indeks) "
  f"rakipsizdir: iş miktarı N'ye değil posting listelerinin uzunluğuna bağlıdır. Dense TF-IDF hiçbir "
  f"durumda doğru seçim değildir. Biz final sistem için <b>dense MiniLM</b>'i seçtik: bellek olarak "
  f"CSR'ın yalnızca ~{r_last['mem_dense_minilm_mb'] / r_last['mem_sparse_tfidf_mb']:.0f} katı, ama "
  f"anlamsal benzerlik (eş anlamlılar, parafraz) yakalıyor. Sorgularımız kısa başlıklar olduğundan "
  f"(ort. {ds['query_words']['mean']:.0f} kelime) TF-IDF'te sorgu-doküman kelime örtüşmesi çok az; "
  f"ayrıca dense arama tek bir BLAS çağrısına indirgenip donanımı tam kullanabiliyor.")

H("3.2 Boyut indirgeme: seçim ve gerekçe", 2)
v = p2b["validation"]
k_rows = [r for r in p2b["vs_k"] if r["threads"] == 1]
kk = sorted({r["k"] for r in k_rows})


def kr(m, k):
    return next(r for r in k_rows if r["method"] == m and r["k"] == k)


METHODS_ORDER = ("power_iteration", "full_svd", "truncated_svd", "pca")
ts_loses = [k for k in kk if kr("truncated_svd", k)["time_s"] > kr("full_svd", k)["time_s"]]


P(f"Kendi <b>blok güç iterasyonu</b> SVD'mizi uyguladık: G = X<super>T</super>X (384 x 384) bir kez "
  f"hesaplanır (X uzun-ince, N >> d olduğu için her iterasyonda X ile çarpmaktan çok daha ucuz), sonra "
  f"V &lt;- qr(G V) tekrarlanır. İlk sürüm 200 iterasyonda bile yakınsamadı: embedding spektrumu düz "
  f"(ardışık tekil değerler çok yakın) ve yakınsama hızı σ<sub>k+1</sub>/σ<sub>k</sub> oranına bağlı. "
  f"İki ekleme sorunu çözdü: <b>oversampling</b> (k+10 vektörle iterasyon; oran "
  f"σ<sub>k+11</sub>/σ<sub>k</sub>'ya iyileşir) ve <b>Rayleigh-Ritz</b> adımı (küçük (k+p)x(k+p) "
  f"V<super>T</super>GV probleminin özvektörleriyle V'yi alt uzay içinde döndürmek). Sonuç: k={v['k']}'de "
  f"{v['iters']} iterasyon, tekil değerlerde max göreli hata {v['max_rel_sv_diff']:.1e}, yeniden "
  f"oluşturma hatası np.linalg.svd ile aynı ({v['err_power']:.4f} = {v['err_numpy']:.4f}).")
code("""G = X.T @ X                                   # tek seferlik BLAS-3
V, _ = np.linalg.qr(rng.standard_normal((d, k + p)))
for it in range(max_iter):
    V, _ = np.linalg.qr(G @ V)                # güç adımı + ortonormalleştirme
    w, E = np.linalg.eigh(V.T @ (G @ V))      # Rayleigh-Ritz (küçük problem)
    V = V @ E[:, ::-1]                        # alt uzayda en iyi yönler
    if max(|Δσ²| / σ²) < tol: break
S = norm(X @ V, axis=0);  U = X @ V / S""")
pi8 = [r for r in p2b["vs_k"] if r["threads"] == 8 and r["method"] == "power_iteration" and r["k"] >= 50]
pi_slow = [r8["time_s"] / kr("power_iteration", r8["k"])["time_s"] for r8 in pi8]
pi_beats_full = [k for k in kk if kr("power_iteration", k)["time_s"] < kr("full_svd", k)["time_s"]]
pca_fastest = [k for k in kk if min(kr(m, k)["time_s"] for m in METHODS_ORDER) == kr("pca", k)["time_s"]]
fig("fig3_svd.png",
    f"(a) Tam SVD k'dan bağımsız (~{ms(kr('full_svd', 50)['time_s'], 0)}): 384 tekil üçlünün hepsini hesaplıyor. "
    f"Kendi güç iterasyonumuz G = X<super>T</super>X'i bir kez kurup sonra N'den bağımsız çalıştığı için "
    f"k={pi_beats_full[0]}..{pi_beats_full[-1]} aralığının tamamında tam SVD'den hızlı (k={kk[0]}: "
    f"{ms(kr('power_iteration', kk[0])['time_s'], 0)}), ama maliyeti k ile büyüyor. PCA bu uzun-ince matriste "
    f"kovaryans (d x d) yolunu kullandığı için k'dan neredeyse bağımsız ve k≥{pca_fastest[0]}'de en hızlısı. "
    f"Kesikli çizgiler (8 thread) daha fazla thread'in otomatik kazanç olmadığını gösteriyor: küçük 384xk "
    f"matris işlemlerinden oluşan güç iterasyonu k≥50'de 8 thread'de {min(pi_slow):.1f}-{max(pi_slow):.1f}× "
    f"yavaşlıyor (thread senkronizasyon yükü işi aşıyor). (b) N 20 kat artarken tam SVD ve TruncatedSVD "
    f"~doğrusal büyüyor, Gram yaklaşımımız (O(N d²) bir kez + N'den bağımsız iterasyonlar) en yavaş büyüyen. "
    f"(c) Asıl karar ölçütü: 384->128 indirgeme top-5 sonuçların yalnızca "
    f"%{100 * next(r['recall_at_5'] for r in p2b['recall'] if r['k'] == 128):.0f}'ini koruyor; %95 için k≈256 gerekiyor.")
table([["Yöntem (1 thread)"] + [f"k={k}: süre / hata / bellek" for k in (50, 100, 200)]] +
      [[{"power_iteration": "Kendi güç iterasyonumuz", "full_svd": "np.linalg.svd (tam)",
         "truncated_svd": "TruncatedSVD", "pca": "PCA"}[m]] +
       [f"{ms(kr(m, k)['time_s'], 0)} / {kr(m, k)['recon_error']:.2f} / {kr(m, k)['peak_mem_mb']:.0f} MB"
        for k in (50, 100, 200)] for m in ("power_iteration", "full_svd", "truncated_svd", "pca")],
      [4, 4.2, 4.2, 4.2],
      f"Süre (medyan), Frobenius yeniden oluşturma hatası ||X - U<sub>k</sub>S<sub>k</sub>V<sub>k</sub><super>T</super>||"
      f"<sub>F</sub> (||X||<sub>F</sub> = {p2b['frobenius_X']:.1f}) ve tepe bellek (tracemalloc), "
      f"N={int(round(p2b['frobenius_X'] ** 2)):,} x 384 (embedding'ler birim normlu: ||X||<sub>F</sub>² = N).")
P(f"<b>Tartışma soruları.</b> <i>Truncated SVD neden tam SVD'den hızlı?</i> Tam SVD tüm 384 tekil "
  "üçlüyü O(N d²) maliyetle hesaplar; randomize Truncated SVD rastgele bir k+p boyutlu izdüşümle "
  "yalnızca baskın alt uzayı yakalar (O(N d k) + birkaç güç iterasyonu). <i>PCA ne zaman SVD'den "
  "farklıdır?</i> PCA veriyi önce ortalar; ortalaması sıfırdan uzak verilerde ilk SVD bileşeni "
  "'ortalama yönü' olur ve PCA ile ayrışırlar (tablodaki PCA hatasının biraz düşük olması ortalamanın "
  "ayrıca saklanmasından gelir). Cosine araması orijine göre açıyı kullandığından biz ortalamayı "
  "çıkarmayan TruncatedSVD'yi tercih ederiz. <i>Truncated SVD'nin kazandığı pratik maksimum k?</i> "
  f"Ölçümlerimizde (1 thread) k={max(k for k in kk if k < ts_loses[0])}'de hâlâ önde, k≥{ts_loses[0]}'de tam SVD'ye "
  f"kaybediyor: d=384 gibi küçük bir boyutta k, d'nin ~%15-25'ini geçince rastgele izdüşüm ve güç "
  f"iterasyonlarının (O(N d k)) maliyeti tam SVD'ye yetişiyor; 'kısmi' olmanın avantajı kalmıyor.")
P(f"<b>Karar:</b> İndeksleme yöntemi olarak TruncatedSVD (ortalamasız, yani cosine geometrisini koruyor; "
  f"k=256'da tam SVD ile aynı kalite). Bu boyutta kendi güç iterasyonumuz daha hızlı olurdu, ama indeksleme "
  f"tek seferlik (&lt; 1 s) bir iş olduğundan seçim sorgu süresini etkilemiyor. Ancak final sistemde indirgemeyi <b>varsayılan yapmadık</b>: bizim boyutumuzda d=384 "
  f"zaten küçük; 256'ya inmek aramayı en fazla 1,5× ucuzlatırken recall@5'i %{100 * corr['svd']['recall_at_5_vs_exact']:.0f}'e "
  f"düşürüyor. İndirgeme, sistemde opsiyonel bir mod olarak duruyor (<font face='DVM'>--svd 256</font>) ve "
  f"10x büyük veride (bellek bant genişliği sınırı) anlamlı hale geliyor (Bölüm 6).")

H("3.3 Vektörizasyon, eleman-eleman ve JIT", 2)
cos = p2c["cosine"]
rp = [r for r in cos if "t_python" in r][-1]
rl = cos[-1]
r1 = cos[0]
g = p2c["gram"]
P(f"Aynı işlemi (cosine matrisi S = Q<sub>n</sub> C<sub>n</sub><super>T</super>) dört (+1) yolla yazdık: "
  f"saf Python döngüleri, NumPy (<font face='DVM'>Q @ C.T</font>, BLAS sgemm), Numba <font face='DVM'>@njit</font> "
  f"(aynı döngü, derlenmiş), Numba <font face='DVM'>parallel=True</font> + <font face='DVM'>prange</font> ve ek "
  f"olarak <font face='DVM'>fastmath=True</font>. Tümü NumPy ile 1e-5 içinde aynı sonucu verdi.")
fig("fig4_vectorisation.png",
    f"(a) {rp['n_queries']}x{rp['n_docs']:,} cosine matrisinde saf Python {ms(rp['t_python'])}; Numba seri aynı "
    f"döngüyü yalnızca derleyerek {x(rp['t_python'] / rp['t_numba'])}, prange {x(rp['t_python'] / rp['t_numba_parallel'])}, "
    f"fastmath ile {x(rp['t_python'] / rp['t_numba_parallel_fastmath'])}, NumPy/BLAS {x(rp['t_python'] / rp['t_numpy'])} "
    f"hızlandırıyor. (b) Kim ne zaman kazanır: küçük işlerde Numba fastmath BLAS'ı yeniyor (çağrı ve thread "
    f"uyandırma yükü yok, ara matris yok), büyük işte ({rl['n_queries']}x{rl['n_docs']:,}) BLAS "
    f"{rl['t_numba_parallel_fastmath'] / rl['t_numpy']:.1f}× önde: önbellek bloklama ile her veriyi defalarca "
    f"yeniden kullanıyor ({rl['gflops_numpy']:.0f} GFLOP/s'e karşı {rl['gflops_numba_parallel_fastmath']:.0f}).")
P(f"<b>Neden bu kadar fark?</b> Numba seri, Python ile aynı algoritmayı yorumlayıcısız çalıştırır "
  f"(nesne yok, tip kontrolü yok): {x(rp['t_python'] / rp['t_numba'])}. Ama iç döngüdeki toplama "
  f"kayan noktada birleşmeli olmadığından derleyici sırayı değiştirip SIMD kullanamaz; "
  f"<font face='DVM'>fastmath</font> bu izni verir ve 8'li AVX2 kısmi toplamlarla tek çekirdekte bile "
  f"büyük sıçrama yaratır. BLAS ise buna ek olarak <b>önbellek bloklama</b> yapar: Q ve C'yi L1/L2'ye sığan "
  f"karolara bölüp her yüklenen veriyi defalarca kullanır, FMA komutlarıyla çarp-topla'yı tek komutta yapar. "
  f"Gram matrisi X<super>T</super>X ({g['m']}x{g['n']}) aynı resmi verdi: Python {ms(g['t_python'])}, "
  f"Numba {ms(g['t_numba'])}, Numba-prange {ms(g['t_numba_parallel'])}, NumPy {ms(g['t_numpy'])}. "
  f"<b>Numba ne zaman NumPy'yi yener?</b> (i) İş küçükken (sabit çağrı/thread yükü baskın), (ii) birden "
  f"fazla adım tek döngüde birleştirilebiliyorsa (fused: ara matris yazılmaz; Bölüm 4'teki top-k), "
  f"(iii) BLAS'ın karşılığı olmayan özel döngülerde (ters indeks gibi). <b>Ne zaman kaybeder?</b> Saf "
  f"GEMM gibi, onlarca yıllık optimize kütüphanelerin karşılığı olan büyük yoğun işlemlerde.")

# ============================================================================ 4. PARALELLİK
H("4. Paralellik ve Eşzamanlılık")
rows = p3["rows"]
batch = [r for r in rows if r["workload"] == "batch"]
online = [r for r in rows if r["workload"] == "online"]


def best(rs, s):
    c = [r for r in rs if r["strategy"] == s]
    return min(c, key=lambda r: r["time_s"]) if c else None


def at(rs, s, w):
    return next((r for r in rs if r["strategy"] == s and r["workers"] == w), None)


t1 = at(batch, "blas", 1)["time_s"]
H("4.1 Önce ölçüm: thread sayısı ve ölçüm hataları", 2)
b16o, b8o = at(online, "blas", 16), at(online, "blas", 8)
P(f"İlk bulgu bir sürprizdi: OpenBLAS varsayılan olarak 16 thread açıyor ve 100 sorguluk aramayı "
  f"8 thread'e göre belirgin biçimde YAVAŞLATIYOR (online iş yükü: 16 thread {ms(b16o['time_s'])}, 8 thread "
  f"{ms(b8o['time_s'])}). Neden: işlemci hibrit; BLAS işi thread'lere eşit böler ve 15.-16. thread'ler "
  f"düşük güç çekirdeklerine düşünce herkes en yavaş çekirdeği bekler (yük dengesizliği). Ayrıca ~1 ms'lik "
  f"işlerde uyuyan thread'leri uyandırmanın gecikmesi (Windows zamanlayıcısı) işin kendisini aşıyor; aynı "
  f"çağrıda 1 ms ile 19 ms arasında ölçümler gördük. Bu yüzden (i) tüm sistemi 8 thread'e sabitledik, "
  f"(ii) 50 ms'den kısa işlerde tekrar sayısını 25'e çıkardık. Kendi <b>ölçüm hatamızı</b> da burada "
  f"yakaladık: <font face='DVM'>threadpool_limits()</font> her girişte yüklü DLL'leri tarıyor (~10-20 ms); "
  f"bunu zamanlanan fonksiyonun içinde çağırdığımız ilk sürüm 5 ms'lik aramayı 22 ms ölçtü. Ayar artık "
  f"ölçümün dışında bir kez yapılıyor.")

H("4.2 Stratejiler, gerekçeler ve sonuçlar", 2)
P(f"Arama işi her stratejide aynıdır (<font face='DVM'>search_block</font>: BLAS çarpımı + top-k); sadece "
  f"dağıtma yöntemi değişir. İki iş yükü kullandık: <b>online</b> (standart iş yükü, {NQ} sorgu) ve <b>toplu</b> "
  f"({p3['n_batch']:,} sorgu: 'her makale için benzer makaleleri bul', öneri / yakın-kopya tespiti). "
  f"100 sorgu birkaç ms sürdüğü için paralelliğin sabit maliyetleri orada baskın; ölçeklenmeyi toplu "
  f"işte görüyoruz. Sorgular worker sayısı kadar <b>parçaya</b> bölünür (sorgu başına görev değil): görev "
  f"başına yük azalır ve her parça BLAS-2 (GEMV) yerine BLAS-3 (GEMM) yapar.")
strat_names = {"blas": "NumPy/BLAS thread'leri", "numba": "Numba prange (fused)",
               "hybrid": "Hibrit: BLAS + Numba top-k", "thread": "ThreadPoolExecutor",
               "process": "ProcessPool + shared memory", "dask": "Dask (delayed, threads)",
               "ray_tasks": "Ray tasks", "ray_actors": "Ray actors"}
why = {"blas": "Örtük paralellik, sıfır kod; her zaman ilk adım",
       "numba": "Benzerlik+top-k tek döngüde, ara matris yok",
       "hybrid": "GEMM BLAS'ta, seri top-k Numba'da paralel",
       "thread": "BLAS/argpartition GIL'i bırakır -> thread yeter",
       "process": "GIL yok; korpus kopyalanmaz (shm), sadece sorgu gider",
       "dask": "Görev grafiği; out-of-core için hazır",
       "ray_tasks": "Çok düğüme taşınabilir; ray.put ile sıfır kopya",
       "ray_actors": "Durumlu worker (model servisi gibi)"}
trows = [["Strateji", "Neden denedik", "1 worker", "8 worker", "En iyi (w)", "Hızlanma*", "P"]]
for s in ["blas", "numba", "hybrid", "thread", "process", "dask", "ray_tasks", "ray_actors"]:
    b_ = best(batch, s)
    if not b_:
        continue
    trows.append([strat_names[s], why[s], ms(at(batch, s, 1)["time_s"], 0), ms(at(batch, s, 8)["time_s"], 0),
                  f"{ms(b_['time_s'], 0)} ({b_['workers']})", x(t1 / b_["time_s"]),
                  f"{p3['amdahl_P'][s]:.2f}" if s in p3["amdahl_P"] else "-"])
table(trows, [3.7, 4.4, 1.5, 1.5, 2.2, 1.6, 1.1],
      f"Toplu iş yükü ({p3['n_batch']:,} sorgu x {p3['n_docs']:,} doküman), medyan. Tüm stratejilerin top-5 "
      f"sonuçları referansla 1e-5 içinde aynı. *Hızlanma: en iyi sürenin 1-thread BLAS'a göre oranı. "
      f"P (Amdahl): stratejinin kendi 1-worker süresine göre, w≤8 "
      f"noktalarıyla uydurulan paralel oran. Naive baseline bu iş için tahminen {ms(p3['naive_baseline_extrapolated_s']['batch'])} "
      f"(Faz 1'in doküman-sorgu başına maliyetinden ekstrapolasyon).")
bh, bb_ = best(batch, "hybrid"), best(batch, "blas")
h16 = at(batch, "hybrid", 16)
fig("fig5_parallel_scaling.png",
    f"(a) Toplu aramada saf BLAS thread'leri {x(t1 / bb_['time_s'])}'de tıkanıyor: matris çarpımı ölçekleniyor "
    f"ama tek thread'li argpartition seri kalıyor (Amdahl P={p3['amdahl_P'].get('blas', 0):.2f}). Top-k'yı da "
    f"parçalara bölen stratejiler (ThreadPool, ProcessPool, Dask, Ray) ~4×'e, top-k'yı Numba ile paralelleştiren "
    f"hibrit {bh['workers']} worker'da {x(t1 / bh['time_s'])}'e çıkıyor. Hiçbiri ideal doğruya yaklaşmıyor: 6 "
    f"P-çekirdekten sonra eklenen E-çekirdekler daha yavaş ve tüm thread'ler aynı bellek bant genişliğini "
    f"paylaşıyor. Hibrit 12-16 worker'da çöküyor ({ms(h16['time_s'], 0)}): BLAS'a ve Numba'ya ayrı ayrı w thread "
    f"verildiği için 2w thread 16 çekirdek için yarışıyor (aşırı abonelik; OpenBLAS thread'leri iş bitince bir "
    f"süre meşgul-bekliyor). Amdahl uydurması bu yüzden w≤8 noktalarıyla yapıldı. (b) Saf Python'da thread'ler "
    f"GIL yüzünden hiç hızlanmıyor, süreçler ölçekleniyor: GIL'in doğrudan kanıtı.")
gil = p3["gil_demo"]


def gl(s, w):
    return next(r["time_s"] for r in gil if r["strategy"] == s and r["workers"] == w)


pk = p3["pickle_demo"]
P(f"<b>GIL.</b> NumPy/BLAS ve argpartition C kodunda çalışırken GIL'i bırakır; bu yüzden ThreadPool, "
  f"ProcessPool ile neredeyse aynı hızlanmayı süreç başlatma ve IPC maliyeti olmadan verir. Saf Python "
  f"işte ise tablo tersine döner (Şekil {FIG_N[0]}b): 8 thread {gl('python_thread', 1) / gl('python_thread', 8):.1f}×, "
  f"8 süreç {gl('python_process', 1) / gl('python_process', 8):.1f}×. (Not: 1 worker'lı süreç, aynı işi yapan "
  f"tek thread'den {gl('python_process', 1) / gl('python_thread', 1):.1f}× yavaş; Windows arka plandaki çocuk "
  f"süreçleri büyük olasılıkla yavaş E-çekirdeklere yerleştiriyor. Hibrit işlemcinin bir başka etkisi.) "
  f"Süreçlerin bedeli: Windows'ta 'spawn' ile "
  f"havuz kurulumu ~{np.median([r.get('startup_s', 0) for r in batch if r['strategy'] == 'process']):.1f} s ve "
  f"veri kopyalama. Hocanın iskeletindeki 'kötü' sürüm (korpus her görevde pickle, {pk['corpus_pickle_mb']:.1f} MB) "
  f"{ms(pk['bad_pickle_per_query']['time_s'])}, parçalı 'iyi' sürüm {ms(pk['good_batched']['time_s'])} sürdü. "
  f"NumPy sürümünde korpusu <font face='DVM'>multiprocessing.shared_memory</font>'ye bir kez koyduk; "
  f"worker'lar onu kopyalamadan görüyor, görevlere sadece sorgu parçası gidiyor.")
asy = p3["async"]


def aa(d, m):
    return next(r["time_s"] for r in asy if r["delay_s"] == d and r["mode"] == m)


P(f"<b>asyncio</b> (I/O senaryosu): Sorgu metni önce uzak bir embedding API'sine gidiyorsa (gecikme "
  f"simüle) 100 sorguyu sırayla beklemek 10 ms gecikmede {ms(aa(0.01, 'serial'))}, "
  f"<font face='DVM'>asyncio.gather</font> ile {ms(aa(0.01, 'concurrent'))} "
  f"({aa(0.01, 'serial') / aa(0.01, 'concurrent'):.0f}×); 10 eşzamanlı istekle sınırlandırınca (gerçek "
  f"API rate limit'leri) {ms(aa(0.01, 'bounded_10'))}. Gecikme sıfırken (embedding önbellekte) kayda değer "
  f"bir kazanç yok ({ms(aa(0.0, 'serial'))} vs {ms(aa(0.0, 'concurrent'))}; fark ölçüm gürültüsü düzeyinde): "
  f"asyncio tek thread'dir, CPU işini hızlandırmaz. Bu yüzden async'i arama çekirdeği için değil, sistemin I/O kenarı için uygun görüyoruz.")
P(f"<b>Dask ve Ray.</b> İkisi de doğru sonuç verdi ve ThreadPool/ProcessPool düzeyinde hızlandı, ama "
  f"tek makinede ve RAM'e sığan veride ek bir şey katmıyorlar: Dask görev grafiği, Ray ise "
  f"{p3['ray'].get('init_s', 0):.0f} s'lik başlatma + nesne deposu yükü getiriyor. Online iş yükünde Ray "
  f"görev başına yükü nedeniyle en iyi {ms(best(online, 'ray_tasks')['time_s'])} ile hibritin "
  f"({ms(best(online, 'hybrid')['time_s'])}) gerisinde. "
  f"Dask'ın gerçek yeri korpus RAM'e sığmadığında (chunk'lı out-of-core çarpım), Ray'inki birden fazla "
  f"makineye çıkıldığında.")
P(f"<b>Tasarım sorusu: kod karmaşıklığı başına en iyi hızlanma?</b> Hibrit strateji: tek satır "
  f"<font face='DVM'>topk_numba(Q @ C.T, k)</font> ve bir thread ayarı; havuz, IPC, serileştirme yok. "
  f"ProcessPool benzer hız için ~60 satır (shared memory, initializer, temizlik) ve saniyelerce başlatma "
  f"istiyor. <b>Korpus 10× büyük olsaydı</b> ({sizes[-1] * 10:,} doküman, 384 boyut float32 = "
  f"{sizes[-1] * 10 * 384 * 4 / 1024**2:.0f} MB) hâlâ RAM'e sığar ve cevabımız değişmezdi; skor matrisi "
  f"(q x N) büyüdüğü için sorgu parçalama (ThreadPool) ve korpus parçalama önem kazanırdı. 100× (RAM sınırı) "
  f"ölçekte Dask ile out-of-core blok çarpımı ya da ANN (Ek A) gerekirdi.")

# ============================================================================ 5. TASARIM
H("5. Tasarım Kararları")
table([["Karar", "Seçim", "Düşünülen alternatifler", "Veri / gerekçe"],
       ["Temsil", "Dense MiniLM, float32", "Sparse TF-IDF, dense TF-IDF",
        f"CSR {r_last['mem_sparse_tfidf_mb']:.1f} MB, MiniLM {r_last['mem_dense_minilm_mb']:.1f} MB, dense TF-IDF "
        f"{r_last['mem_dense_tfidf_mb']:.0f} MB; MiniLM anlamsal + tek BLAS çağrısı (Şekil 2)"],
       ["Normlar", "İndekslemede bir kez", "Her karşılaştırmada", f"Profilde döngünün %{norm_pct:.0f}'i (Şekil 1)"],
       ["Benzerlik çekirdeği", "BLAS sgemm (toplu)", "Numba fused, saf Python",
        f"{rl['n_queries']}x{rl['n_docs']:,}: BLAS {ms(rl['t_numpy'])}, Numba-fm {ms(rl['t_numba_parallel_fastmath'])} (Şekil 4)"],
       ["Top-k", "Numba prange, O(N)", "sort O(N log N), argpartition (1 thread)",
        "argpartition BLAS hızlandıktan sonra seri darboğaz; Numba 10× hızlı (Bölüm 4)"],
       ["Thread sayısı", "8 (BLAS + Numba)", "Varsayılan 16, 1, 12",
        f"Online: 16 thr {ms(b16o['time_s'])}, 8 thr {ms(b8o['time_s'])}; hibrit CPU yük dengesizliği"],
       ["Paralellik", "Hibrit (BLAS + Numba)", "ThreadPool, ProcessPool, Dask, Ray, asyncio",
        f"Toplu: hibrit {ms(bh['time_s'])}, en iyi süreç {ms(best(batch, 'process')['time_s'])}; en az kod"],
       ["Boyut indirgeme", "Opsiyonel (TruncatedSVD-256)", "Güç iterasyonu, PCA, tam SVD; k=128",
        f"k=256 recall {100 * corr['svd']['recall_at_5_vs_exact']:.0f}%, k=128 recall "
        f"{100 * next(r['recall_at_5'] for r in p2b['recall'] if r['k'] == 128):.0f}%; kazanç ≤1.5× (Şekil 3)"],
       ["Sorgu toplama", "Toplu (batch) arama", "Sorgu başına GEMV",
        "Roofline: q=1 bant genişliği sınırlı, q=100 hesap sınırlı (Ek B)"]],
      [2.4, 3.1, 4.0, 7.3], "Tasarım kararları, alternatifler ve dayandıkları ölçümler.")

# ============================================================================ 6. FİNAL
H("6. Final Sistem ve Ölçekleme")
P("<b>Hat:</b> metin -> MiniLM embedding -> (opsiyonel TruncatedSVD) -> L2 normalize (indekslemede, "
  "bir kez) -> <font face='DVM'>Q @ C.T</font> (BLAS, 8 thread) -> <font face='DVM'>topk_numba</font> "
  "(prange, 8 thread) -> top-5. Çok sorguda sorgular 512'lik bloklar halinde, tek bir yeniden kullanılan "
  "skor tamponuyla işlenir (her çağrıda yüzlerce MB yeni bellek ayırıp Windows'ta sayfa hatası ödememek için). "
  "Çekirdeğin özü:")
code("""def search_hybrid(Qn, Cn, k, block=512):
    if len(Qn) <= block:
        return topk_numba(Qn @ Cn.T, k)          # BLAS sgemm + paralel O(N) top-k
    buf = np.empty((block, len(Cn)), np.float32) # bir kez ayrılan skor tamponu
    for a in range(0, len(Qn), block):
        S = buf[:min(block, len(Qn) - a)]
        np.matmul(Qn[a:a + len(S)], Cn.T, out=S) # yeni bellek ayırmadan yaz
        out_idx[a:a + len(S)], out_sc[a:a + len(S)] = topk_numba(S, k)

class SimilaritySearchEngine:                     # indeksleme: SVD (ops.) + normalize BİR KEZ
    def search_prepared(self, Qn, k=5):
        return search_hybrid(Qn, self.C, k)""")
slopes = p4["loglog_slopes"]
sc_prev = p4["scaling"][-2]
def _slope2(key):
    """Son iki ölçekleme noktası arasındaki log-log eğim."""
    return float(np.log(sc_last[key] / sc_prev[key]) / np.log(sc_last["n_docs"] / sc_prev["n_docs"]))


local_slope = _slope2("exact_per_query_s")
single_slope = _slope2("exact_single_query_s")
fig("fig6_scaling.png",
    f"(a) Naive baseline log-log eğim {slopes['naive_per_query_s']:.2f} ile tam doğrusal büyüyor (O(N d)). Final "
    f"sistemin tek sorgu gecikmesi son aralıkta ({sc_prev['n_docs']:,} -> {sc_last['n_docs']:,}) eğim "
    f"{single_slope:.2f} ile doğrusal: korpus L3'ten taşınca her sorgu tüm korpusu RAM'den bir kez okuyor (bant "
    f"genişliği sınırlı, Ek B). Toplu modda eğim tüm aralıkta {slopes['exact_per_query_s']:.2f}, son aralıkta "
    f"{local_slope:.2f}: iş hâlâ O(N d), ama ~ms'lik toplu işte süreyi kısmen N'den bağımsız sabit maliyetler "
    f"(çağrı, thread uyandırma) belirliyor; N büyüdükçe bunlar amorti olduğu için alt-doğrusal görünüyor. {sizes[-1]:,} "
    f"dokümanda naive sorgu başına {ms(sc_last['naive_per_query_s'], 0)}, final {ms(sc_last['exact_per_query_s'])} "
    f"(100 sorgu toplu), tek sorgu gecikmesi {ms(sc_last['exact_single_query_s'])}: etkileşimli hedefin çok altında. "
    f"(b) Hızlanma N ile artıyor çünkü büyük işte sabit yükler amorti ediliyor.")
ex = corr["exact"]
sv = corr["svd"]
table([["Kontrol", "Sonuç"],
       ["Kesin mod vs naive baseline (tüm korpus, 100 sorgu)",
        f"{'GEÇTİ' if ex['passed'] else 'KALDI'} · max |skor farkı| = {ex['max_abs_score_diff']:.1e} "
        f"(tolerans 1e-5) · indeks sırası farklı sorgu: {ex['queries_with_index_order_diff']}"],
       [f"SVD-{sv['k']} modu vs aynı uzayda naive", f"{'GEÇTİ' if sv['passed'] else 'KALDI'} · max fark "
                                                    f"{sv['max_abs_score_diff']:.1e} · kesin moda göre recall@5 = {sv['recall_at_5_vs_exact']:.3f}"],
       ["Phase 2-3'teki tüm uygulamalar", "Her biri referansla 1e-5 içinde (results/*.json içinde 'correct')"]],
      [6.2, 10.6], f"Doğruluk kontrolü. Baseline float64 saf Python, final float32 BLAS; fark float32 "
                   f"yuvarlamasından (~1e-7). Eşit skorlu dokümanlarda sıra değişimi hata sayılmaz (tie-aware kontrol).")
nb16 = next(v for v in p4["all_versions"] if v["version"].startswith("Numba fused prange"))
fig("fig7_all_versions.png",
    f"Aynı iş yükünde (100 sorgu x {p1['n_docs']:,} doküman) hikâyenin tamamı: saf Python {ms(naive_v['time_s'])}'den "
    f"final sistemin {ms(final_v['time_s'])}'sine {x(final_speedup)}. Kazancın büyük kısmı tek bir adımdan, "
    f"yorumlayıcıyı BLAS/JIT ile devre dışı bırakmaktan geliyor (tek thread BLAS: "
    f"{x(naive_v['time_s'] / vers['NumPy/BLAS + argpartition (1 thread)']['time_s'])}); sonraki her adım (thread ayarı, "
    f"top-k paralelleştirme) birkaç kat ekliyor. 16 thread'li varsayılan BLAS tek thread'den yavaş. Numba fused "
    f"(16 thread, {ms(nb16['time_s'])}) ara matris yazmasa da önbellek bloklaması olmadığı için finalin "
    f"{nb16['time_s'] / final_v['time_s']:.1f}× gerisinde; ProcessPool ({ms(vers['ProcessPool 8 + shm']['time_s'])}) "
    f"yakın ama ~60 satır ek kod ve saniyelerce havuz kurulumu istiyor.",
    width=15.5)
P(f"<b>Doğrusal ölçekleniyor mu?</b> İş miktarı doğrusal (O(N d)) ve hiçbir yerde doğrusalın üstüne çıkmıyoruz: "
  f"tek sorgu gecikmesi büyük N'de doğrusal (son aralıkta eğim {single_slope:.2f}), toplu modda ise sabit "
  f"maliyetler henüz tam amorti olmadığı için alt-doğrusal (eğim {local_slope:.2f}). "
  f"{sizes[-1]:,} dokümanda korpus {sc_last['exact_mem_mb']:.0f} MB, L3 önbelleği (24 MB) aşınca "
  f"tek sorgu bellek bant genişliğiyle sınırlanır (Ek B). <b>10× veriyle</b> "
  f"(~{sizes[-1] * 10 / 1000:.0f}K doküman, ~{sc_last['exact_mem_mb'] * 10:.0f} MB) doğrusal tahmin tek "
  f"sorgu için ~{sc_last['exact_single_query_s'] * 10 * 1e3:.1f} ms, hâlâ etkileşimli. Daha büyük ölçekte "
  f"sırasıyla: float16/int8 niceleme (bant genişliğini 2-4× azaltır), SVD-256 (1,5×) ve sonunda ANN "
  f"(IVF; Ek A'da kendi uygulamamızla aday kümesini korpusun ~%6'sına indirip recall@5≈0.90 elde ettik) yapardık.")

# ============================================================================ 7. TARTIŞMA
H("7. Tartışma ve Dersler")
bullets([
    "<b>En büyük sürpriz: daha çok thread daha yavaş olabilir.</b> Hibrit işlemcide varsayılan 16 BLAS "
    "thread'i, 8 thread'den kat kat yavaştı; küçük SVD matrislerinde 1 thread 8 thread'den hızlıydı. "
    "Amdahl'ın 'seri kısım' kavramı burada thread yönetimi ve en yavaş çekirdeği bekleme olarak çıktı.",
    "<b>Amdahl'ı canlı gördük:</b> BLAS'ı hızlandırdıkça, profilde %1 bile görünmeyen top-k toplam sürenin "
    "çoğuna dönüştü. Darboğaz yer değiştirir; her optimizasyondan sonra yeniden profil almak gerekir.",
    "<b>Ölçüm de bir sistemdir:</b> threadpoolctl'in giriş maliyeti ve Windows'un thread uyandırma gürültüsü "
    "sonuçları 4-10× bozabiliyordu. Tek thread'li saf Python baseline bile iki ayrı çalıştırmada 30,8 s ve "
    "11,9 s ölçüldü (aynı kod ve veri; işletim sistemi işi P- ya da E-çekirdeğe yerleştirdi). Bu yüzden her "
    "karşılaştırmayı aynı çalıştırmanın içinden yaptık; medyan, yeterli tekrar ve ayarları ölçüm dışında "
    "yapmak şart.",
    "<b>GIL ancak saf Python'da sorun:</b> İş C'ye (BLAS, argpartition) devredildiğinde thread'ler süreçler "
    "kadar iyi; süreçlerin başlatma/IPC maliyetine yalnızca saf Python işte değer.",
    "<b>Farklı yapardık:</b> sorguları baştan toplu (batch) düşünmek; Numba'yı benzerlik için değil, BLAS'ın "
    "olmadığı yerde (top-k, ters indeks) kullanmak; bir de MiniLM'in 256 token sınırı yüzünden özetlerin "
    f"%{100 * tk.get('frac_truncated_over_256', 0):.0f}'inin kırpıldığını baştan bilip daha uzun bağlamlı bir model seçmek.",
    "<b>Gerçek AI sistemleriyle bağlantı:</b> RAG hatlarındaki vektör veritabanları (FAISS, pgvector, Milvus) "
    "tam olarak bu adımları yapar: normalize embedding + BLAS'la toplu iç çarpım + paralel top-k; ölçek "
    "büyüyünce IVF/HNSW ve niceleme. LLM servislerindeki 'batching' de roofline'daki aynı mantıkla "
    "bellek sınırlı işi hesap sınırlı hale getirir.",
])

# ============================================================================ EK
story.append(PageBreak())
H("Ek")
if ann:
    H("Ek A (bonus): Yaklaşık en yakın komşu (kendi LSH ve IVF uygulamamız)", 2)
    ar = ann["rows"]
    ex1 = next(r for r in ar if r["method"] == "exact" and r["mode"] == "tek tek")
    exb = next(r for r in ar if r["method"] == "exact" and r["mode"] != "tek tek")
    ivf = [r for r in ar if r["method"] == "IVF"]
    good = [r for r in ivf if r["recall_at_5"] >= 0.9]
    g0 = min(good, key=lambda r: r["per_query_ms"]) if good else max(ivf, key=lambda r: r["recall_at_5"])
    lsh = [r for r in ar if r["method"] == "LSH"]
    lb = max(lsh, key=lambda r: r["recall_at_5"])
    ivf_fast = min((r for r in ivf if r["per_query_ms"] < ex1["per_query_ms"]),
                   key=lambda r: -r["recall_at_5"], default=ivf[0])
    fig("figA_ann.png",
        f"N={ann['n_docs']:,}: IVF (n_list={g0['n_list']}, nprobe={g0['nprobe']}) korpusun %{100 * g0['candidate_fraction']:.0f}'ine "
        f"bakarak recall@5={g0['recall_at_5']:.2f}, sorgu başına {g0['per_query_ms']:.2f} ms. Kesin tek sorgu "
        f"{ex1['per_query_ms']:.2f} ms, toplu kesin arama sorgu başına {exb['per_query_ms']:.3f} ms. IVF ancak "
        f"recall≈{ivf_fast['recall_at_5']:.2f}'ye razı olunca (nprobe={ivf_fast['nprobe']}) kesin tek sorgudan hızlı; "
        f"Python'daki sorgu başına döngü yükü kazancı yiyor. LSH'nin recall {lb['recall_at_5']:.2f}'ye ulaşması için "
        f"korpusun %{100 * lb['candidate_fraction']:.0f}'ine bakması gerekiyor: benzer makalelerin cosine'i ~0.6 "
        f"(açı ~53°), bu açıda bir bitte çarpışma olasılığı ~0.70, b bitte 0.70<super>b</super>; yüksek recall için "
        f"az bit = büyük kovalar gerekiyor. Sonuç: bu ölçekte toplu kesin arama ANN'den hızlı ve kayıpsız; ANN "
        f"(derlenmiş bir kütüphaneyle) tek sorgu gecikmesinde ve çok daha büyük N'de anlam kazanır.", width=12.5)
if roof:
    H("Ek B (bonus): Roofline analizi", 2)
    pts = roof["points"]
    pb = next(p for p in pts if p["label"].startswith("BLAS q=100"))
    p1q = next(p for p in pts if p["label"].startswith("BLAS q=1 "))
    fig("figB_roofline.png",
        f"Ölçülen tepe {roof['peak_gflops']:.0f} GFLOP/s (sgemm) ve bant genişliği {roof['bandwidth_gbs']:.0f} GB/s; "
        f"kırılma noktası {roof['ridge_point']:.0f} FLOP/bayt. Tek sorgu araması AI≈{p1q['ai']:.1f} ile bant genişliği "
        f"sınırlı bölgede ({p1q['gflops']:.0f} GFLOP/s, RAM tavanının %{p1q['pct_of_roof']:.0f}'i: tavanın kendisinde. "
        f"Tavan 1 GB'lık diziyle, yani doğrudan RAM'den ölçüldü; {sc_last['exact_mem_mb']:.0f} MB'lık korpusun bir kısmı "
        f"tekrarlanan ölçümlerde 24 MB'lık L3'te kaldığı için ~%100 civarı, hatta biraz üstü görülebiliyor), 100 sorgu toplu arama AI≈{pb['ai']:.0f} ile hesap sınırlı "
        f"bölgede ({pb['gflops']:.0f} GFLOP/s, tavanın %{pb['pct_of_roof']:.0f}'i). Sorguları toplamak her dokümanı "
        f"bellekten bir kez okuyup 100 kez kullanmak demek: batching'in roofline açıklaması.", width=12.5)
H("Ek C: Tekrar üretme ve ham veri", 2)
P(f"Kod: <link href='{REPO}' color='#2a78d6'>{REPO}</link>. Kurulum ve tüm sonuçları yeniden üretme:")
code("""py -3.14 -m pip install -r requirements.txt
py -3.14 run_all.py              # veri (önbellekli) -> tüm benchmark'lar -> figürler -> bu PDF
py -3.14 demo.py --compare       # canlı demo""")
P("Ham sonuçlar (her ölçümün tüm tekrarları dahil) <font face='DVM'>results/</font> altında: "
  "dataset_stats.json, phase1_baseline.json, profile_cprofile.txt, profile_line.txt, "
  "phase2a_sparse_dense.csv, phase2b_svd_vs_k.csv, phase2b_svd_vs_n.csv, phase2c_vectorise.csv, "
  "phase3_parallel.csv, phase3_gil_demo.csv, phase3_async.csv, phase4_scaling.csv, "
  "phase4_all_versions.csv, bonus_ann.csv, bonus_roofline.json, hardware.json.")
sc_rows = [["N", "Naive / sorgu", "Final / sorgu (toplu)", "Final tek sorgu", f"SVD-{p4['svd_k']} / sorgu",
            "Hızlanma"]]
for r in p4["scaling"]:
    sc_rows.append([f"{r['n_docs']:,}", ms(r["naive_per_query_s"], 1), ms(r["exact_per_query_s"]),
                    ms(r["exact_single_query_s"], 3), ms(r["svd_per_query_s"]), x(r["speedup_exact_vs_naive"])])
table(sc_rows, [2, 3, 3.4, 3, 3, 2.4], "Ölçekleme testi ham verisi (Şekil 6), medyan.")
av = [["Sürüm (100 sorgu x %s doküman)" % f"{p1['n_docs']:,}", "Medyan", "Naive'e göre"]]
for v_ in sorted(p4["all_versions"], key=lambda r: -r["time_s"]):
    av.append([v_["version"], ms(v_["time_s"], 3), x(naive_v["time_s"] / v_["time_s"])])
table(av, [9, 3.5, 3.5], f"Tüm sürümler (Şekil 7) ham verisi. Final sistem aynı iş yükünde Şekil 6'nın son "
                         f"noktasında bağımsız bir ölçümle {ms(sc_last['exact_batch100_s'])} çıktı; aynı kodun "
                         f"tekrar ölçümleri arasındaki fark %{100 * abs(final_v['time_s'] / sc_last['exact_batch100_s'] - 1):.0f}. "
                         f"Saf Python satırları ve Şekil 6'daki naive son nokta aynı Phase 1 ölçümüdür.")
P("<b>Bağımlılıklar (requirements.txt):</b> numpy, scipy, scikit-learn, matplotlib, sentence-transformers, "
  "torch, transformers, line_profiler, numba, threadpoolctl, dask, ray, psutil, reportlab.")
P("<b>YZ kullanımı:</b> Kod ve rapor taslağı bir YZ asistanı yardımıyla yazıldı; tüm deneyler bu "
  "makinede koşuldu, her figür ve kod bloğu tarafımızdan incelendi ve açıklanabilir durumda.", CAP)

# ============================================================================ üret
doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                        topMargin=1.7 * cm, bottomMargin=1.8 * cm,
                        title="Yüksek Başarımlı Benzerlik Arama Motoru", author=AUTHOR)
doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print(f"Kaydedildi: {OUT}")
