#!/usr/bin/env python3
"""
Analyze .bib files from WoS queries for state-of-the-art review.
Deduplicates, ranks by citations, clusters by theme, identifies key papers.
"""

import re
import os
from collections import Counter, defaultdict
from pathlib import Path

DOCS_DIR = Path("/home/franciscoparrao/proyectos/spectral-indices-discovery/docs")
OUTPUT_DIR = Path("/home/franciscoparrao/proyectos/spectral-indices-discovery/docs")

QUERY_THEMES = {
    "Q1": "Alteración hidrotermal + RS",
    "Q2": "Índices espectrales minerales",
    "Q3": "ASTER alteración",
    "Q4": "Sentinel-2 geología",
    "Q5": "Symbolic regression + RS",
    "Q6": "SR descubrimiento fórmulas",
    "Q7": "ML alteración hidrotermal",
    "Q8": "Zonas estudio Chile",
    "Q9": "Sentinel-2 SWIR minerales",
    "Q10": "ASTER vs Sentinel-2",
    "Q11": "Espectroscopía minerales",
    "Q12": "GP/SR índices espectrales",
    "Q13": "Andes alteración + RS",
    "Q14": "Optimización índices",
}


def parse_bib_robust(filepath):
    """Parse WoS .bib file robustly."""
    entries = []

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    # Split by entry starts
    raw_entries = re.split(r'\n(?=@\w+\{)', content)

    for raw in raw_entries:
        raw = raw.strip()
        if not raw.startswith("@"):
            continue

        entry = {}

        # Entry type and key
        m = re.match(r'@(\w+)\{\s*([^,]+)', raw)
        if m:
            entry["_type"] = m.group(1).lower()
            entry["_key"] = m.group(2).strip()

        # Extract all fields: Field = {value} potentially multi-line
        # WoS uses Field-Name = {value},
        field_pattern = re.compile(
            r'(\w[\w-]*)\s*=\s*\{',
        )

        pos = 0
        for m in field_pattern.finditer(raw):
            field_name = m.group(1).lower().replace("-", "_")
            start = m.end()  # position after {

            # Find matching closing brace, handling nested braces
            depth = 1
            i = start
            while i < len(raw) and depth > 0:
                if raw[i] == '{':
                    depth += 1
                elif raw[i] == '}':
                    depth -= 1
                i += 1

            if depth == 0:
                value = raw[start:i-1]
                # Clean up whitespace from multi-line values
                value = re.sub(r'\s+', ' ', value).strip()
                entry[field_name] = value

        if entry.get("_key"):
            entries.append(entry)

    return entries


def normalize_title(title):
    t = title.lower()
    t = re.sub(r"[^a-z0-9\s]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def main():
    # Parse all .bib files
    all_entries = {}
    for i in range(1, 15):
        fp = DOCS_DIR / f"Q{i}.bib"
        if fp.exists():
            entries = parse_bib_robust(fp)
            all_entries[f"Q{i}"] = entries
            # Quick sanity check
            years = [e.get("year", "?") for e in entries[:3]]
            cites = [e.get("times_cited", "?") for e in entries[:3]]
            print(f"Q{i}: {len(entries)} entries (sample years: {years}, cites: {cites})")

    # Deduplicate
    seen_doi = {}
    seen_title = {}
    unique = []
    duplicates = 0

    for qname, entries in all_entries.items():
        for entry in entries:
            title = entry.get("title", "")
            doi = entry.get("doi", "").lower().strip()
            norm_t = normalize_title(title)

            if doi and doi in seen_doi:
                seen_doi[doi]["_queries"].add(qname)
                duplicates += 1
                continue
            if norm_t and len(norm_t) > 10 and norm_t in seen_title:
                seen_title[norm_t]["_queries"].add(qname)
                duplicates += 1
                continue

            entry["_queries"] = {qname}
            if doi:
                seen_doi[doi] = entry
            if norm_t and len(norm_t) > 10:
                seen_title[norm_t] = entry
            unique.append(entry)

    print(f"\nTotal bruto: {sum(len(v) for v in all_entries.values())}")
    print(f"Duplicados: {duplicates}")
    print(f"Únicos: {len(unique)}")

    # Parse numeric fields
    for e in unique:
        try:
            e["_year"] = int(e.get("year", "0"))
        except:
            e["_year"] = 0
        try:
            e["_cites"] = int(e.get("times_cited", "0"))
        except:
            e["_cites"] = 0

    # Verify parsing
    years_sample = [e["_year"] for e in unique if e["_year"] > 0]
    cites_sample = [e["_cites"] for e in unique if e["_cites"] > 0]
    print(f"Entries with valid year: {len(years_sample)}")
    print(f"Entries with citations > 0: {len(cites_sample)}")
    if years_sample:
        print(f"Year range: {min(years_sample)}-{max(years_sample)}")
    if cites_sample:
        print(f"Citation range: {min(cites_sample)}-{max(cites_sample)}")

    # === BUILD REPORT ===
    R = []
    R.append("# Análisis Bibliométrico — Spectral Indices Discovery\n")
    R.append(f"**Fecha**: 2026-03-22  ")
    R.append(f"**Total entradas brutas**: {sum(len(v) for v in all_entries.values())}  ")
    R.append(f"**Entradas únicas**: {len(unique)}  ")
    R.append(f"**Duplicados eliminados**: {duplicates}\n")

    # --- Volume per query ---
    R.append("## Volumen por Query\n")
    R.append("| Query | Tema | Entradas |")
    R.append("|-------|------|----------|")
    for i in range(1, 15):
        qn = f"Q{i}"
        n = len(all_entries.get(qn, []))
        R.append(f"| {qn} | {QUERY_THEMES.get(qn, '')} | {n} |")

    # --- Year distribution ---
    R.append("\n## Distribución Temporal\n")
    year_counts = Counter(e["_year"] for e in unique if e["_year"] > 1990)

    periods = [
        ("2000-2005", range(2000, 2006)),
        ("2006-2010", range(2006, 2011)),
        ("2011-2015", range(2011, 2016)),
        ("2016-2020", range(2016, 2021)),
        ("2021-2026", range(2021, 2027)),
    ]
    R.append("| Período | Papers |")
    R.append("|---------|--------|")
    for label, yrs in periods:
        n = sum(year_counts.get(y, 0) for y in yrs)
        R.append(f"| {label} | {n} |")

    R.append("\nDetalle anual (2015-2026):\n")
    R.append("```")
    for y in range(2015, 2027):
        n = year_counts.get(y, 0)
        bar = "█" * (n // 10)
        R.append(f"{y}: {n:4d} {bar}")
    R.append("```")

    # --- TOP CITED PAPERS (the most important section) ---
    R.append("\n## Top 50 Papers más Citados\n")
    R.append("Papers ordenados por Times Cited en WoS.\n")

    by_cites = sorted(unique, key=lambda x: x["_cites"], reverse=True)

    R.append("| # | Citas | Año | Primer Autor | Título | Revista | Queries |")
    R.append("|---|-------|-----|--------------|--------|---------|---------|")
    for rank, e in enumerate(by_cites[:50], 1):
        author = e.get("author", "?").split(",")[0].split(" and ")[0].strip()
        title = e.get("title", "?")
        if len(title) > 80:
            title = title[:77] + "..."
        journal = e.get("journal", "?")
        if len(journal) > 35:
            journal = journal[:32] + "..."
        year = e.get("year", "?")
        cites = e["_cites"]
        queries = ", ".join(sorted(e["_queries"]))
        R.append(f"| {rank} | {cites} | {year} | {author} | {title} | {journal} | {queries} |")

    # --- Top cited per query ---
    R.append("\n## Top 10 por Query (más citados)\n")
    for i in range(1, 15):
        qn = f"Q{i}"
        theme = QUERY_THEMES.get(qn, "")
        q_papers = [e for e in unique if qn in e["_queries"]]
        q_papers.sort(key=lambda x: x["_cites"], reverse=True)

        R.append(f"\n### {qn}: {theme}\n")
        R.append("| Citas | Año | Primer Autor | Título | Revista |")
        R.append("|-------|-----|--------------|--------|---------|")
        for e in q_papers[:10]:
            author = e.get("author", "?").split(",")[0].split(" and ")[0].strip()
            title = e.get("title", "?")
            if len(title) > 90:
                title = title[:87] + "..."
            journal = e.get("journal", "?")
            if len(journal) > 35:
                journal = journal[:32] + "..."
            R.append(f"| {e['_cites']} | {e.get('year', '?')} | {author} | {title} | {journal} |")

    # --- Top journals ---
    R.append("\n## Top 25 Revistas\n")
    journal_counts = Counter()
    for e in unique:
        j = e.get("journal", "").strip()
        if j:
            journal_counts[j] += 1

    R.append("| Revista | Papers |")
    R.append("|---------|--------|")
    for j, c in journal_counts.most_common(25):
        R.append(f"| {j} | {c} |")

    # --- Author keywords ---
    R.append("\n## Top 40 Keywords (autor)\n")
    kw_counts = Counter()
    for e in unique:
        kw = e.get("keywords", "")
        for k in re.split(r";", kw):
            k = k.strip().lower()
            if k and len(k) > 2:
                kw_counts[k] += 1

    R.append("| Keyword | Frecuencia |")
    R.append("|---------|------------|")
    for k, c in kw_counts.most_common(40):
        R.append(f"| {k} | {c} |")

    # --- Keywords Plus ---
    R.append("\n## Top 30 Keywords Plus (WoS)\n")
    kwp_counts = Counter()
    for e in unique:
        kwp = e.get("keywords_plus", "")
        for k in re.split(r";", kwp):
            k = k.strip().lower()
            if k and len(k) > 2:
                kwp_counts[k] += 1

    R.append("| Keyword | Frecuencia |")
    R.append("|---------|------------|")
    for k, c in kwp_counts.most_common(30):
        R.append(f"| {k} | {c} |")

    # --- Research areas ---
    R.append("\n## Áreas de Investigación\n")
    area_counts = Counter()
    for e in unique:
        areas = e.get("research_areas", "")
        for a in re.split(r";", areas):
            a = a.strip()
            if a and len(a) > 3:
                area_counts[a] += 1

    R.append("| Área | Papers |")
    R.append("|------|--------|")
    for a, c in area_counts.most_common(20):
        R.append(f"| {a} | {c} |")

    # --- Multi-query papers ---
    R.append("\n## Papers en 3+ Queries (alta relevancia transversal)\n")
    multi = sorted(
        [e for e in unique if len(e["_queries"]) >= 3],
        key=lambda x: (len(x["_queries"]), x["_cites"]),
        reverse=True,
    )
    for e in multi[:60]:
        author = e.get("author", "?").split(",")[0].split(" and ")[0].strip()
        title = e.get("title", "Sin título")
        year = e.get("year", "?")
        cites = e["_cites"]
        queries = ", ".join(sorted(e["_queries"]))
        nq = len(e["_queries"])
        R.append(f"- **[{nq}Q, {cites}c]** {author} ({year}). _{title}_. [{queries}]")

    # --- Query overlap ---
    R.append("\n## Solapamiento entre Queries\n")
    overlap = Counter(len(e["_queries"]) for e in unique)
    for n, c in sorted(overlap.items()):
        R.append(f"- Papers en {n} query(s): {c}")

    # --- Document types ---
    R.append("\n## Tipos de Documento\n")
    type_counts = Counter(e.get("type", "?") for e in unique)
    for t, c in type_counts.most_common():
        R.append(f"- {t}: {c}")

    # --- Papers per query exclusive ---
    R.append("\n## Papers exclusivos por query\n")
    for i in range(1, 15):
        qn = f"Q{i}"
        exclusive = [e for e in unique if e["_queries"] == {qn}]
        R.append(f"- {qn} ({QUERY_THEMES.get(qn, '')}): {len(exclusive)} exclusivos")

    # --- Citation distribution ---
    R.append("\n## Distribución de Citas\n")
    cite_ranges = [
        ("0", lambda c: c == 0),
        ("1-10", lambda c: 1 <= c <= 10),
        ("11-50", lambda c: 11 <= c <= 50),
        ("51-100", lambda c: 51 <= c <= 100),
        ("101-200", lambda c: 101 <= c <= 200),
        ("201-500", lambda c: 201 <= c <= 500),
        (">500", lambda c: c > 500),
    ]
    R.append("| Rango de citas | Papers |")
    R.append("|----------------|--------|")
    for label, fn in cite_ranges:
        n = sum(1 for e in unique if fn(e["_cites"]))
        R.append(f"| {label} | {n} |")

    # Write report
    report_text = "\n".join(R)
    out_path = OUTPUT_DIR / "bibliometric_analysis.md"
    with open(out_path, "w") as f:
        f.write(report_text)
    print(f"\nReport: {out_path}")

    # Export deduplicated master .bib
    bib_path = OUTPUT_DIR / "master_deduplicated.bib"
    with open(bib_path, "w") as f:
        for e in unique:
            key = e.get("_key", "unknown")
            etype = e.get("_type", "article")
            f.write(f"@{etype}{{{key},\n")
            for k, v in e.items():
                if k.startswith("_"):
                    continue
                f.write(f"  {k} = {{{v}}},\n")
            f.write(f"  queries = {{{', '.join(sorted(e['_queries']))}}},\n")
            f.write("}\n\n")
    print(f"Master .bib: {bib_path}")


if __name__ == "__main__":
    main()
