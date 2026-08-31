#!/bin/bash
# Download remaining S2 bands in parallel (2 at a time)
# Each band takes ~5-10 min with max-scenes=4

SURTGIS="$HOME/proyectos/surtgis/target/release/surtgis"
OUTDIR="data/land_cover"
BBOX="11.0,50.2,11.3,50.5"
DATETIME="2021-06-01/2021-09-01"

download_band() {
    local band=$1
    local asset=$(echo "$band" | tr '[:upper:]' '[:lower:]')
    local outfile="${OUTDIR}/thuringia_${band}.tif"

    if [ -f "$outfile" ] && [ -s "$outfile" ]; then
        echo "[SKIP] $band already exists"
        return 0
    fi

    echo "[START] $band ..."
    $SURTGIS stac composite \
        --catalog pc \
        --collection sentinel-2-l2a \
        --asset "$asset" \
        --bbox "$BBOX" \
        --datetime "$DATETIME" \
        --max-scenes 4 \
        --max-memory 4G \
        --compress \
        "$outfile" 2>&1 | tail -2

    if [ $? -eq 0 ] && [ -s "$outfile" ]; then
        echo "[DONE] $band ($(du -h "$outfile" | cut -f1))"
    else
        echo "[FAIL] $band"
    fi
}

# Bands to download (B02 and B03 already done)
BANDS="B04 B05 B06 B07 B08 B8A B11 B12"

# Run 2 at a time using GNU parallel or xargs
echo "Downloading 8 S2 bands (2 at a time)..."
echo "Start: $(date)"

export -f download_band
export SURTGIS OUTDIR BBOX DATETIME

echo "$BANDS" | tr ' ' '\n' | xargs -P 2 -I{} bash -c 'download_band {}'

echo ""
echo "Finished: $(date)"
echo "Files:"
ls -lh ${OUTDIR}/thuringia_*.tif 2>/dev/null
