#!/usr/bin/env bash
# Screenshots for the report (maps + UI). Needs the API running on :8000 and the UI served on :5500.
set -e
cd "$(dirname "$0")/.."
C="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
S=report/screenshots; mkdir -p $S
shot() { "$C" --headless=new --disable-gpu --hide-scrollbars --window-size=$3 --virtual-time-budget=${4:-12000} --screenshot="$S/$2" "$1" 2>/dev/null; }
shot "file://$PWD/results/maps/eda_train_00000003.html" eda_folium.png 1400,850
shot "file://$PWD/results/maps/student_test_00000056.html" map_full.png 1400,900
shot "file://$PWD/results/maps/student_test_00000056_zoom_clf_vs_hmm.html" map_zoom.png 1400,900
shot "http://localhost:5500/index.html?api=http://localhost:8000&trip=00000009" ui.png 1500,950 20000
