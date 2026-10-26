"""Build the data files for the GDELT explorer artifact from the combined daily file.

    python code/export_gdelt_explorer.py          # writes <gdelt dir>/explorer/

Files (day index 0 = 2003-11-01, 7,213 days; every array is little-endian and base64-encoded inside JSON):
  meta.json          countries, the 20 root categories, every code's official CAMEO name
  root_<ISO3>.json   t = events per day (all codes, incl. malformed) ; r = 20 x days events per day per root category; w = bytes per value
  sub_<ISO3>.json    sparse events per day per full CAMEO code: codes, off (row offsets per code), d (day index, uint16), c (events, w bytes)
"""
import base64
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gdelt_bq as g  # noqa: E402
from direction_baseline import COUNTRY_NAMES  # noqa: E402

START = pd.Timestamp("2003-11-01")
END = pd.Timestamp("2023-07-31")
NDAYS = (END - START).days + 1
SRC = g.DEFAULT_DIR / "pulls" / "events_daily_2003-11-01_2023-07-31.parquet"
OUT = g.DEFAULT_DIR / "explorer"
ROOTS = [f"{i:02d}" for i in range(1, 21)]


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    names = json.load(open(Path(__file__).resolve().parent / "cameo_codes.json"))
    df = pd.read_parquet(SRC, columns=["date", "country", "event_code", "n_events"])
    df["day"] = (df["date"] - START).dt.days
    fips2iso = {g.to_fips(c): c for c in g.WVS_COMMON_COUNTRIES}
    meta = {"start": str(START.date()), "ndays": NDAYS,
            "countries": [{"iso3": c, "fips": g.to_fips(c), "name": COUNTRY_NAMES[c]} for c in sorted(g.WVS_COMMON_COUNTRIES, key=lambda c: COUNTRY_NAMES[c])],
            "roots": [{"code": r, "name": names[r].capitalize()} for r in ROOTS],
            "codes": {k: v for k, v in names.items() if len(k) >= 3}}
    (OUT / "meta.json").write_text(json.dumps(meta, separators=(",", ":")))
    tot_bytes = {"root": 0, "sub": 0}
    for fips, iso in sorted(fips2iso.items(), key=lambda kv: kv[1]):
        d = df[df["country"] == fips]
        total = np.zeros(NDAYS, dtype=np.uint32)
        np.add.at(total, d["day"].to_numpy(), d["n_events"].to_numpy())
        ok = d[d["event_code"].str.fullmatch(r"(0[1-9]|1[0-9]|20)\d{1,2}")]
        roots = np.zeros((20, NDAYS), dtype=np.uint32)
        np.add.at(roots, (ok["event_code"].str[:2].astype(int).to_numpy() - 1, ok["day"].to_numpy()), ok["n_events"].to_numpy())
        w = 2 if roots.max() < 65535 and total.max() < 65535 else 4
        rj = {"t": b64(total.astype(f"<u{w}")), "r": b64(roots.astype(f"<u{w}")), "w": w}
        (OUT / f"root_{iso}.json").write_text(json.dumps(rj, separators=(",", ":")))
        s = ok.sort_values(["event_code", "day"])
        codes = list(dict.fromkeys(s["event_code"]))
        counts = s.groupby("event_code", sort=True).size().reindex(codes).to_numpy()
        off = np.concatenate([[0], np.cumsum(counts)]).tolist()
        cw = 2 if s["n_events"].max() < 65535 else 4
        sj = {"codes": codes, "off": off, "d": b64(s["day"].to_numpy().astype("<u2")), "c": b64(s["n_events"].to_numpy().astype(f"<u{cw}")), "w": cw}
        (OUT / f"sub_{iso}.json").write_text(json.dumps(sj, separators=(",", ":")))
        tot_bytes["root"] += (OUT / f"root_{iso}.json").stat().st_size
        tot_bytes["sub"] += (OUT / f"sub_{iso}.json").stat().st_size
        print(iso, f"root {(OUT / f'root_{iso}.json').stat().st_size / 1e6:.2f} MB (w={w})  sub {(OUT / f'sub_{iso}.json').stat().st_size / 1e6:.2f} MB (w={cw}, {len(s):,} rows)")
    print({k: f"{v / 1e6:.1f} MB" for k, v in tot_bytes.items()}, "meta", f"{(OUT / 'meta.json').stat().st_size / 1e3:.0f} KB")


if __name__ == "__main__":
    main()
