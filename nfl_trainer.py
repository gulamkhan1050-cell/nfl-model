"""
nfl_trainer.py - build the NFL model (moneyline, spread, total) from free public data.

Run:   python nfl_trainer.py                     (2015 .. current season, downloads once, cached)
       python nfl_trainer.py --seasons 2019 2020 2021 2022 2023 2024 2025 2026
       python nfl_trainer.py --no-pbp            (skip play-by-play EPA, much faster, a bit weaker)
       python nfl_trainer.py --demo              (synthetic data, no internet, just to check the pipeline)
Then:  python build.py

Data:
  * nflverse games.csv   - every game since 1999 with scores, rest days, closing spread / total / moneyline,
                           starting QBs, roof, surface, temperature, wind, division flag, playoffs
  * nflverse play-by-play - per-season EPA (expected points added) for pass / rush offence and defence
Both are cached in cache_nfl/ so re-runs are quick.

Requires: pip install pandas numpy requests
"""
import argparse, io, json, math, os, random, sys, time
from collections import defaultdict, deque
from datetime import datetime, timezone
import numpy as np

CACHE = "cache_nfl"; OUT = "model.json"
GAMES_URL = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"
PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.csv.gz"
BASE, HOME_ELO, K_ELO = 1500.0, 48.0, 20.0          # Elo settings (FiveThirtyEight-style)
FORM_N, EMA = 5, 0.12                                 # form window and EPA smoothing rate
TEAM_FIX = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA", "WSH": "WAS", "JAC": "JAX"}
TEAM_NAMES = {"ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens", "BUF": "Buffalo Bills", "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
              "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns", "DAL": "Dallas Cowboys", "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
              "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars", "KC": "Kansas City Chiefs", "LA": "Los Angeles Rams", "LAC": "Los Angeles Chargers",
              "LV": "Las Vegas Raiders", "MIA": "Miami Dolphins", "MIN": "Minnesota Vikings", "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
              "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers", "SEA": "Seattle Seahawks", "SF": "San Francisco 49ers", "TB": "Tampa Bay Buccaneers",
              "TEN": "Tennessee Titans", "WAS": "Washington Commanders"}


# stadium: lat, lon, UTC offset (standard time), elevation m.   Older homes for teams that moved.
STADIUM = {"ARI": (33.5276, -112.2626, -7, 330), "ATL": (33.7554, -84.4009, -5, 300), "BAL": (39.2780, -76.6227, -5, 10), "BUF": (42.7738, -78.7870, -5, 180), "CAR": (35.2258, -80.8528, -5, 220),
           "CHI": (41.8623, -87.6167, -6, 180), "CIN": (39.0955, -84.5161, -5, 150), "CLE": (41.5061, -81.6995, -5, 180), "DAL": (32.7473, -97.0945, -6, 170), "DEN": (39.7439, -105.0201, -7, 1610),
           "DET": (42.3400, -83.0456, -5, 180), "GB": (44.5013, -88.0622, -6, 210), "HOU": (29.6847, -95.4107, -6, 15), "IND": (39.7601, -86.1639, -5, 220), "JAX": (30.3240, -81.6373, -5, 5),
           "KC": (39.0489, -94.4839, -6, 270), "LA": (33.9535, -118.3390, -8, 30), "LAC": (33.9535, -118.3390, -8, 30), "LV": (36.0909, -115.1833, -8, 620), "MIA": (25.9580, -80.2389, -5, 3),
           "MIN": (44.9736, -93.2575, -6, 250), "NE": (42.0909, -71.2643, -5, 90), "NO": (29.9511, -90.0812, -6, 3), "NYG": (40.8135, -74.0745, -5, 3), "NYJ": (40.8135, -74.0745, -5, 3),
           "PHI": (39.9008, -75.1675, -5, 10), "PIT": (40.4468, -80.0158, -5, 230), "SEA": (47.5952, -122.3316, -8, 5), "SF": (37.4030, -121.9700, -8, 5), "TB": (27.9759, -82.5033, -5, 10),
           "TEN": (36.1665, -86.7713, -6, 130), "WAS": (38.9076, -76.8645, -5, 50)}
OLD_HOME = {("LA", 2016): (38.6328, -90.1885, -6, 140), ("LA", 2020): (34.0141, -118.2879, -8, 50), ("LAC", 2017): (32.7831, -117.1196, -8, 60), ("LAC", 2020): (33.8644, -118.2611, -8, 20), ("LV", 2020): (37.7516, -122.2005, -8, 5)}
def stadium(team, season):
    """coordinates of a team's home in a given season"""
    for (t, until), v in sorted(OLD_HOME.items(), key=lambda kv: kv[0][1]):
        if t == team and season < until: return v
    return STADIUM.get(team, (39.8, -98.5, -6, 300))
def haversine(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1])); h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(h))
INJ_URL = "https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_{season}.csv"
SNAP_URL = "https://github.com/nflverse/nflverse-data/releases/download/snap_counts/snap_counts_{season}.csv"
DEPTH_URL = "https://github.com/nflverse/nflverse-data/releases/download/depth_charts/depth_charts_{season}.csv"
POS_W = {"QB": 0.0, "T": 1.0, "OT": 1.0, "G": 0.9, "OG": 0.9, "C": 0.9, "OL": 0.9, "WR": 1.0, "TE": 0.6, "RB": 0.6, "FB": 0.2, "HB": 0.6,
         "DE": 1.0, "DT": 0.8, "NT": 0.7, "DL": 0.9, "EDGE": 1.0, "OLB": 0.9, "ILB": 0.7, "MLB": 0.7, "LB": 0.7, "CB": 1.0, "S": 0.7, "SS": 0.7, "FS": 0.7, "DB": 0.7, "K": 0.3, "P": 0.2, "LS": 0.2}
OFF_POS = {"T", "OT", "G", "OG", "C", "OL", "WR", "TE", "RB", "FB", "HB", "QB"}
STATUS_W = {"out": 1.0, "doubtful": 0.85, "questionable": 0.35, "injured reserve": 1.0, "ir": 1.0, "pup": 1.0, "suspended": 1.0}
def inj_score(rows, snaps=None, depth=None):
    """(offence, defence) injury load. A player's weight is his position value x status, scaled by how much he
    actually plays: last week's snap share if known, else his depth-chart rank, else a neutral 0.55 starter guess."""
    off = deff = 0.0
    for r in rows:
        pos, status, name = (list(r) + [None])[:3]
        pos = str(pos).upper(); base = POS_W.get(pos, 0.5) * STATUS_W.get(str(status).strip().lower(), 0.0)
        if base == 0: continue
        is_off = pos in OFF_POS; use = None
        if snaps and name:
            sn = snaps.get(name) or snaps.get(str(name).strip())
            if sn: use = (sn[0] if is_off else sn[1]) / 100.0
        if use is None and depth and name:
            dp = depth.get(name) or depth.get(str(name).strip())
            if dp: use = {1: 0.85, 2: 0.35, 3: 0.12}.get(dp[1], 0.05)
        if use is None: use = 0.55
        w = base * min(use, 1.0)
        if is_off: off += w
        else: deff += w
    return off, deff

def load_injuries(seasons, snaps=None, depth=None):
    """(season, week, team) -> (off_load, def_load) from nflverse injury reports, weighted by snap share, cached."""
    import pandas as pd
    os.makedirs(CACHE, exist_ok=True); out = {}; snaps = snaps or {}; depth = depth or {}
    for s in seasons:
        p = os.path.join(CACHE, f"inj_w_{s}.json")
        if os.path.exists(p) and (s < datetime.now().year or time.time() - os.path.getmtime(p) < 6 * 3600):
            out.update({tuple(json.loads(k)): v for k, v in json.load(open(p)).items()}); continue
        try: raw = fetch(INJ_URL.format(season=s))
        except SystemExit: print(f"  no injury report for {s}, skipping"); continue
        df = pd.read_csv(io.StringIO(raw), low_memory=False); df = df[df.report_status.notna()]
        nm = "full_name" if "full_name" in df.columns else ("player_name" if "player_name" in df.columns else None)
        d = {}
        for (wk, team), grp in df.groupby(["week", "team"]):
            t = fix(team); wk = int(wk)
            sn = snaps.get((int(s), wk - 1, t)) or snaps.get((int(s), wk, t)) or {}      # last week's usage: what he would have played
            dp = depth.get((int(s), wk, t)) or {}
            rows = zip(grp.position.fillna(""), grp.report_status.fillna(""), grp[nm].fillna("") if nm else [""] * len(grp))
            d[json.dumps([int(s), wk, t])] = list(inj_score(rows, sn, dp))
        json.dump(d, open(p, "w")); out.update({tuple(json.loads(k)): v for k, v in d.items()}); print(f"  injuries {s}: {len(d)} team-weeks")
    return out


# ================================================================== data
def fix(t): t = str(t).upper(); return TEAM_FIX.get(t, t)
def fetch(url, binary=False):
    import requests
    for i in range(4):
        try:
            r = requests.get(url, timeout=120, headers={"User-Agent": "nfl-predictor/1.0"}); r.raise_for_status(); return r.content if binary else r.text
        except Exception as e:
            print(f"  retry {i+1} {url.split('/')[-1]}: {e}"); time.sleep(3 * (i + 1))
    raise SystemExit(f"could not download {url}")

def load_games(seasons):
    import pandas as pd
    os.makedirs(CACHE, exist_ok=True); p = os.path.join(CACHE, "games.csv")
    if not os.path.exists(p) or time.time() - os.path.getmtime(p) > 6 * 3600:
        print("downloading games.csv ..."); open(p, "w", encoding="utf-8").write(fetch(GAMES_URL))
    df = pd.read_csv(p, low_memory=False); df = df[df.season.isin(seasons)].copy()
    df = df[df.home_score.notna() & df.away_score.notna()]
    out = []
    for r in df.itertuples(index=False):
        gt = str(r.gametime) if isinstance(r.gametime, str) else "13:00"
        try: ts = int(datetime.strptime(f"{r.gameday} {gt}", "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc).timestamp()) + 5 * 3600
        except Exception: ts = int(datetime.strptime(str(r.gameday), "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()) + 18 * 3600
        f = lambda v: None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)
        s = lambda v: "" if v is None or (isinstance(v, float) and math.isnan(v)) else str(v)
        out.append({"id": str(r.game_id), "season": int(r.season), "week": int(r.week), "type": str(r.game_type), "ts": ts, "date": str(r.gameday),
                    "home": fix(r.home_team), "away": fix(r.away_team), "hs": int(r.home_score), "as": int(r.away_score),
                    "neutral": s(getattr(r, "location", "Home")).lower() == "neutral", "home_rest": f(r.home_rest) or 7, "away_rest": f(r.away_rest) or 7,
                    "spread": f(r.spread_line), "total_line": f(r.total_line), "home_ml": f(r.home_moneyline), "away_ml": f(r.away_moneyline),
                    "div": bool(r.div_game), "roof": s(r.roof).lower(), "surface": s(r.surface).lower(), "temp": f(r.temp), "wind": f(r.wind),
                    "home_qb": s(r.home_qb_id) or ("QB-" + fix(r.home_team)), "away_qb": s(r.away_qb_id) or ("QB-" + fix(r.away_team)),
                    "home_qb_name": s(r.home_qb_name), "away_qb_name": s(r.away_qb_name), "home_coach": s(r.home_coach), "away_coach": s(r.away_coach)})
    out.sort(key=lambda g: (g["ts"], g["id"])); return out

def load_snaps(seasons):
    """(season, week, team) -> {player_name: {off, def}} snap share. Lets us weight an injury by how much that player actually plays."""
    import pandas as pd
    os.makedirs(CACHE, exist_ok=True); out = {}
    for s in seasons:
        p = os.path.join(CACHE, f"snaps_{s}.json")
        if os.path.exists(p) and (s < datetime.now().year or time.time() - os.path.getmtime(p) < 6 * 3600):
            out.update({tuple(json.loads(k)): v for k, v in json.load(open(p)).items()}); continue
        try: raw = fetch(SNAP_URL.format(season=s))
        except SystemExit: print(f"  no snap counts for {s}, skipping"); continue
        df = pd.read_csv(io.StringIO(raw), low_memory=False)
        need = {"season", "week", "team", "player", "offense_pct", "defense_pct"}
        if not need.issubset(df.columns): print(f"  snap counts {s}: unexpected columns, skipping"); continue
        d = {}
        for (wk, team), grp in df.groupby(["week", "team"]):
            d[json.dumps([int(s), int(wk), fix(team)])] = {str(r.player): [float(r.offense_pct or 0), float(r.defense_pct or 0)] for r in grp.itertuples(index=False)}
        json.dump(d, open(p, "w")); out.update({tuple(json.loads(k)): v for k, v in d.items()}); print(f"  snap counts {s}: {len(d)} team-weeks")
    return out


def load_depth(seasons):
    """(season, week, team) -> {player_name: (position, depth rank)} from official depth charts."""
    import pandas as pd
    os.makedirs(CACHE, exist_ok=True); out = {}
    for s in seasons:
        p = os.path.join(CACHE, f"depth_{s}.json")
        if os.path.exists(p) and (s < datetime.now().year or time.time() - os.path.getmtime(p) < 6 * 3600):
            out.update({tuple(json.loads(k)): v for k, v in json.load(open(p)).items()}); continue
        try: raw = fetch(DEPTH_URL.format(season=s))
        except SystemExit: print(f"  no depth charts for {s}, skipping"); continue
        df = pd.read_csv(io.StringIO(raw), low_memory=False)
        nm = "football_name" if "football_name" in df.columns else ("full_name" if "full_name" in df.columns else None)
        pos = "depth_position" if "depth_position" in df.columns else ("position" if "position" in df.columns else None)
        rk = "depth_team" if "depth_team" in df.columns else None
        tm = "club_code" if "club_code" in df.columns else ("team" if "team" in df.columns else None)
        if not all([nm, pos, rk, tm]) or "week" not in df.columns: print(f"  depth charts {s}: unexpected columns, skipping"); continue
        d = {}
        for (wk, team), grp in df.groupby(["week", tm]):
            try: wk = int(wk)
            except Exception: continue
            d[json.dumps([int(s), wk, fix(team)])] = {str(r[1]): [str(r[2]), int(r[3]) if str(r[3]).isdigit() else 9] for r in grp[[nm, pos, rk]].itertuples()}
        json.dump(d, open(p, "w")); out.update({tuple(json.loads(k)): v for k, v in d.items()}); print(f"  depth charts {s}: {len(d)} team-weeks")
    return out


def load_pbp(seasons):
    """Per team-game EPA/play splits from nflverse play-by-play. Cached as small json per season."""
    import pandas as pd
    os.makedirs(CACHE, exist_ok=True); agg = {}
    for s in seasons:
        p = os.path.join(CACHE, f"epa_{s}.json")
        if os.path.exists(p) and (s < datetime.now().year or time.time() - os.path.getmtime(p) < 6 * 3600):
            agg.update(json.load(open(p))); continue
        print(f"downloading play-by-play {s} (~50-70 MB) ..."); 
        try: raw = fetch(PBP_URL.format(season=s), binary=True)
        except SystemExit: print(f"  no play-by-play for {s} yet, skipping"); continue
        df = pd.read_csv(io.BytesIO(raw), compression="gzip", usecols=["game_id", "posteam", "defteam", "epa", "pass", "rush", "success", "play_type", "yards_gained"], low_memory=False)
        df = df[df.posteam.notna() & df.epa.notna() & df.play_type.isin(["pass", "run"])]
        d = {}
        for (gid, team), grp in df.groupby(["game_id", "posteam"]):
            pas = grp[grp["pass"] == 1]; rus = grp[grp["rush"] == 1]
            d.setdefault(str(gid), {})[fix(team)] = {"off": float(grp.epa.mean()), "off_pass": float(pas.epa.mean()) if len(pas) else 0.0, "off_rush": float(rus.epa.mean()) if len(rus) else 0.0,
                                                     "succ": float(grp.success.mean()), "plays": int(len(grp)), "pass_rate": float(len(pas) / max(len(grp), 1))}
        json.dump(d, open(p, "w")); agg.update(d); print(f"  {s}: {len(d)} games")
    return agg

def demo(seasons):
    random.seed(7); np.random.seed(7); teams = sorted(TEAM_NAMES); strength = {t: random.gauss(0, 5) for t in teams}; qbs = {t: (f"qb{t}", random.gauss(0, 2.5)) for t in teams}
    games, epa = [], {}; gid = 0
    for s in seasons:
        for t in teams:
            strength[t] += random.gauss(0, 1.5)
        for w in range(1, 19):
            ts_ = teams[:]; random.shuffle(ts_)
            for i in range(0, 32, 2):
                h, a = ts_[i], ts_[i + 1]; dome = h in ("ATL", "DET", "NO", "MIN", "LV", "LA", "ARI", "IND", "HOU", "DAL")
                mu = 2.4 + strength[h] - strength[a] + qbs[h][1] - qbs[a][1]; tot = 44 + random.gauss(0, 3)
                m = mu + random.gauss(0, 13); t_ = max(20, tot + random.gauss(0, 10)); hs = int(round((t_ + m) / 2)); as_ = int(round((t_ - m) / 2))
                if hs == as_: hs += 3
                ts = int(datetime(s, 9, 7, 18, tzinfo=timezone.utc).timestamp()) + (w - 1) * 7 * 86400 + (i // 2) * 600; gid += 1
                games.append({"id": f"{s}_{w:02d}_{a}_{h}", "season": s, "week": w, "type": "REG", "ts": ts, "date": datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d"), "home": h, "away": a, "hs": hs, "as": as_,
                              "neutral": False, "home_rest": 7 if random.random() > 0.1 else 13, "away_rest": 7 if random.random() > 0.1 else 4,
                              "spread": round(mu + random.gauss(0, 1.5), 1) if random.random() < 0.95 else None, "total_line": round(tot + random.gauss(0, 1), 1), "home_ml": None, "away_ml": None,
                              "div": random.random() < 0.35, "roof": "dome" if dome else "outdoors", "surface": "grass", "temp": None if dome else 60 + random.gauss(0, 15), "wind": None if dome else abs(random.gauss(7, 5)),
                              "home_qb": qbs[h][0], "away_qb": qbs[a][0], "home_qb_name": qbs[h][0], "away_qb_name": qbs[a][0], "home_coach": "", "away_coach": ""})
                g = games[-1]; g["home_ml"] = -110 - mu * 25 if mu > 0 else 100 - mu * 25; g["away_ml"] = 100 + mu * 25 if mu > 0 else -110 + mu * 25
                epa[g["id"]] = {h: {"off": (m / 60) + random.gauss(0, 0.1), "off_pass": (m / 50) + random.gauss(0, .12), "off_rush": random.gauss(0, .1), "succ": 0.45 + m / 300, "plays": 62, "pass_rate": 0.58},
                                a: {"off": (-m / 60) + random.gauss(0, 0.1), "off_pass": (-m / 50) + random.gauss(0, .12), "off_rush": random.gauss(0, .1), "succ": 0.45 - m / 300, "plays": 62, "pass_rate": 0.58}}
    games.sort(key=lambda g: g["ts"]); return games, epa
def demo_inj(games):
    random.seed(3); return {(g["season"], g["week"], t): [round(random.expovariate(1 / 1.2), 2), round(random.expovariate(1 / 1.2), 2)] for g in games for t in (g["home"], g["away"])}


# ================================================================== state
def ml_prob(home_ml, away_ml):
    if home_ml is None or away_ml is None: return None
    f = lambda o: 100 / (o + 100) if o > 0 else -o / (-o + 100)
    qh, qa = f(home_ml), f(away_ml); return qh / (qh + qa)

class State:
    def __init__(self, epa, inj=None):
        self.epa = epa or {}; self.inj = inj or {}; self.coach_n = defaultdict(int); self.coach_of = {}; self.elo = defaultdict(lambda: BASE); self.season = {}
        self.off = defaultdict(float); self.deff = defaultdict(float); self.off_pass = defaultdict(float); self.def_pass = defaultdict(float); self.off_rush = defaultdict(float); self.def_rush = defaultdict(float)
        self.succ = defaultdict(float); self.succ_allowed = defaultdict(float); self.pass_rate = defaultdict(lambda: 0.58); self.epa_n = defaultdict(int)
        self.margin = defaultdict(float); self.pts_for = defaultdict(lambda: 22.0); self.pts_ag = defaultdict(lambda: 22.0); self.form = defaultdict(lambda: deque(maxlen=FORM_N))
        self.qb = {}; self.qb_name = {}; self.qb_team = {}; self.last_qb = {}; self.last_ts = {}; self.games = defaultdict(int); self.coach = {}
        self.lg_total = deque(maxlen=400); self.lg_total.append(44.0)

    def new_season(self, t, s):
        if self.season.get(t) != s:
            if t in self.season:                      # regress everything a third of the way to the mean between seasons
                self.elo[t] = BASE + (self.elo[t] - BASE) * 2 / 3
                for d in (self.off, self.deff, self.off_pass, self.def_pass, self.off_rush, self.def_rush, self.succ, self.succ_allowed, self.margin): d[t] *= 0.6
                self.pts_for[t] = 22 + (self.pts_for[t] - 22) * 0.6; self.pts_ag[t] = 22 + (self.pts_ag[t] - 22) * 0.6
                self.form[t].clear()
            self.season[t] = s
    def qb_rating(self, q):
        r, n = self.qb.get(q, (0.0, 0)); return r * n / (n + 8)          # shrunk toward 0 (league-average starter) until ~8 starts
    def features(self, g):
        h, a = g["home"], g["away"]
        for t in (h, a): self.new_season(t, g["season"])
        elo_d = self.elo[h] - self.elo[a] + (0 if g["neutral"] else HOME_ELO)
        qb_h, qb_a = self.qb_rating(g["home_qb"]), self.qb_rating(g["away_qb"])
        qb_new_h = 1.0 if (self.last_qb.get(h) and self.last_qb[h] != g["home_qb"]) else 0.0
        qb_new_a = 1.0 if (self.last_qb.get(a) and self.last_qb[a] != g["away_qb"]) else 0.0
        dome = 1.0 if g["roof"] in ("dome", "closed") else 0.0
        temp = 68.0 if dome or g["temp"] is None else g["temp"]; wind = 0.0 if dome or g["wind"] is None else g["wind"]
        fh = float(np.mean(self.form[h])) if self.form[h] else 0.0; fa = float(np.mean(self.form[a])) if self.form[a] else 0.0
        ih = self.inj.get((g["season"], g["week"], h), (0.0, 0.0)); ia = self.inj.get((g["season"], g["week"], a), (0.0, 0.0))
        sh_, sa_ = stadium(h, g["season"]), stadium(a, g["season"]); venue = sh_ if not g["neutral"] else None
        travel = haversine(sa_, venue) / 1000 if venue else 0.0; tz = (sa_[2] - venue[2]) if venue else 0.0
        kick = datetime.fromtimestamp(g["ts"], timezone.utc).hour + (venue[2] if venue else -5); kick = kick % 24
        body = 1.0 if (venue and tz <= -2 and kick <= 13) else 0.0                   # west-coast team kicking off at 10am body time
        alt = 1.0 if (venue and venue[3] >= 1500) else 0.0
        nch = 1.0 if (g["home_coach"] and self.coach_n[(h, g["home_coach"])] < 6) else 0.0; nca = 1.0 if (g["away_coach"] and self.coach_n[(a, g["away_coach"])] < 6) else 0.0
        side = [elo_d / 100, self.off[h] - self.off[a], self.deff[a] - self.deff[h], self.off_pass[h] - self.off_pass[a], self.def_pass[a] - self.def_pass[h],
                self.off_rush[h] - self.off_rush[a], self.def_rush[a] - self.def_rush[h], self.succ[h] - self.succ[a], self.succ_allowed[a] - self.succ_allowed[h],
                self.margin[h] - self.margin[a], fh - fa, qb_h - qb_a, qb_new_h - qb_new_a,
                (min(g["home_rest"], 14) - min(g["away_rest"], 14)) / 7, 1.0 if g["home_rest"] <= 5 else 0.0, 1.0 if g["away_rest"] <= 5 else 0.0,
                1.0 if g["div"] else 0.0, 1.0 if g["type"] != "REG" else 0.0, 1.0 if g["neutral"] else 0.0, 1.0 if g["week"] <= 3 else 0.0,
                ia[0] - ih[0], ia[1] - ih[1], travel, tz / 3, body, alt, nca - nch]
        tot = [self.off[h] + self.off[a], self.deff[h] + self.deff[a], self.off_pass[h] + self.off_pass[a], self.succ[h] + self.succ[a], self.pass_rate[h] + self.pass_rate[a] - 1.16,
               self.pts_for[h] + self.pts_for[a] - 44, self.pts_ag[h] + self.pts_ag[a] - 44, abs(elo_d) / 100, abs(qb_h) + abs(qb_a), qb_h + qb_a,
               dome, (temp - 60) / 20, wind / 10, (temp - 60) / 20 * wind / 10, 1.0 if g["surface"] in ("grass",) else 0.0, 1.0 if g["div"] else 0.0, 1.0 if g["type"] != "REG" else 0.0,
               1.0 if g["week"] <= 3 else 0.0, 1.0 if g["week"] >= 16 else 0.0, float(np.mean(self.lg_total)) - 44,
               ih[0] + ia[0], ih[1] + ia[1], travel, alt]
        return side, tot
    def update(self, g):
        h, a, m = g["home"], g["away"], g["hs"] - g["as"]
        eh = 1 / (1 + 10 ** (-(self.elo[h] - self.elo[a] + (0 if g["neutral"] else HOME_ELO)) / 400)); sh = 1.0 if m > 0 else 0.0 if m < 0 else 0.5
        mult = math.log(abs(m) + 1) * 2.2 / (((self.elo[h] - self.elo[a]) if m > 0 else (self.elo[a] - self.elo[h])) * 0.001 + 2.2)
        d = K_ELO * mult * (sh - eh); self.elo[h] += d; self.elo[a] -= d
        for t, pf, pa, mg in ((h, g["hs"], g["as"], m), (a, g["as"], g["hs"], -m)):
            self.pts_for[t] += EMA * (pf - self.pts_for[t]); self.pts_ag[t] += EMA * (pa - self.pts_ag[t]); self.margin[t] += EMA * (mg - self.margin[t]); self.form[t].append(mg)
            self.last_ts[t] = g["ts"]; self.games[t] += 1
        self.lg_total.append(g["hs"] + g["as"])
        e = self.epa.get(g["id"], {})
        for t, o in ((h, a), (a, h)):
            x = e.get(t); y = e.get(o)
            if x:
                self.off[t] += EMA * (x["off"] - self.off[t]); self.off_pass[t] += EMA * (x["off_pass"] - self.off_pass[t]); self.off_rush[t] += EMA * (x["off_rush"] - self.off_rush[t])
                self.succ[t] += EMA * (x["succ"] - 0.45 - self.succ[t]); self.pass_rate[t] += EMA * (x["pass_rate"] - self.pass_rate[t]); self.epa_n[t] += 1
            if y:
                self.deff[t] += EMA * (y["off"] - self.deff[t]); self.def_pass[t] += EMA * (y["off_pass"] - self.def_pass[t]); self.def_rush[t] += EMA * (y["off_rush"] - self.def_rush[t])
                self.succ_allowed[t] += EMA * (y["succ"] - 0.45 - self.succ_allowed[t])
        for t, q, name, mg in ((h, g["home_qb"], g["home_qb_name"], m), (a, g["away_qb"], g["away_qb_name"], -m)):
            x = e.get(t); perf = (x["off"] * 60 if x else mg * 0.5)             # a QB's game value: offensive EPA scaled to points, else half the margin
            adj = perf - (self.deff[a if t == h else h] * 60 if x else 0)         # credit for facing a good defence
            r, n = self.qb.get(q, (0.0, 0)); self.qb[q] = (r + (adj - r) / min(n + 1, 12), n + 1); self.qb_name[q] = name or q; self.qb_team[q] = t; self.last_qb[t] = q
        self.coach[h] = g["home_coach"]; self.coach[a] = g["away_coach"]; self.coach_n[(h, g["home_coach"])] += 1; self.coach_n[(a, g["away_coach"])] += 1

WIN_FEATS = ["elo_diff", "off_epa_diff", "def_epa_diff", "pass_off_diff", "pass_def_diff", "rush_off_diff", "rush_def_diff", "success_diff", "success_allowed_diff",
             "margin_diff", "form5_diff", "qb_diff", "qb_change_diff", "rest_diff", "home_short_week", "away_short_week", "div_game", "playoff", "neutral", "early_season",
             "inj_offense_adv", "inj_defense_adv", "away_travel_1000km", "away_tz_shift", "away_body_clock", "altitude", "new_coach_adv"]
TOT_FEATS = ["off_epa_sum", "def_epa_sum", "pass_off_sum", "success_sum", "pass_rate_sum", "pts_for_sum", "pts_against_sum", "mismatch", "qb_quality", "qb_sum",
             "dome", "temp", "wind", "temp_x_wind", "grass", "div_game", "playoff", "early_season", "late_season", "league_scoring", "inj_offense_sum", "inj_defense_sum", "away_travel_1000km", "altitude"]


# ================================================================== models
def train_logistic(X, y, l2=1.0, iters=60):
    mu, sd = X.mean(0), X.std(0) + 1e-9; Xs = (X - mu) / sd; n, d = Xs.shape
    Xb = np.hstack([Xs, np.ones((n, 1))]); w = np.zeros(d + 1); R = np.eye(d + 1) * l2; R[d, d] = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(Xb @ w))); g = Xb.T @ (p - y) + R @ w; H = (Xb * (p * (1 - p))[:, None]).T @ Xb + R
        step = np.linalg.solve(H, g); w -= step
        if np.abs(step).max() < 1e-8: break
    return {"w": w[:d], "b": float(w[d]), "mu": mu, "sd": sd}
def predict_logistic(m, X): return 1 / (1 + np.exp(-(((X - m["mu"]) / m["sd"]) @ m["w"] + m["b"])))
def train_ridge(X, y, l2=3.0):
    mu, sd = X.mean(0), X.std(0) + 1e-9; Xs = (X - mu) / sd; n, d = Xs.shape
    Xb = np.hstack([Xs, np.ones((n, 1))]); R = np.eye(d + 1) * l2; R[d, d] = 0.0
    w = np.linalg.solve(Xb.T @ Xb + R, Xb.T @ y); return {"w": w[:d], "b": float(w[d]), "mu": mu, "sd": sd}
def predict_ridge(m, X): return ((X - m["mu"]) / m["sd"]) @ m["w"] + m["b"]
def choose_l2_logistic(X, y, grid=(1, 3, 10, 30, 100, 300, 1000, 3000), blocks=5, label="win"):
    """Pick the ridge strength by out-of-fold log loss, always training on the past. Hardcoding it
    lets every feature keep whatever weight the training seasons happened to hand it."""
    m = len(X); best = (grid[0], 1e9)
    for l2 in grid:
        tot = cnt = 0.0
        for bi in range(1, blocks):
            cut = m * bi // blocks; hi = m * (bi + 1) // blocks
            mod = train_logistic(X[:cut], y[:cut], l2=l2)
            p = np.clip(predict_logistic(mod, X[cut:hi]), 1e-9, 1 - 1e-9); yy = y[cut:hi]
            tot += float(-np.sum(yy * np.log(p) + (1 - yy) * np.log(1 - p))); cnt += len(yy)
        ll = tot / max(cnt, 1)
        print(f"    l2={l2:<6} out-of-fold log loss {ll:.4f}")
        if ll < best[1]: best = (l2, ll)
    print(f"  chose l2={best[0]} for the {label} model")
    if best[0] == grid[-1]:
        print("  WARNING: largest value tried - these features carry little signal.")
    return best[0]


def choose_l2_ridge(X, y, grid=(1, 3, 10, 30, 100, 300, 1000, 3000), blocks=5, label="margin"):
    """Same, scored on out-of-fold mean absolute error - the number that decides which side of a
    spread or total the pick lands on."""
    m = len(X); best = (grid[0], 1e9)
    for l2 in grid:
        tot = cnt = 0.0
        for bi in range(1, blocks):
            cut = m * bi // blocks; hi = m * (bi + 1) // blocks
            mod = train_ridge(X[:cut], y[:cut], l2=l2)
            tot += float(np.abs(predict_ridge(mod, X[cut:hi]) - y[cut:hi]).sum()); cnt += hi - cut
        mae = tot / max(cnt, 1)
        print(f"    l2={l2:<6} out-of-fold MAE {mae:.4f}")
        if mae < best[1]: best = (l2, mae)
    print(f"  chose l2={best[0]} for the {label} model")
    return best[0]


def fit_temperature(X, y, l2, blocks=5):
    """p' = sigmoid(t * logit(p)), fitted only on games the model never trained on. A model that
    says 75% and wins 68% is wrong exactly where a Kelly stake is largest."""
    m = len(X); L = []; Y = []
    for bi in range(1, blocks):
        cut = m * bi // blocks; hi = m * (bi + 1) // blocks
        mod = train_logistic(X[:cut], y[:cut], l2=l2)
        p = np.clip(predict_logistic(mod, X[cut:hi]), 1e-6, 1 - 1e-6)
        L.append(np.log(p / (1 - p))); Y.append(y[cut:hi])
    if not L: return 1.0
    L = np.concatenate(L); Y = np.concatenate(Y); best = (1.0, 1e9)
    for t in np.arange(0.40, 1.41, 0.02):
        q = np.clip(1 / (1 + np.exp(-t * L)), 1e-9, 1 - 1e-9)
        ll = float(-np.mean(Y * np.log(q) + (1 - Y) * np.log(1 - q)))
        if ll < best[1]: best = (float(t), ll)
    print(f"  calibration temperature {best[0]:.2f} (1.00 = already calibrated, lower = overconfident)")
    return best[0]


def apply_temp(p, t):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return 1 / (1 + np.exp(-t * np.log(p / (1 - p))))


def oof(train_fn, pred_fn, X, y, k=5):
    """Out-of-fold predictions over contiguous (chronological) blocks, so the stack is fitted on unseen games."""
    n = len(y); out = np.zeros(n); edges = np.linspace(0, n, k + 1).astype(int)
    for i in range(k):
        te = np.zeros(n, bool); te[edges[i]:edges[i + 1]] = True
        m = train_fn(X[~te], y[~te]); out[te] = pred_fn(m, X[te])
    return out
def pack(m): return {"w": [round(float(v), 6) for v in m["w"]], "b": round(m["b"], 6), "mu": [round(float(v), 6) for v in m["mu"]], "sd": [round(float(max(v, 1e-6)), 6) for v in m["sd"]]}
def evaluate(name, p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6); acc = ((p > 0.5) == (y == 1)).mean(); ll = -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean(); brier = ((p - y) ** 2).mean()
    print(f"  {name:<30} accuracy {acc*100:5.1f}%   log-loss {ll:.4f}   brier {brier:.4f}"); return acc, ll, brier
def logit(p): p = np.clip(p, 0.02, 0.98); return np.log(p / (1 - p))
def ats(pred, line, margin):
    """share of games where picking the side of the line the prediction favours would have covered (pushes excluded)"""
    ok = (pred != line) & (margin != line); return float(((pred > line) == (margin > line))[ok].mean()) if ok.sum() else float("nan"), int(ok.sum())


# ================================================================== main
def main():
    now_year = datetime.now().year
    ap = argparse.ArgumentParser(); ap.add_argument("--seasons", nargs="+", type=int, default=list(range(2015, now_year + 1))); ap.add_argument("--demo", action="store_true")
    ap.add_argument("--no-pbp", action="store_true"); ap.add_argument("--out", default=OUT); ap.add_argument("--holdout", type=int, default=1, help="number of most recent seasons held out for testing")
    args = ap.parse_args()
    inj = {}; st_snaps = {}; st_depth = {}
    if args.demo: games, epa = demo(args.seasons); inj = demo_inj(games)
    else:
        games = load_games(args.seasons); epa = {} if args.no_pbp else load_pbp(sorted({g["season"] for g in games}))
        yrs = sorted({g["season"] for g in games})
        print("downloading snap counts ..."); snaps = load_snaps(yrs)
        print("downloading depth charts ..."); depth = load_depth(yrs)
        print("downloading injury reports ..."); inj = load_injuries(yrs, snaps, depth)
        st_snaps = {}                                                                   # latest snap share per team, for the app
        for (se, wk, t), d in snaps.items():
            cur = st_snaps.get(t)
            if not cur or (se, wk) > cur[0]: st_snaps[t] = ((se, wk), d)
        st_depth = {}
        for (se, wk, t), d in depth.items():
            cur = st_depth.get(t)
            if not cur or (se, wk) > cur[0]: st_depth[t] = ((se, wk), d)
    if len(games) < 300: sys.exit(f"only {len(games)} completed games - widen --seasons")
    print(f"{len(games)} games, {len(epa)} with play-by-play EPA")

    st = State(epa, inj); XS, XT, Y, MARG, TOT, SPR, TL, ML, SEAS = [], [], [], [], [], [], [], [], []
    for g in games:
        s, t = st.features(g)
        XS.append(s); XT.append(t); Y.append(1.0 if g["hs"] > g["as"] else 0.0 if g["hs"] < g["as"] else 0.5); MARG.append(g["hs"] - g["as"]); TOT.append(g["hs"] + g["as"])
        SPR.append(np.nan if g["spread"] is None else g["spread"]); TL.append(np.nan if g["total_line"] is None else g["total_line"]); pm = ml_prob(g["home_ml"], g["away_ml"]); ML.append(np.nan if pm is None else pm); SEAS.append(g["season"])
        st.update(g)
    XS, XT, Y, MARG, TOT, SPR, TL, ML, SEAS = map(np.array, (XS, XT, Y, MARG, TOT, SPR, TL, ML, SEAS))
    ties = Y == 0.5; Yb = Y.copy(); Yb[ties] = 1.0                       # ties (rare) count as a home win for the classifier; margin model sees 0
    seasons = sorted(int(v) for v in set(SEAS)); hold = seasons[-args.holdout:] if len(seasons) > args.holdout else seasons[-1:]
    te = np.isin(SEAS, hold); tr = ~te
    if te.sum() < 50:                                                    # current season barely started: hold out the previous one too
        hold = seasons[-args.holdout - 1:]; te = np.isin(SEAS, hold); tr = ~te
    print(f"train {int(tr.sum())} games ({seasons[0]}-{min(hold)-1}), holdout {int(te.sum())} games ({', '.join(map(str, hold))})\n")

    # ---------- moneyline
    print("\n  choosing regularisation on the training seasons only:")
    L2W = choose_l2_logistic(XS[tr], Yb[tr], label="moneyline")
    TEMPW = fit_temperature(XS[tr], Yb[tr], L2W)
    win = train_logistic(XS[tr], Yb[tr], l2=L2W)
    p_te_raw = predict_logistic(win, XS[te]); p_te = apply_temp(p_te_raw, TEMPW)
    print("holdout results - moneyline:"); acc, ll, brier = evaluate("base model (no market)", p_te, Yb[te])
    hm = ~np.isnan(ML[te]); stack = None
    if hm.sum() > 30:
        evaluate("market moneyline alone", ML[te][hm], Yb[te][hm]); evaluate("base model, same games", p_te[hm], Yb[te][hm])
        oo = oof(train_logistic, predict_logistic, XS[tr], Yb[tr]); ok = ~np.isnan(ML[tr])
        bl = train_logistic(np.c_[logit(oo[ok]), logit(ML[tr][ok])], Yb[tr][ok], l2=0.1)
        ps = predict_logistic(bl, np.c_[logit(p_te[hm]), logit(ML[te][hm])]); evaluate("STACKED model+market", ps, Yb[te][hm])
        wz = bl["w"] / bl["sd"]; stack = {"a": round(float(wz[0]), 5), "b": round(float(wz[1]), 5), "c": round(float(bl["b"] - (bl["mu"] / bl["sd"] * bl["w"]).sum()), 5)}
        print(f"  stacked blend: logit(p) = {wz[0]:+.2f}*model + {wz[1]:+.2f}*market (+const) on {int(ok.sum())} train games with a moneyline")
    print("\nmoneyline feature weights (standardised, bigger = more influence):")
    for n_, w_ in sorted(zip(WIN_FEATS, win["w"]), key=lambda x: -abs(x[1])): print(f"  {n_:<22} {w_:+.3f}")

    # ---------- spread (margin)
    print("\n  choosing regularisation for the margin model:")
    L2M = choose_l2_ridge(XS[tr], MARG[tr].astype(float), label="margin")
    mg = train_ridge(XS[tr], MARG[tr].astype(float), l2=L2M); m_te = predict_ridge(mg, XS[te])
    sd_m = float(np.std(MARG[tr] - predict_ridge(mg, XS[tr])))
    print(f"\nholdout results - margin (home minus away):\n  model MAE {np.abs(m_te - MARG[te]).mean():.3f}   (residual sd {sd_m:.2f} points)")
    hs = ~np.isnan(SPR[te]); mstack = None
    if hs.sum() > 30:
        print(f"  closing spread MAE {np.abs(SPR[te][hs] - MARG[te][hs]).mean():.3f}   model on same games {np.abs(m_te[hs] - MARG[te][hs]).mean():.3f}")
        oo = oof(train_ridge, predict_ridge, XS[tr], MARG[tr].astype(float)); ok = ~np.isnan(SPR[tr])
        A = np.c_[oo[ok], SPR[tr][ok], np.ones(int(ok.sum()))]; coef = np.linalg.lstsq(A, MARG[tr][ok], rcond=None)[0]
        mb = coef[0] * m_te[hs] + coef[1] * SPR[te][hs] + coef[2]; sd_mb = float(np.std(MARG[tr][ok] - (A @ coef)))
        a1, n1 = ats(m_te[hs], SPR[te][hs], MARG[te][hs]); a2, n2 = ats(mb, SPR[te][hs], MARG[te][hs])
        print(f"  blended margin = {coef[0]:.2f}*model + {coef[1]:.2f}*spread + {coef[2]:+.2f}   MAE {np.abs(mb - MARG[te][hs]).mean():.3f}   (residual sd {sd_mb:.2f})")
        print(f"  against the spread: raw model right side {a1*100:.1f}% of {n1}   blended right side {a2*100:.1f}% of {n2}   (52.4% needed to beat -110)")
        mstack = {"a": round(float(coef[0]), 5), "b": round(float(coef[1]), 5), "c": round(float(coef[2]), 5), "sd": round(sd_mb, 3), "ats_raw": round(a1, 4), "ats_blend": round(a2, 4)}
    print("\nmargin feature weights (points per standard deviation):")
    for n_, w_ in sorted(zip(WIN_FEATS, mg["w"]), key=lambda x: -abs(x[1]))[:12]: print(f"  {n_:<22} {w_:+.3f}")

    # ---------- total
    print("\n  choosing regularisation for the total model:")
    L2T = choose_l2_ridge(XT[tr], TOT[tr].astype(float), label="total")
    tm = train_ridge(XT[tr], TOT[tr].astype(float), l2=L2T); t_te = predict_ridge(tm, XT[te])
    sd_t = float(np.std(TOT[tr] - predict_ridge(tm, XT[tr])))
    print(f"\nholdout results - total points:\n  model MAE {np.abs(t_te - TOT[te]).mean():.3f}   (residual sd {sd_t:.2f})")
    ht = ~np.isnan(TL[te]); tstack = None
    if ht.sum() > 30:
        print(f"  closing total MAE {np.abs(TL[te][ht] - TOT[te][ht]).mean():.3f}   model on same games {np.abs(t_te[ht] - TOT[te][ht]).mean():.3f}")
        oo = oof(train_ridge, predict_ridge, XT[tr], TOT[tr].astype(float)); ok = ~np.isnan(TL[tr])
        A = np.c_[oo[ok], TL[tr][ok], np.ones(int(ok.sum()))]; coef = np.linalg.lstsq(A, TOT[tr][ok], rcond=None)[0]
        tb = coef[0] * t_te[ht] + coef[1] * TL[te][ht] + coef[2]; sd_tb = float(np.std(TOT[tr][ok] - (A @ coef)))
        a1, n1 = ats(t_te[ht], TL[te][ht], TOT[te][ht]); a2, n2 = ats(tb, TL[te][ht], TOT[te][ht])
        print(f"  blended total = {coef[0]:.2f}*model + {coef[1]:.2f}*line + {coef[2]:+.2f}   MAE {np.abs(tb - TOT[te][ht]).mean():.3f}   (residual sd {sd_tb:.2f})")
        print(f"  over/under: raw model right side {a1*100:.1f}% of {n1}   blended right side {a2*100:.1f}% of {n2}   (52.4% needed to beat -110)")
        tstack = {"a": round(float(coef[0]), 5), "b": round(float(coef[1]), 5), "c": round(float(coef[2]), 5), "sd": round(sd_tb, 3), "ou_raw": round(a1, 4), "ou_blend": round(a2, 4)}
    print("\ntotal feature weights (points per standard deviation):")
    for n_, w_ in sorted(zip(TOT_FEATS, tm["w"]), key=lambda x: -abs(x[1]))[:12]: print(f"  {n_:<22} {w_:+.3f}")

    # ---------- refit on everything and write
    win_f = train_logistic(XS, Yb, l2=L2W)
    mg_f = train_ridge(XS, MARG.astype(float), l2=L2M)
    tm_f = train_ridge(XT, TOT.astype(float), l2=L2T)
    teams = {t: {"name": TEAM_NAMES.get(t, t), "elo": round(st.elo[t], 1), "off": round(st.off[t], 4), "def": round(st.deff[t], 4), "off_pass": round(st.off_pass[t], 4), "def_pass": round(st.def_pass[t], 4),
                 "off_rush": round(st.off_rush[t], 4), "def_rush": round(st.def_rush[t], 4), "succ": round(st.succ[t], 4), "succ_allowed": round(st.succ_allowed[t], 4), "pass_rate": round(st.pass_rate[t], 4),
                 "margin": round(st.margin[t], 3), "pts_for": round(st.pts_for[t], 2), "pts_ag": round(st.pts_ag[t], 2), "form": [int(v) for v in st.form[t]], "qb": st.last_qb.get(t), "last_ts": st.last_ts.get(t, 0),
                 "season": int(st.season.get(t) or 0), "games": st.games[t], "coach": st.coach.get(t, ""), "coach_games": st.coach_n[(t, st.coach.get(t, ""))]} for t in st.elo if t in TEAM_NAMES}
    cutoff = max(st.last_ts.values()) - 2 * 365 * 86400
    qbs = {q: {"name": st.qb_name.get(q, q), "rating": round(st.qb_rating(q), 3), "starts": st.qb[q][1], "team": st.qb_team.get(q)} for q in st.qb if st.qb[q][1] >= 1 and st.last_ts.get(st.qb_team.get(q), 0) >= cutoff}
    out = {"game": "nfl", "version": 1, "as_of": datetime.now(timezone.utc).strftime("%Y-%m-%d"), "games_used": len(games), "seasons": seasons, "holdout_seasons": hold, "home_elo": HOME_ELO,
           "l2_win": float(L2W), "l2_margin": float(L2M), "l2_total": float(L2T), "temp": float(TEMPW),
           "win_feats": WIN_FEATS, "tot_feats": TOT_FEATS, "win_model": pack(win_f), "margin_model": pack(mg_f), "total_model": pack(tm_f), "sd_margin": round(sd_m, 3), "sd_total": round(sd_t, 3),
           "stack": stack, "margin_stack": mstack, "total_stack": tstack, "lg_total": round(float(np.mean(st.lg_total)), 3), "has_epa": bool(epa),
           "stadiums": {t: list(v) for t, v in STADIUM.items()}, "pos_w": POS_W, "off_pos": sorted(OFF_POS), "status_w": STATUS_W, "has_injuries": bool(inj),
           "usage": {t: {n: [round(v[0], 1), round(v[1], 1)] for n, v in d.items() if (v[0] or 0) >= 5 or (v[1] or 0) >= 5} for t, ((se, wk), d) in st_snaps.items()},
           "depth": {t: {n: v for n, v in d.items() if v[1] <= 2} for t, ((se, wk), d) in st_depth.items()}, "holdout": {"n": int(te.sum()), "accuracy": round(float(acc), 4), "logloss": round(float(ll), 4), "brier": round(float(brier), 4), "margin_mae": round(float(np.abs(m_te - MARG[te]).mean()), 3), "total_mae": round(float(np.abs(t_te - TOT[te]).mean()), 3)},
           "teams": teams, "qbs": qbs}
    json.dump(out, open(args.out, "w", encoding="utf-8"), separators=(",", ":"))
    print(f"\nwrote {args.out}: {len(teams)} teams, {len(qbs)} quarterbacks, {os.path.getsize(args.out)//1024} KB")


if __name__ == "__main__":
    main()
