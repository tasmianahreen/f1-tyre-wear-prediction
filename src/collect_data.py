import fastf1
import pandas as pd
from pathlib import Path

CACHE_DIR = Path("data/cache")
OUTPUT_DIR = Path("data")

CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

fastf1.Cache.enable_cache(str(CACHE_DIR))


def load_race(year: int, race: str, session_type: str = "R") -> pd.DataFrame:
    session = fastf1.get_session(year, race, session_type)
    session.load(weather=True)

    laps = session.laps.copy()

    cols = [
        "Driver",
        "Team",
        "LapNumber",
        "LapTime",
        "Compound",
        "TyreLife",
        "Stint",
        "PitInTime",
        "PitOutTime",
        "IsAccurate",
    ]

    laps = laps[cols]

    laps["LapTimeSeconds"] = laps["LapTime"].dt.total_seconds()

    laps = laps[
        laps["LapTimeSeconds"].notna()
        & laps["TyreLife"].notna()
        & laps["Compound"].notna()
        & (laps["IsAccurate"] == True)
    ].copy()

    laps["BestStintLap"] = laps.groupby(["Driver", "Stint"])["LapTimeSeconds"].transform("min")
    laps["Degradation"] = laps["LapTimeSeconds"] - laps["BestStintLap"]

    laps["PitLap"] = laps["PitInTime"].notna() | laps["PitOutTime"].notna()

    return laps


if __name__ == "__main__":
    df = load_race(2024, "Bahrain", "R")
    df.to_csv("data/bahrain_2024_laps.csv", index=False)
    print(df.head())
    print("Saved to data/bahrain_2024_laps.csv")
