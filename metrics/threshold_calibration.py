"""Validation-only threshold calibration for ETMNet revision protocol."""

import math


THRESHOLD_CANDIDATES = [x / 100.0 for x in range(5, 100, 5)]
EPS = 1e-12


def _value(row, key):
    try:
        value = row.get(key)
        if value in (None, "", "-"):
            return math.nan
        return float(value)
    except Exception:
        return math.nan


def select_diceopt(rows, dice_key="Dice/F1"):
    valid = [row for row in rows if math.isfinite(_value(row, dice_key))]
    if not valid:
        raise ValueError("DiceOpt requires at least one finite validation Dice/F1 value.")
    return sorted(valid, key=lambda r: (-_value(r, dice_key), abs(float(r["threshold"]) - 0.50), float(r["threshold"])))[0]


def add_measurement_normalization(rows, keys=("RAE", "BDE", "CE")):
    rows = [dict(row) for row in rows]
    for key in keys:
        values = [_value(row, key) for row in rows]
        values = [v for v in values if math.isfinite(v)]
        if not values:
            for row in rows:
                row[f"normalized_{key}"] = ""
            continue
        lo, hi = min(values), max(values)
        denom = hi - lo
        for row in rows:
            value = _value(row, key)
            if not math.isfinite(value):
                row[f"normalized_{key}"] = ""
            elif denom <= EPS:
                row[f"normalized_{key}"] = 0.0
            else:
                row[f"normalized_{key}"] = (value - lo) / denom
    for row in rows:
        vals = [_value(row, f"normalized_{key}") for key in keys]
        row["measurement_score"] = sum(vals) if all(math.isfinite(v) for v in vals) else ""
    return rows


def select_measurementopt(rows):
    normalized = add_measurement_normalization(rows)
    valid = [row for row in normalized if math.isfinite(_value(row, "measurement_score"))]
    if not valid:
        raise ValueError("MeasurementOpt requires finite validation RAE, BDE, and CE values.")
    return sorted(valid, key=lambda r: (_value(r, "measurement_score"), abs(float(r["threshold"]) - 0.50), float(r["threshold"])))[0]
