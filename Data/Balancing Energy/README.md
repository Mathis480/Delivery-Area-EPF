# Balancing Energy: Activated aFRR and mFRR (2021–2024)

Quarter-hourly activated balancing energy for Germany and the four control areas. Source: [netztransparenz.de](https://www.netztransparenz.de/de-de/Regelenergie/Daten-Regelreserve/Aktivierte-Regelleistung) (Aktivierte Regelleistung), operational data ("betrieblich", not the later quality-assured data). Time zone: UTC.

## Content

| Folder | Product | Files |
| :--- | :--- | :--- |
| `SRL/` | aFRR (secondary reserve) | `netztransparenz_SRL_aFRR_betrieblich_<year>.csv` |
| `MRL/` | mFRR (tertiary reserve) | `netztransparenz_MRL_mFRR_betrieblich_<year>.csv` |

Columns (`;`-separated, decimal comma): `Datum`, `Zeitzone`, `von`, `bis`, `Einheit` (MW), then positive and negative activation for 50Hertz, Amprion, TenneT TSO, TransnetBW and Deutschland. FCR is not included (no operational data before June 2022).

## Use in the master dataset

Feature names `{zone}_{SRL|MRL}_{positive|negative}` (DE1–DE4 mapped from the TSO columns, `DE` from Deutschland). Lagged by 6 quarter-hours: published 15 min after the end of the quarter-hour (netztransparenz, "Erläuterungen zu Daten Regelenergie", 10.07.2023, §4.1.2/4.1.3) and available at the forecast time; see `ex_post_lag_qh` in `features.csv`.

## Download

On the source page, select "Aktivierte aFRR betrieblich" or "Aktivierte mFRR betrieblich", resolution 15 min, time zone UTC, one calendar year per file, and export as CSV.
