# Fundamentals: Load, Wind, Solar and Cross-Border Trading (2021–2024)

Quarter-hourly actual values and day-ahead forecasts for Germany and the four control areas. Source: ENTSO-E Transparency Platform via [Fraunhofer ISE Energy-Charts](https://www.energy-charts.info/). Time zone: UTC. Actual values may be revised vintages (downloaded after the fact).

## Content

| Folder | Variable | Areas | Columns |
| :--- | :--- | :--- | :--- |
| `LOAD/` | load | DE, DE1–DE4 | actual, day-ahead forecast (D-1 12:00) |
| `ONSHORE/` | wind onshore | DE, DE1–DE4 | actual, day-ahead forecast |
| `OFFSHORE/` | wind offshore | DE, DE3, DE4 (no offshore connections in DE1, DE2) | actual, day-ahead forecast |
| `SOLAR/` | solar | DE, DE1–DE4 | actual, day-ahead forecast |
| `CROSS_BORDER/` | cross-border commercial trading | DE | net value (MW) |

File names: `energy-charts_<VARIABLE>_Forecasts_and_actual_values_in_<Germany | the_control_area_of_<TSO>>_in_<year>.csv`.

## Use in the master dataset

Feature names `{zone}_{load|solar|wind_onshore|wind_offshore}_{actual|forecast|diff}` and `DE_cross_border_trading`. Forecasts are known the day before. Actual values (and the derived forecast errors) are lagged by 9 quarter-hours: published 60 min after the end of the quarter-hour (Reg. (EU) 543/2013, Art. 6(2)(a), 16(2)(b)) and available at the forecast time. `DE_cross_border_trading` (scheduled commercial exchanges including intraday; varies within the hour) is lagged by 12 quarter-hours: Art. 12(1)(f), 12(2)(e) allow updates up to two hours after each intraday nomination process; assuming the last nomination by delivery start, the value is final 105 min after the end of the quarter-hour. See `ex_post_lag_qh` in `features.csv`.

## Download

Energy-Charts → Power forecast, interval "year", time zone UTC:
`https://www.energy-charts.info/charts/power_forecast/chart.htm?l=en&c=DE&interval=year&year=<year>&dataType=<type>&timezone=utc`
with `<type>` = `load`, `solar`, `wind_onshore`, `wind_offshore` (Germany) or `<TSO>-<type>` (control area, e.g. `Amprion-load`). Cross-border trading: `https://www.energy-charts.info/charts/price_spot_market/chart.htm?l=en&c=DE&legendItems=iy1&year=<year>&interval=year&timezone=utc`.
