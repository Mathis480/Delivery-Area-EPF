# Fundamentals: Power Generation and Load Forecasts (2021–2024)

This directory contains quarterly-hour (15-minute) fundamental time series data (actual measured values and Day-Ahead forecasts) for Germany as a whole and across the four German Transmission System Operator (TSO) control areas for the years 2021 through 2024.

## 1. Data Provenance and Attribution

* **Primary Data Source:** [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/)

    * **TransnetBW GmbH** (Control Area DE1 - Baden-Württemberg)
    * **Amprion GmbH** (Control Area DE2 - Western / South-Western Germany)
    * **TenneT TSO GmbH** (Control Area DE3 - Northern to Southern corridor)
    * **50Hertz Transmission GmbH** (Control Area DE4 - Eastern Germany & Baltic Sea)

* **Platform & Archival Provider:** [Fraunhofer ISE Energy-Charts](https://www.energy-charts.info/)

## 2. Dataset Structure and Quality Preprocessing

* **Time Resolution & Timezone:** Strictly 15-minute intervals parsed in **Pure UTC** (`Date (UTC)`).

* **Retained Features (exactly 3 columns per file):**
  1. `Date (UTC)`: UTC interval start timestamp.
  2. Actual Value: Measured real-time generation or load (`Solar`, `Wind onshore`, `Wind offshore`, or `Load`).
  3. Day-Ahead Forecast: Official D-1 forecast submitted prior to gate closure (`... forecast (Day-Ahead, D-1 ...)`).

## 3. Directory Structure

The 72 CSV files are organized into four dedicated domain subdirectories:
* `LOAD/` (20 files): Grid load actuals and Day-Ahead forecasts for Germany and the 4 TSO zones (2021–2024).
* `ONSHORE/` (20 files): Wind Onshore actual infeed and Day-Ahead forecasts (2021–2024).
* `OFFSHORE/` (12 files): Wind Offshore actual infeed and Day-Ahead forecasts for Germany, 50Hertz, and TenneT (2021–2024; TransnetBW and Amprion have no offshore transmission grid connections).
* `SOLAR/` (20 files): Solar PV actual infeed and Day-Ahead forecasts (2021–2024).

### LOAD (Total Load)

| Filename | Market / TSO Area | Year | Interactive Chart Link |
| :--- | :--- | :---: | :--- |
| `energy-charts_LOAD_Forecasts_and_actual_values_in_Germany_in_2021.csv` | Germany (National Total) | 2021 | [Energy-Charts (2021)](https://www.energy-charts.info/charts/power_forecast/chart.htm?l=en&c=DE&interval=year&year=2021&dataType=load&timezone=utc) |
| `energy-charts_LOAD_Forecasts_and_actual_values_in_Germany_in_2022.csv` | Germany (National Total) | 2022 | [Energy-Charts (2022)](https://www.energy-charts.info/charts/power_forecast/chart.htm?l=en&c=DE&interval=year&year=2022&dataType=load&timezone=utc) |
| `energy-charts_LOAD_Forecasts_and_actual_values_in_Germany_in_2023.csv` | Germany (National Total) | 2023 | [Energy-Charts (2023)](https://www.energy-charts.info/charts/power_forecast/chart.htm?l=en&c=DE&interval=year&year=2023&dataType=load&timezone=utc) |
| `energy-charts_LOAD_Forecasts_and_actual_values_in_Germany_in_2024.csv` | Germany (National Total) | 2024 | [Energy-Charts (2024)](https://www.energy-charts.info/charts/power_forecast/chart.htm?l=en&c=DE&interval=year&year=2024&dataType=load&timezone=utc) |
| `energy-charts_LOAD_Forecasts_and_actual_values_in_the_control_area_of_50Hertz_in_2021.csv` | 50Hertz (TSO Zone DE4) | 2021 | [Energy-Charts (2021)](https://www.energy-charts.info/charts/power_forecast/chart.htm?l=en&c=DE&interval=year&year=2021&dataType=50Hertz-load&timezone=utc) |
| `energy-charts_LOAD_Forecasts_and_actual_values_in_the_control_area_of_50Hertz_in_2022.csv` | 50Hertz (TSO Zone DE4) | 2022 | [Energy-Charts (2022)](https://www.energy-charts.info/charts/power_forecast/chart.htm?l=en&c=DE&interval=year&year=2022&dataType=50Hertz-load&timezone=utc) |
| `energy-charts_LOAD_Forecasts_and_actual_values_in_the_control_area_of_50Hertz_in_2023.csv` | 50Hertz (TSO Zone DE4) | 2023 | [Energy-Charts (2023)](https://www.energy-charts.info/charts/power_forecast/chart.htm?l=en&c=DE&interval=year&year=2023&dataType=50Hertz-load&timezone=utc) |
| `energy-charts_LOAD_Forecasts_and_actual_values_in_the_control_area_of_50Hertz_in_2024.csv` | 50Hertz (TSO Zone DE4) | 2024 | [Energy-Charts (2024)](https://www.energy-charts.info/charts/power_forecast/chart.htm?l=en&c=DE&interval=year&year=2024&dataType=50Hertz-load&timezone=utc) |
| `energy-charts_LOAD_Forecasts_and_actual_values_in_the_control_area_of_Amprion_in_2021.csv` | Amprion (TSO Zone DE2) | 2021 | [Energy-Charts (2021)](https://www.energy-charts.info/charts/power_forecast/chart.htm?l=en&c=DE&interval=year&year=2021&dataType=Amprion-load&timezone=utc) |
| `energy-charts_LOAD_Forecasts_and_actual_values_in_the_control_area_of_Amprion_in_2022.csv` | Amprion (TSO Zone DE2) | 2022 | [Energy-Charts (2022)](https://www.energy-charts.info/charts/power_forecast/chart.htm?l=en&c=DE&interval=year&year=2022&dataType=Amprion-load&timezone=utc) |
| `energy-charts_LOAD_Forecasts_and_actual_values_in_the_control_area_of_Amprion_in_2023.csv` | Amprion (TSO Zone DE2) | 2023 | [Energy-Charts (2023)](https://www.energy-charts.info/charts/power_forecast/chart.htm?l=en&c=DE&interval=year&year=2023&dataType=Amprion-load&timezone=utc) |
| `energy-charts_LOAD_Forecasts_and_actual_values_in_the_control_area_of_Amprion_in_2024.csv` | Amprion (TSO Zone DE2) | 2024 | [Energy-Charts (2024)](https://www.energy-charts.info/charts/power_forecast/chart.htm?l=en&c=DE&interval=year&year=2024&dataType=Amprion-load&timezone=utc) |
| `energy-charts_LOAD_Forecasts_and_actual_values_in_the_control_area_of_TenneT_in_2021.csv` | TenneT (TSO Zone DE3) | 2021 | [Energy-Charts (2021)](https://www.energy-charts.info/charts/power_forecast/chart.htm?l=en&c=DE&interval=year&year=2021&dataType=TenneT-load&timezone=utc) |
| `energy-charts_LOAD_Forecasts_and_actual_values_in_the_control_area_of_TenneT_in_2022.csv` | TenneT (TSO Zone DE3) | 2022 | [Energy-Charts (2022)](https://www.energy-charts.info/charts/power_forecast/chart.htm?l=en&c=DE&interval=year&year=2022&dataType=TenneT-load&timezone=utc) |
| `energy-charts_LOAD_Forecasts_and_actual_values_in_the_control_area_of_TenneT_in_2023.csv` | TenneT (TSO Zone DE3) | 2023 | [Energy-Charts (2023)](https://www.energy-charts.info/charts/power_forecast/chart.htm?l=en&c=DE&interval=year&year=2023&dataType=TenneT-load&timezone=utc) |
| `energy-charts_LOAD_Forecasts_and_actual_values_in_the_control_area_of_TenneT_in_2024.csv` | TenneT (TSO Zone DE3) | 2024 | [Energy-Charts (2024)](https://www.energy-charts.info/charts/power_forecast/chart.htm?l=en&c=DE&interval=year&year=2024&dataType=TenneT-load&timezone=utc) |
| `energy-charts_LOAD_Forecasts_and_actual_values_in_the_control_area_of_TransnetBW_in_2021.csv` | TransnetBW (TSO Zone DE1) | 2021 | [Energy-Charts (2021)](https://www.energy-charts.info/charts/power_forecast/chart.htm?l=en&c=DE&interval=year&year=2021&dataType=TransnetBW-load&timezone=utc) |
| `energy-charts_LOAD_Forecasts_and_actual_values_in_the_control_area_of_TransnetBW_in_2022.csv` | TransnetBW (TSO Zone DE1) | 2022 | [Energy-Charts (2022)](https://www.energy-charts.info/charts/power_forecast/chart.htm?l=en&c=DE&interval=year&year=2022&dataType=TransnetBW-load&timezone=utc) |
| `energy-charts_LOAD_Forecasts_and_actual_values_in_the_control_area_of_TransnetBW_in_2023.csv` | TransnetBW (TSO Zone DE1) | 2023 | [Energy-Charts (2023)](https://www.energy-charts.info/charts/power_forecast/chart.htm?l=en&c=DE&interval=year&year=2023&dataType=TransnetBW-load&timezone=utc) |
| `energy-charts_LOAD_Forecasts_and_actual_values_in_the_control_area_of_TransnetBW_in_2024.csv` | TransnetBW (TSO Zone DE1) | 2024 | [Energy-Charts (2024)](https://www.energy-charts.info/charts/power_forecast/chart.htm?l=en&c=DE&interval=year&year=2024&dataType=TransnetBW-load&timezone=utc) |


### ONSHORE (Wind Onshore Generation)

| Filename | Market / TSO Area | Year | Interactive Chart Link |
| :--- | :--- | :---: | :--- |
| `energy-charts_ONSHORE_Forecasts_and_actual_values_in_Germany_in_2021.csv` | Germany (National Total) | 2021 | [Energy-Charts (2021)](https://www.energy-charts.info/charts/power_forecast/chart.htm?l=en&c=DE&interval=year&year=2021&dataType=wind_onshore&timezone=utc) |
| `energy-charts_ONSHORE_Forecasts_and_actual_values_in_Germany_in_2022.csv` | Germany (National Total) | 2022 | [Energy-Charts (2022)](https://www.energy-charts.info/charts/power_forecast/chart.htm?l=en&c=DE&interval=year&year=2022&dataType=wind_onshore&timezone=utc) |
| `energy-charts_ONSHORE_Forecasts_and_actual_values_in_Germany_in_2023.csv` | Germany (National Total) | 2023 | [Energy-Charts (2023)](https://www.energy-charts.info/charts/power_forecast/chart.htm?l=en&c=DE&interval=year&year=2023&dataType=wind_onshore&timezone=utc) |
| `energy-charts_ONSHORE_Forecasts_and_actual_values_in_Germany_in_2024.csv` | Germany (National Total) | 2024 | [Energy-Charts (2024)](https://www.energy-charts.info/charts/power_forecast/chart.htm?l=en&c=DE&interval=year&year=2024&dataType=wind_onshore&timezone=utc) |
| `energy-charts_ONSHORE_Forecasts_and_actual_values_in_the_control_area_of_50Hertz_in_2021.csv` | 50Hertz (TSO Zone DE4) | 2021 | [Energy-Charts (2021)](https://www.energy-charts.info/charts/power_forecast/chart.htm?l=en&c=DE&interval=year&year=2021&dataType=50Hertz-wind_onshore&timezone=utc) |
| `energy-charts_ONSHORE_Forecasts_and_actual_values_in_the_control_area_of_50Hertz_in_2022.csv` | 50Hertz (TSO Zone DE4) | 2022 | [Energy-Charts (2022)](https://www.energy-charts.info/charts/power_forecast/chart.htm?l=en&c=DE&interval=year&year=2022&dataType=50Hertz-wind_onshore&timezone=utc) |
| `energy-charts_ONSHORE_Forecasts_and_actual_values_in_the_control_area_of_50Hertz_in_2023.csv` | 50Hertz (TSO Zone DE4) | 2023 | [Energy-Charts (2023)](https://www.energy-charts.info/charts/power_forecast/chart.htm?l=en&c=DE&interval=year&year=2023&dataType=50Hertz-wind_onshore&timezone=utc) |
| `energy-charts_ONSHORE_Forecasts_and_actual_values_in_the_control_area_of_50Hertz_in_2024.csv` | 50Hertz (TSO Zone DE4) | 2024 | [Energy-Charts (2024)](https://www.energy-charts.info/charts/power_forecast/chart.htm?l=en&c=DE&interval=year&year=2024&dataType=50Hertz-wind_onshore&timezone=utc) |
| `energy-charts_ONSHORE_Forecasts_and_actual_values_in_the_control_area_of_Amprion_in_2021.csv` | Amprion (TSO Zone DE2) | 2021 | [Energy-Charts (2021)](https://www.energy-charts.info/charts/power_forecast/chart.htm?l=en&c=DE&interval=year&year=2021&dataType=Amprion-wind_onshore&timezone=utc) |
| `energy-charts_ONSHORE_Forecasts_and_actual_values_in_the_control_area_of_Amprion_in_2022.csv` | Amprion (TSO Zone DE2) | 2022 | [Energy-Charts (2022)](https://www.energy-charts.info/charts/power_forecast/chart.htm?l=en&c=DE&interval=year&year=2022&dataType=Amprion-wind_onshore&timezone=utc) |
| `energy-charts_ONSHORE_Forecasts_and_actual_values_in_the_control_area_of_Amprion_in_2023.csv` | Amprion (TSO Zone DE2) | 2023 | [Energy-Charts (2023)](https://www.energy-charts.info/charts/power_forecast/chart.htm?l=en&c=DE&interval=year&year=2023&dataType=Amprion-wind_onshore&timezone=utc) |
| `energy-charts_ONSHORE_Forecasts_and_actual_values_in_the_control_area_of_Amprion_in_2024.csv` | Amprion (TSO Zone DE2) | 2024 | [Energy-Charts (2024)](https://www.energy-charts.info/charts/power_forecast/chart.htm?l=en&c=DE&interval=year&year=2024&dataType=Amprion-wind_onshore&timezone=utc) |
| `energy-charts_ONSHORE_Forecasts_and_actual_values_in_the_control_area_of_TenneT_in_2021.csv` | TenneT (TSO Zone DE3) | 2021 | [Energy-Charts (2021)](https://www.energy-charts.info/charts/power_forecast/chart.htm?l=en&c=DE&interval=year&year=2021&dataType=TenneT-wind_onshore&timezone=utc) |
| `energy-charts_ONSHORE_Forecasts_and_actual_values_in_the_control_area_of_TenneT_in_2022.csv` | TenneT (TSO Zone DE3) | 2022 | [Energy-Charts (2022)](https://www.energy-charts.info/charts/power_forecast/chart.htm?l=en&c=DE&interval=year&year=2022&dataType=TenneT-wind_onshore&timezone=utc) |
| `energy-charts_ONSHORE_Forecasts_and_actual_values_in_the_control_area_of_TenneT_in_2023.csv` | TenneT (TSO Zone DE3) | 2023 | [Energy-Charts (2023)](https://www.energy-charts.info/charts/power_forecast/chart.htm?l=en&c=DE&interval=year&year=2023&dataType=TenneT-wind_onshore&timezone=utc) |
| `energy-charts_ONSHORE_Forecasts_and_actual_values_in_the_control_area_of_TenneT_in_2024.csv` | TenneT (TSO Zone DE3) | 2024 | [Energy-Charts (2024)](https://www.energy-charts.info/charts/power_forecast/chart.htm?l=en&c=DE&interval=year&year=2024&dataType=TenneT-wind_onshore&timezone=utc) |
| `energy-charts_ONSHORE_Forecasts_and_actual_values_in_the_control_area_of_TransnetBW_in_2021.csv` | TransnetBW (TSO Zone DE1) | 2021 | [Energy-Charts (2021)](https://www.energy-charts.info/charts/power_forecast/chart.htm?l=en&c=DE&interval=year&year=2021&dataType=TransnetBW-wind_onshore&timezone=utc) |
| `energy-charts_ONSHORE_Forecasts_and_actual_values_in_the_control_area_of_TransnetBW_in_2022.csv` | TransnetBW (TSO Zone DE1) | 2022 | [Energy-Charts (2022)](https://www.energy-charts.info/charts/power_forecast/chart.htm?l=en&c=DE&interval=year&year=2022&dataType=TransnetBW-wind_onshore&timezone=utc) |
| `energy-charts_ONSHORE_Forecasts_and_actual_values_in_the_control_area_of_TransnetBW_in_2023.csv` | TransnetBW (TSO Zone DE1) | 2023 | [Energy-Charts (2023)](https://www.energy-charts.info/charts/power_forecast/chart.htm?l=en&c=DE&interval=year&year=2023&dataType=TransnetBW-wind_onshore&timezone=utc) |
| `energy-charts_ONSHORE_Forecasts_and_actual_values_in_the_control_area_of_TransnetBW_in_2024.csv` | TransnetBW (TSO Zone DE1) | 2024 | [Energy-Charts (2024)](https://www.energy-charts.info/charts/power_forecast/chart.htm?l=en&c=DE&interval=year&year=2024&dataType=TransnetBW-wind_onshore&timezone=utc) |


### OFFSHORE (Wind Offshore Generation)

| Filename | Market / TSO Area | Year | Interactive Chart Link |
| :--- | :--- | :---: | :--- |
| `energy-charts_OFFSHORE_Forecasts_and_actual_values_in_Germany_in_2021.csv` | Germany (National Total) | 2021 | [Energy-Charts (2021)](https://www.energy-charts.info/charts/power_forecast/chart.htm?l=en&c=DE&interval=year&year=2021&dataType=wind_offshore&timezone=utc) |
| `energy-charts_OFFSHORE_Forecasts_and_actual_values_in_Germany_in_2022.csv` | Germany (National Total) | 2022 | [Energy-Charts (2022)](https://www.energy-charts.info/charts/power_forecast/chart.htm?l=en&c=DE&interval=year&year=2022&dataType=wind_offshore&timezone=utc) |
| `energy-charts_OFFSHORE_Forecasts_and_actual_values_in_Germany_in_2023.csv` | Germany (National Total) | 2023 | [Energy-Charts (2023)](https://www.energy-charts.info/charts/power_forecast/chart.htm?l=en&c=DE&interval=year&year=2023&dataType=wind_offshore&timezone=utc) |
| `energy-charts_OFFSHORE_Forecasts_and_actual_values_in_Germany_in_2024.csv` | Germany (National Total) | 2024 | [Energy-Charts (2024)](https://www.energy-charts.info/charts/power_forecast/chart.htm?l=en&c=DE&interval=year&year=2024&dataType=wind_offshore&timezone=utc) |
| `energy-charts_OFFSHORE_Forecasts_and_actual_values_in_the_control_area_of_50Hertz_in_2021.csv` | 50Hertz (TSO Zone DE4) | 2021 | [Energy-Charts (2021)](https://www.energy-charts.info/charts/power_forecast/chart.htm?l=en&c=DE&interval=year&year=2021&dataType=50Hertz-wind_offshore&timezone=utc) |
| `energy-charts_OFFSHORE_Forecasts_and_actual_values_in_the_control_area_of_50Hertz_in_2022.csv` | 50Hertz (TSO Zone DE4) | 2022 | [Energy-Charts (2022)](https://www.energy-charts.info/charts/power_forecast/chart.htm?l=en&c=DE&interval=year&year=2022&dataType=50Hertz-wind_offshore&timezone=utc) |
| `energy-charts_OFFSHORE_Forecasts_and_actual_values_in_the_control_area_of_50Hertz_in_2023.csv` | 50Hertz (TSO Zone DE4) | 2023 | [Energy-Charts (2023)](https://www.energy-charts.info/charts/power_forecast/chart.htm?l=en&c=DE&interval=year&year=2023&dataType=50Hertz-wind_offshore&timezone=utc) |
| `energy-charts_OFFSHORE_Forecasts_and_actual_values_in_the_control_area_of_50Hertz_in_2024.csv` | 50Hertz (TSO Zone DE4) | 2024 | [Energy-Charts (2024)](https://www.energy-charts.info/charts/power_forecast/chart.htm?l=en&c=DE&interval=year&year=2024&dataType=50Hertz-wind_offshore&timezone=utc) |
| `energy-charts_OFFSHORE_Forecasts_and_actual_values_in_the_control_area_of_TenneT_in_2021.csv` | TenneT (TSO Zone DE3) | 2021 | [Energy-Charts (2021)](https://www.energy-charts.info/charts/power_forecast/chart.htm?l=en&c=DE&interval=year&year=2021&dataType=TenneT-wind_offshore&timezone=utc) |
| `energy-charts_OFFSHORE_Forecasts_and_actual_values_in_the_control_area_of_TenneT_in_2022.csv` | TenneT (TSO Zone DE3) | 2022 | [Energy-Charts (2022)](https://www.energy-charts.info/charts/power_forecast/chart.htm?l=en&c=DE&interval=year&year=2022&dataType=TenneT-wind_offshore&timezone=utc) |
| `energy-charts_OFFSHORE_Forecasts_and_actual_values_in_the_control_area_of_TenneT_in_2023.csv` | TenneT (TSO Zone DE3) | 2023 | [Energy-Charts (2023)](https://www.energy-charts.info/charts/power_forecast/chart.htm?l=en&c=DE&interval=year&year=2023&dataType=TenneT-wind_offshore&timezone=utc) |
| `energy-charts_OFFSHORE_Forecasts_and_actual_values_in_the_control_area_of_TenneT_in_2024.csv` | TenneT (TSO Zone DE3) | 2024 | [Energy-Charts (2024)](https://www.energy-charts.info/charts/power_forecast/chart.htm?l=en&c=DE&interval=year&year=2024&dataType=TenneT-wind_offshore&timezone=utc) |


### SOLAR (Solar Photovoltaic Generation)

| Filename | Market / TSO Area | Year | Interactive Chart Link |
| :--- | :--- | :---: | :--- |
| `energy-charts_SOLAR_Forecasts_and_actual_values_in_Germany_in_2021.csv` | Germany (National Total) | 2021 | [Energy-Charts (2021)](https://www.energy-charts.info/charts/power_forecast/chart.htm?l=en&c=DE&interval=year&year=2021&dataType=solar&timezone=utc) |
| `energy-charts_SOLAR_Forecasts_and_actual_values_in_Germany_in_2022.csv` | Germany (National Total) | 2022 | [Energy-Charts (2022)](https://www.energy-charts.info/charts/power_forecast/chart.htm?l=en&c=DE&interval=year&year=2022&dataType=solar&timezone=utc) |
| `energy-charts_SOLAR_Forecasts_and_actual_values_in_Germany_in_2023.csv` | Germany (National Total) | 2023 | [Energy-Charts (2023)](https://www.energy-charts.info/charts/power_forecast/chart.htm?l=en&c=DE&interval=year&year=2023&dataType=solar&timezone=utc) |
| `energy-charts_SOLAR_Forecasts_and_actual_values_in_Germany_in_2024.csv` | Germany (National Total) | 2024 | [Energy-Charts (2024)](https://www.energy-charts.info/charts/power_forecast/chart.htm?l=en&c=DE&interval=year&year=2024&dataType=solar&timezone=utc) |
| `energy-charts_SOLAR_Forecasts_and_actual_values_in_the_control_area_of_50Hertz_in_2021.csv` | 50Hertz (TSO Zone DE4) | 2021 | [Energy-Charts (2021)](https://www.energy-charts.info/charts/power_forecast/chart.htm?l=en&c=DE&interval=year&year=2021&dataType=50Hertz-solar&timezone=utc) |
| `energy-charts_SOLAR_Forecasts_and_actual_values_in_the_control_area_of_50Hertz_in_2022.csv` | 50Hertz (TSO Zone DE4) | 2022 | [Energy-Charts (2022)](https://www.energy-charts.info/charts/power_forecast/chart.htm?l=en&c=DE&interval=year&year=2022&dataType=50Hertz-solar&timezone=utc) |
| `energy-charts_SOLAR_Forecasts_and_actual_values_in_the_control_area_of_50Hertz_in_2023.csv` | 50Hertz (TSO Zone DE4) | 2023 | [Energy-Charts (2023)](https://www.energy-charts.info/charts/power_forecast/chart.htm?l=en&c=DE&interval=year&year=2023&dataType=50Hertz-solar&timezone=utc) |
| `energy-charts_SOLAR_Forecasts_and_actual_values_in_the_control_area_of_50Hertz_in_2024.csv` | 50Hertz (TSO Zone DE4) | 2024 | [Energy-Charts (2024)](https://www.energy-charts.info/charts/power_forecast/chart.htm?l=en&c=DE&interval=year&year=2024&dataType=50Hertz-solar&timezone=utc) |
| `energy-charts_SOLAR_Forecasts_and_actual_values_in_the_control_area_of_Amprion_in_2021.csv` | Amprion (TSO Zone DE2) | 2021 | [Energy-Charts (2021)](https://www.energy-charts.info/charts/power_forecast/chart.htm?l=en&c=DE&interval=year&year=2021&dataType=Amprion-solar&timezone=utc) |
| `energy-charts_SOLAR_Forecasts_and_actual_values_in_the_control_area_of_Amprion_in_2022.csv` | Amprion (TSO Zone DE2) | 2022 | [Energy-Charts (2022)](https://www.energy-charts.info/charts/power_forecast/chart.htm?l=en&c=DE&interval=year&year=2022&dataType=Amprion-solar&timezone=utc) |
| `energy-charts_SOLAR_Forecasts_and_actual_values_in_the_control_area_of_Amprion_in_2023.csv` | Amprion (TSO Zone DE2) | 2023 | [Energy-Charts (2023)](https://www.energy-charts.info/charts/power_forecast/chart.htm?l=en&c=DE&interval=year&year=2023&dataType=Amprion-solar&timezone=utc) |
| `energy-charts_SOLAR_Forecasts_and_actual_values_in_the_control_area_of_Amprion_in_2024.csv` | Amprion (TSO Zone DE2) | 2024 | [Energy-Charts (2024)](https://www.energy-charts.info/charts/power_forecast/chart.htm?l=en&c=DE&interval=year&year=2024&dataType=Amprion-solar&timezone=utc) |
| `energy-charts_SOLAR_Forecasts_and_actual_values_in_the_control_area_of_TenneT_in_2021.csv` | TenneT (TSO Zone DE3) | 2021 | [Energy-Charts (2021)](https://www.energy-charts.info/charts/power_forecast/chart.htm?l=en&c=DE&interval=year&year=2021&dataType=TenneT-solar&timezone=utc) |
| `energy-charts_SOLAR_Forecasts_and_actual_values_in_the_control_area_of_TenneT_in_2022.csv` | TenneT (TSO Zone DE3) | 2022 | [Energy-Charts (2022)](https://www.energy-charts.info/charts/power_forecast/chart.htm?l=en&c=DE&interval=year&year=2022&dataType=TenneT-solar&timezone=utc) |
| `energy-charts_SOLAR_Forecasts_and_actual_values_in_the_control_area_of_TenneT_in_2023.csv` | TenneT (TSO Zone DE3) | 2023 | [Energy-Charts (2023)](https://www.energy-charts.info/charts/power_forecast/chart.htm?l=en&c=DE&interval=year&year=2023&dataType=TenneT-solar&timezone=utc) |
| `energy-charts_SOLAR_Forecasts_and_actual_values_in_the_control_area_of_TenneT_in_2024.csv` | TenneT (TSO Zone DE3) | 2024 | [Energy-Charts (2024)](https://www.energy-charts.info/charts/power_forecast/chart.htm?l=en&c=DE&interval=year&year=2024&dataType=TenneT-solar&timezone=utc) |
| `energy-charts_SOLAR_Forecasts_and_actual_values_in_the_control_area_of_TransnetBW_in_2021.csv` | TransnetBW (TSO Zone DE1) | 2021 | [Energy-Charts (2021)](https://www.energy-charts.info/charts/power_forecast/chart.htm?l=en&c=DE&interval=year&year=2021&dataType=TransnetBW-solar&timezone=utc) |
| `energy-charts_SOLAR_Forecasts_and_actual_values_in_the_control_area_of_TransnetBW_in_2022.csv` | TransnetBW (TSO Zone DE1) | 2022 | [Energy-Charts (2022)](https://www.energy-charts.info/charts/power_forecast/chart.htm?l=en&c=DE&interval=year&year=2022&dataType=TransnetBW-solar&timezone=utc) |
| `energy-charts_SOLAR_Forecasts_and_actual_values_in_the_control_area_of_TransnetBW_in_2023.csv` | TransnetBW (TSO Zone DE1) | 2023 | [Energy-Charts (2023)](https://www.energy-charts.info/charts/power_forecast/chart.htm?l=en&c=DE&interval=year&year=2023&dataType=TransnetBW-solar&timezone=utc) |
| `energy-charts_SOLAR_Forecasts_and_actual_values_in_the_control_area_of_TransnetBW_in_2024.csv` | TransnetBW (TSO Zone DE1) | 2024 | [Energy-Charts (2024)](https://www.energy-charts.info/charts/power_forecast/chart.htm?l=en&c=DE&interval=year&year=2024&dataType=TransnetBW-solar&timezone=utc) |

