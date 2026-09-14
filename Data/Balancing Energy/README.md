# Balancing Energy: Activated Balancing Reserves (SRL, MRL) (2021–2024)

This directory contains quarterly-hour (15-minute) time series data for activated balancing reserves (balancing power) across Germany and its four Transmission System Operator (TSO) control areas for the years 2021 through 2024.

## 1. Data Provenance and Attribution

* **Primary Source:** [Netztransparenz.de](https://www.netztransparenz.de/)
    * **50Hertz Transmission GmbH** (Control Area DE4)
    * **Amprion GmbH** (Control Area DE2)
    * **TenneT TSO GmbH** (Control Area DE3)
    * **TransnetBW GmbH** (Control Area DE1)
* **Source Webpage:** [Netztransparenz > Regelenergie > Daten Regelreserve > Aktivierte Regelleistung](https://www.netztransparenz.de/de-de/Regelenergie/Daten-Regelreserve/Aktivierte-Regelleistung)

## 2. Dataset Structure and Methodological Rigor

### Operational vs. Quality-Assured Data (Look-Ahead Bias Prevention)
* **Dataset Selection:** Strictly **Operational Data ("Betrieblich")** instead of Quality-Assured ("Qualitätsgesichert").

### Temporal Properties & Timezone
* **Time Resolution:** Exactly 15-minute delivery intervals (`von` - `bis`).
* **Timezone:** Strictly **Pure UTC** (`Zeitzone: UTC`).

### Column Specification (15 Columns)
* **Timestamp & Metadata:**
  * `Datum`: Delivery date (`DD.MM.YYYY` format in UTC).
  * `Zeitzone`: Explicitly labeled `UTC`.
  * `von` / `bis`: Interval start and end times in UTC (`HH:MM`).
  * `Einheit`: Power unit (`MW`).
* **Positive Activation (Upward Balancing Power / Feed-in Increase):**
  * `50Hertz (Positiv)`
  * `Amprion (Positiv)`
  * `TenneT TSO (Positiv)`
  * `TransnetBW (Positiv)`
  * `Deutschland (Positiv)` (aggregate national total)
* **Negative Activation (Downward Balancing Power / Feed-in Decrease):**
  * `50Hertz (Negativ)`
  * `Amprion (Negativ)`
  * `TenneT TSO (Negativ)`
  * `TransnetBW (Negativ)`
  * `Deutschland (Negativ)` (aggregate national total)

## 3. Data Availability and Scope Note

* **SRL (aFRR - Automatic Frequency Restoration Reserve):** Fully available across all 4 years (2021, 2022, 2023, 2024).
* **MRL (mFRR - Manual Frequency Restoration Reserve):** Fully available across all 4 years (2021, 2022, 2023, 2024).
* **PRL Exclusion:** Primary Control Reserve (PRL / FCR) is excluded from this dataset because historical operational data prior to June 21, 2022 is not available on the platform.

## 4. Directory Structure and File Inventory

```
Data/Balancing Energy/
├── SRL/
│   ├── netztransparenz_SRL_aFRR_betrieblich_2021.csv
│   ├── netztransparenz_SRL_aFRR_betrieblich_2022.csv
│   ├── netztransparenz_SRL_aFRR_betrieblich_2023.csv
│   └── netztransparenz_SRL_aFRR_betrieblich_2024.csv
├── MRL/
│   ├── netztransparenz_MRL_mFRR_betrieblich_2021.csv
│   ├── netztransparenz_MRL_mFRR_betrieblich_2022.csv
│   ├── netztransparenz_MRL_mFRR_betrieblich_2023.csv
│   └── netztransparenz_MRL_mFRR_betrieblich_2024.csv
└── README.md
```

### SRL (Secondary Balancing Power / aFRR: Aktivierte aFRR betrieblich)

| Filename | Coverage | Timezone | Direct 1-Click CSV Download (UTC) | Interactive Portal Section |
| :--- | :---: | :---: | :--- | :--- |
| `netztransparenz_SRL_aFRR_betrieblich_2021.csv` | 01.01.2021–31.12.2021 | UTC | [Direct Download (2021 UTC)](https://www.netztransparenz.de/DesktopModules/LotesCharts/CsvDownloadHandler.ashx?request=eyJMb2NhbEZyb20iOiAiMjAyMS0wMS0wMSIsICJMb2NhbFRvIjogIjIwMjItMDEtMDEiLCAiUmVzdWx0VGltZVpvbmUiOiAidXRjIiwgIlNldHRpbmdzIjogeyJEYXRhVHlwZSI6IDIwLCAiQ3VsdHVyZU5hbWUiOiAiZGUtREUiLCAiRGlhZ3JhbVR5cGUiOiAibGluZSIsICJUaW1lSW50ZXJ2YWwiOiAxNSwgIkRhdGFVbml0IjogIk1XIiwgIkNzdkNvbHVtbnMiOiBbIjUwSGVydHoiLCAiQW1wcmlvbiIsICJUZW5uZVQgVFNPIiwgIlRyYW5zbmV0QlciXSwgIlRzb0lkcyI6IFswLCAxLCAyLCAzLCA0XSwgIk5ydkRpcmVjdGlvbiI6IDMsICJXZWJBcGlCYXNlVXJpIjogImh0dHBzOi8vbG90ZXMtVU5CLXN2Yy1uZXR6dC5jb3JwLnRyYW5zbWlzc2lvbi1pdC5kZS9TdGF0aXN0aWtBcGkvIiwgIlByb2R1a3RJZCI6IDIsICJUaXRsZSI6ICJBa3RpdmllcnRlIGFGUlIgYmV0cmllYmxpY2giLCAiV2ViQXBpUm91dGUiOiAiTnJ2U2FsZG8vQWt0aXZpZXJ0ZVNSTC9CZXRyaWVibGljaCJ9fQ%3D%3D) | [Interactive Chart (SRL)](https://www.netztransparenz.de/de-de/Regelenergie/Daten-Regelreserve/Aktivierte-Regelleistung#dnn_ctr2113_ModuleContent) |
| `netztransparenz_SRL_aFRR_betrieblich_2022.csv` | 01.01.2022–31.12.2022 | UTC | [Direct Download (2022 UTC)](https://www.netztransparenz.de/DesktopModules/LotesCharts/CsvDownloadHandler.ashx?request=eyJMb2NhbEZyb20iOiAiMjAyMi0wMS0wMSIsICJMb2NhbFRvIjogIjIwMjMtMDEtMDEiLCAiUmVzdWx0VGltZVpvbmUiOiAidXRjIiwgIlNldHRpbmdzIjogeyJEYXRhVHlwZSI6IDIwLCAiQ3VsdHVyZU5hbWUiOiAiZGUtREUiLCAiRGlhZ3JhbVR5cGUiOiAibGluZSIsICJUaW1lSW50ZXJ2YWwiOiAxNSwgIkRhdGFVbml0IjogIk1XIiwgIkNzdkNvbHVtbnMiOiBbIjUwSGVydHoiLCAiQW1wcmlvbiIsICJUZW5uZVQgVFNPIiwgIlRyYW5zbmV0QlciXSwgIlRzb0lkcyI6IFswLCAxLCAyLCAzLCA0XSwgIk5ydkRpcmVjdGlvbiI6IDMsICJXZWJBcGlCYXNlVXJpIjogImh0dHBzOi8vbG90ZXMtVU5CLXN2Yy1uZXR6dC5jb3JwLnRyYW5zbWlzc2lvbi1pdC5kZS9TdGF0aXN0aWtBcGkvIiwgIlByb2R1a3RJZCI6IDIsICJUaXRsZSI6ICJBa3RpdmllcnRlIGFGUlIgYmV0cmllYmxpY2giLCAiV2ViQXBpUm91dGUiOiAiTnJ2U2FsZG8vQWt0aXZpZXJ0ZVNSTC9CZXRyaWVibGljaCJ9fQ%3D%3D) | [Interactive Chart (SRL)](https://www.netztransparenz.de/de-de/Regelenergie/Daten-Regelreserve/Aktivierte-Regelleistung#dnn_ctr2113_ModuleContent) |
| `netztransparenz_SRL_aFRR_betrieblich_2023.csv` | 01.01.2023–31.12.2023 | UTC | [Direct Download (2023 UTC)](https://www.netztransparenz.de/DesktopModules/LotesCharts/CsvDownloadHandler.ashx?request=eyJMb2NhbEZyb20iOiAiMjAyMy0wMS0wMSIsICJMb2NhbFRvIjogIjIwMjQtMDEtMDEiLCAiUmVzdWx0VGltZVpvbmUiOiAidXRjIiwgIlNldHRpbmdzIjogeyJEYXRhVHlwZSI6IDIwLCAiQ3VsdHVyZU5hbWUiOiAiZGUtREUiLCAiRGlhZ3JhbVR5cGUiOiAibGluZSIsICJUaW1lSW50ZXJ2YWwiOiAxNSwgIkRhdGFVbml0IjogIk1XIiwgIkNzdkNvbHVtbnMiOiBbIjUwSGVydHoiLCAiQW1wcmlvbiIsICJUZW5uZVQgVFNPIiwgIlRyYW5zbmV0QlciXSwgIlRzb0lkcyI6IFswLCAxLCAyLCAzLCA0XSwgIk5ydkRpcmVjdGlvbiI6IDMsICJXZWJBcGlCYXNlVXJpIjogImh0dHBzOi8vbG90ZXMtVU5CLXN2Yy1uZXR6dC5jb3JwLnRyYW5zbWlzc2lvbi1pdC5kZS9TdGF0aXN0aWtBcGkvIiwgIlByb2R1a3RJZCI6IDIsICJUaXRsZSI6ICJBa3RpdmllcnRlIGFGUlIgYmV0cmllYmxpY2giLCAiV2ViQXBpUm91dGUiOiAiTnJ2U2FsZG8vQWt0aXZpZXJ0ZVNSTC9CZXRyaWVibGljaCJ9fQ%3D%3D) | [Interactive Chart (SRL)](https://www.netztransparenz.de/de-de/Regelenergie/Daten-Regelreserve/Aktivierte-Regelleistung#dnn_ctr2113_ModuleContent) |
| `netztransparenz_SRL_aFRR_betrieblich_2024.csv` | 01.01.2024–31.12.2024 | UTC | [Direct Download (2024 UTC)](https://www.netztransparenz.de/DesktopModules/LotesCharts/CsvDownloadHandler.ashx?request=eyJMb2NhbEZyb20iOiAiMjAyNC0wMS0wMSIsICJMb2NhbFRvIjogIjIwMjUtMDEtMDEiLCAiUmVzdWx0VGltZVpvbmUiOiAidXRjIiwgIlNldHRpbmdzIjogeyJEYXRhVHlwZSI6IDIwLCAiQ3VsdHVyZU5hbWUiOiAiZGUtREUiLCAiRGlhZ3JhbVR5cGUiOiAibGluZSIsICJUaW1lSW50ZXJ2YWwiOiAxNSwgIkRhdGFVbml0IjogIk1XIiwgIkNzdkNvbHVtbnMiOiBbIjUwSGVydHoiLCAiQW1wcmlvbiIsICJUZW5uZVQgVFNPIiwgIlRyYW5zbmV0QlciXSwgIlRzb0lkcyI6IFswLCAxLCAyLCAzLCA0XSwgIk5ydkRpcmVjdGlvbiI6IDMsICJXZWJBcGlCYXNlVXJpIjogImh0dHBzOi8vbG90ZXMtVU5CLXN2Yy1uZXR6dC5jb3JwLnRyYW5zbWlzc2lvbi1pdC5kZS9TdGF0aXN0aWtBcGkvIiwgIlByb2R1a3RJZCI6IDIsICJUaXRsZSI6ICJBa3RpdmllcnRlIGFGUlIgYmV0cmllYmxpY2giLCAiV2ViQXBpUm91dGUiOiAiTnJ2U2FsZG8vQWt0aXZpZXJ0ZVNSTC9CZXRyaWVibGljaCJ9fQ%3D%3D) | [Interactive Chart (SRL)](https://www.netztransparenz.de/de-de/Regelenergie/Daten-Regelreserve/Aktivierte-Regelleistung#dnn_ctr2113_ModuleContent) |

### MRL (Tertiary Balancing Power / mFRR: Aktivierte mFRR betrieblich)

| Filename | Coverage | Timezone | Direct 1-Click CSV Download (UTC) | Interactive Portal Section |
| :--- | :---: | :---: | :--- | :--- |
| `netztransparenz_MRL_mFRR_betrieblich_2021.csv` | 01.01.2021–31.12.2021 | UTC | [Direct Download (2021 UTC)](https://www.netztransparenz.de/DesktopModules/LotesCharts/CsvDownloadHandler.ashx?request=eyJMb2NhbEZyb20iOiAiMjAyMS0wMS0wMSIsICJMb2NhbFRvIjogIjIwMjItMDEtMDEiLCAiUmVzdWx0VGltZVpvbmUiOiAidXRjIiwgIlNldHRpbmdzIjogeyJEYXRhVHlwZSI6IDIwLCAiQ3VsdHVyZU5hbWUiOiAiZGUtREUiLCAiRGlhZ3JhbVR5cGUiOiAibGluZSIsICJUaW1lSW50ZXJ2YWwiOiAxNSwgIkRhdGFVbml0IjogIk1XIiwgIkNzdkNvbHVtbnMiOiBbIjUwSGVydHoiLCAiQW1wcmlvbiIsICJUZW5uZVQgVFNPIiwgIlRyYW5zbmV0QlciXSwgIlRzb0lkcyI6IFswLCAxLCAyLCAzLCA0XSwgIk5ydkRpcmVjdGlvbiI6IDMsICJXZWJBcGlCYXNlVXJpIjogImh0dHBzOi8vbG90ZXMtVU5CLXN2Yy1uZXR6dC5jb3JwLnRyYW5zbWlzc2lvbi1pdC5kZS9TdGF0aXN0aWtBcGkvIiwgIlByb2R1a3RJZCI6IDE4LCAiVGl0bGUiOiAiQWt0aXZpZXJ0ZSBtRlJSIGJldHJpZWJsaWNoIiwgIldlYkFwaVJvdXRlIjogIk5ydlNhbGRvL0FrdGl2aWVydGVNUkwvQmV0cmllYmxpY2gifX0%3D) | [Interactive Chart (MRL)](https://www.netztransparenz.de/de-de/Regelenergie/Daten-Regelreserve/Aktivierte-Regelleistung#dnn_ctr2414_ModuleContent) |
| `netztransparenz_MRL_mFRR_betrieblich_2022.csv` | 01.01.2022–31.12.2022 | UTC | [Direct Download (2022 UTC)](https://www.netztransparenz.de/DesktopModules/LotesCharts/CsvDownloadHandler.ashx?request=eyJMb2NhbEZyb20iOiAiMjAyMi0wMS0wMSIsICJMb2NhbFRvIjogIjIwMjMtMDEtMDEiLCAiUmVzdWx0VGltZVpvbmUiOiAidXRjIiwgIlNldHRpbmdzIjogeyJEYXRhVHlwZSI6IDIwLCAiQ3VsdHVyZU5hbWUiOiAiZGUtREUiLCAiRGlhZ3JhbVR5cGUiOiAibGluZSIsICJUaW1lSW50ZXJ2YWwiOiAxNSwgIkRhdGFVbml0IjogIk1XIiwgIkNzdkNvbHVtbnMiOiBbIjUwSGVydHoiLCAiQW1wcmlvbiIsICJUZW5uZVQgVFNPIiwgIlRyYW5zbmV0QlciXSwgIlRzb0lkcyI6IFswLCAxLCAyLCAzLCA0XSwgIk5ydkRpcmVjdGlvbiI6IDMsICJXZWJBcGlCYXNlVXJpIjogImh0dHBzOi8vbG90ZXMtVU5CLXN2Yy1uZXR6dC5jb3JwLnRyYW5zbWlzc2lvbi1pdC5kZS9TdGF0aXN0aWtBcGkvIiwgIlByb2R1a3RJZCI6IDE4LCAiVGl0bGUiOiAiQWt0aXZpZXJ0ZSBtRlJSIGJldHJpZWJsaWNoIiwgIldlYkFwaVJvdXRlIjogIk5ydlNhbGRvL0FrdGl2aWVydGVNUkwvQmV0cmllYmxpY2gifX0%3D) | [Interactive Chart (MRL)](https://www.netztransparenz.de/de-de/Regelenergie/Daten-Regelreserve/Aktivierte-Regelleistung#dnn_ctr2414_ModuleContent) |
| `netztransparenz_MRL_mFRR_betrieblich_2023.csv` | 01.01.2023–31.12.2023 | UTC | [Direct Download (2023 UTC)](https://www.netztransparenz.de/DesktopModules/LotesCharts/CsvDownloadHandler.ashx?request=eyJMb2NhbEZyb20iOiAiMjAyMy0wMS0wMSIsICJMb2NhbFRvIjogIjIwMjQtMDEtMDEiLCAiUmVzdWx0VGltZVpvbmUiOiAidXRjIiwgIlNldHRpbmdzIjogeyJEYXRhVHlwZSI6IDIwLCAiQ3VsdHVyZU5hbWUiOiAiZGUtREUiLCAiRGlhZ3JhbVR5cGUiOiAibGluZSIsICJUaW1lSW50ZXJ2YWwiOiAxNSwgIkRhdGFVbml0IjogIk1XIiwgIkNzdkNvbHVtbnMiOiBbIjUwSGVydHoiLCAiQW1wcmlvbiIsICJUZW5uZVQgVFNPIiwgIlRyYW5zbmV0QlciXSwgIlRzb0lkcyI6IFswLCAxLCAyLCAzLCA0XSwgIk5ydkRpcmVjdGlvbiI6IDMsICJXZWJBcGlCYXNlVXJpIjogImh0dHBzOi8vbG90ZXMtVU5CLXN2Yy1uZXR6dC5jb3JwLnRyYW5zbWlzc2lvbi1pdC5kZS9TdGF0aXN0aWtBcGkvIiwgIlByb2R1a3RJZCI6IDE4LCAiVGl0bGUiOiAiQWt0aXZpZXJ0ZSBtRlJSIGJldHJpZWJsaWNoIiwgIldlYkFwaVJvdXRlIjogIk5ydlNhbGRvL0FrdGl2aWVydGVNUkwvQmV0cmllYmxpY2gifX0%3D) | [Interactive Chart (MRL)](https://www.netztransparenz.de/de-de/Regelenergie/Daten-Regelreserve/Aktivierte-Regelleistung#dnn_ctr2414_ModuleContent) |
| `netztransparenz_MRL_mFRR_betrieblich_2024.csv` | 01.01.2024–31.12.2024 | UTC | [Direct Download (2024 UTC)](https://www.netztransparenz.de/DesktopModules/LotesCharts/CsvDownloadHandler.ashx?request=eyJMb2NhbEZyb20iOiAiMjAyNC0wMS0wMSIsICJMb2NhbFRvIjogIjIwMjUtMDEtMDEiLCAiUmVzdWx0VGltZVpvbmUiOiAidXRjIiwgIlNldHRpbmdzIjogeyJEYXRhVHlwZSI6IDIwLCAiQ3VsdHVyZU5hbWUiOiAiZGUtREUiLCAiRGlhZ3JhbVR5cGUiOiAibGluZSIsICJUaW1lSW50ZXJ2YWwiOiAxNSwgIkRhdGFVbml0IjogIk1XIiwgIkNzdkNvbHVtbnMiOiBbIjUwSGVydHoiLCAiQW1wcmlvbiIsICJUZW5uZVQgVFNPIiwgIlRyYW5zbmV0QlciXSwgIlRzb0lkcyI6IFswLCAxLCAyLCAzLCA0XSwgIk5ydkRpcmVjdGlvbiI6IDMsICJXZWJBcGlCYXNlVXJpIjogImh0dHBzOi8vbG90ZXMtVU5CLXN2Yy1uZXR6dC5jb3JwLnRyYW5zbWlzc2lvbi1pdC5kZS9TdGF0aXN0aWtBcGkvIiwgIlByb2R1a3RJZCI6IDE4LCAiVGl0bGUiOiAiQWt0aXZpZXJ0ZSBtRlJSIGJldHJpZWJsaWNoIiwgIldlYkFwaVJvdXRlIjogIk5ydlNhbGRvL0FrdGl2aWVydGVNUkwvQmV0cmllYmxpY2gifX0%3D) | [Interactive Chart (MRL)](https://www.netztransparenz.de/de-de/Regelenergie/Daten-Regelreserve/Aktivierte-Regelleistung#dnn_ctr2414_ModuleContent) |
