# MTS Anomaly Detection
Multidimensional Time Series Anomaly Detection (MTSAD)


# Glossary
- Anomaly (`+1`) -> Anomalies -> Anomaly Detection
- Normal (`-1`)
- Observed
- Predicted


# Data
## MTS<sup>2</sup> Format
| t | [features] | tag |
|:--:|:--:|:--:|
| t<sub>0</sub> | [v<sub>0</sub>, v<sub>1</sub>, ..., v<sub>d-1</sub>] | -1 / 'n' |
| t<sub>1</sub> | [v<sub>0</sub>, v<sub>1</sub>, ..., v<sub>d-1</sub>] | +1 / 'a' |
| ... | ... | ... |

Columns:

- **t**: datetime.
- **[features]**: *d* features.
- **tag**: `+1` or `a` for anomaly, `-1` or `n` for normal.


MTS<sup>2</sup> Format
:    Multidimensional Time Series Supervised (MTSS)


**Note:**
All datasets should be transformed into this format for further processing.


# Algorithm
## Supervised

## Semi-Supervised

## Unsupervised


# System Architecture


# License
`MTSAnomalyDetection` is licensed under the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0).
See [LICENSE](LICENSE) for the full license text.