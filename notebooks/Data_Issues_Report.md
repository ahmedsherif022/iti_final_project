# Data Issues Report
## HR Analytics: Job Change of Data Scientists

**Dataset size:** 19,158 rows &times; 14 columns (train) | 2,129 rows &times; 13 columns (test)
**Task:** Binary classification — predict if a candidate will look for a new job (`target`)

---

## 1. Missing Data

This is the biggest problem in the dataset. Almost every categorical column has missing values.

| Column | Missing count | Missing % | Severity |
|---|---|---|---|
| `company_type` | 6,140 | **32.05%** | High |
| `company_size` | 5,938 | **30.99%** | High |
| `gender` | 4,508 | **23.53%** | High |
| `major_discipline` | 2,813 | **14.68%** | Medium |
| `last_new_job` | 423 | 2.21% | Low |
| `education_level` | 460 | 2.40% | Low |
| `enrolled_university` | 386 | 2.01% | Low |
| `experience` | 65 | 0.34% | Very low |
| All other columns | 0 | 0% | None |

**Why it matters:** Most ML models (e.g. Logistic Regression, Random Forest) cannot handle `NaN` values and will fail or silently ignore those rows (~32% of rows would be dropped if we removed them).

**How we handled it:**
- Categorical columns (e.g. `gender`, `major_discipline`) → filled with the **most common value (mode)** or the label `"Unknown"`.
- Numeric-style columns (`experience`, `last_new_job`) → filled with the **mode**.
- Result: **0 missing values** left after preprocessing.

---

## 2. Categorical Data (Text Columns)

Almost the whole dataset is textual. 9 of 14 columns are categorical and cannot be fed to a model directly.

**Ordinal columns** (order matters — higher = more):
| Column | Distinct values (in order) |
|---|---|
| `education_level` | Primary School < High School < Graduate < Masters < Phd |
| `company_size` | <10 < 10/49 < 50-99 < 100-500 < 500-999 < 1000-4999 < 5000-9999 < 10000+ |
| `experience` | 0 &ndash; 20+ years |
| `last_new_job` | never < 1 < 2 < 3 < 4 < >4 |

**Nominal columns** (no order — just categories):
| Column | Distinct values | Notes |
|---|---|---|
| `gender` | Male / Female / Other | **Imbalanced:** Male = 13,221, Female = 1,238, Other = 191 |
| `relevent_experience` | Has / No relevant experience | 72% vs 28% (imbalanced) |
| `enrolled_university` | no_enrollment / Full time / Part time | |
| `major_discipline` | STEM / Business / Arts / Humanities / No Major / Other | STEM = 75% of rows |
| `company_type` | Pvt Ltd / Startup / NGO / Public / Other | Pvt Ltd = 51% of rows |

**Why it matters:** Models only understand numbers, so text must be converted.

**How we handled it:**
- **Ordinal encoding** (`map` with an order dictionary) for `education_level` and `company_size` — preserves the natural order.
- **One-hot encoding** (`pd.get_dummies`, `drop_first=True`) for the nominal columns — creates one 0/1 column per category without inventing false order.

---

## 3. "Text Numbers" — Disguised Numeric Values

`experience` and `last_new_job` look like numbers but are stored as **text with symbols**:

- `experience` values: `'<1'`, `'5'`, `'15'`, `'17'`, `'>20'` ... (22 distinct text values)
- `last_new_job` values: `'never'`, `'1'`, `'4'`, `'>4'` ...

**Why it matters:** If left as text, the model treats them as random categories (e.g. `'15'` would not be bigger than `'5'`). The order and the actual numeric meaning are lost.

**How we handled it:** Mapped symbols to real numbers then cast to `int`:
- `'<1' → 0`, `'>20' → 21`, `'never' → 0`, `'>4' → 5`
- Result: clean integer columns the model can compare properly.

---

## 4. High-Cardinality Column: `city`

| Column | Distinct values |
|---|---|
| `city` | **123 unique codes** (city_1 ... city_123) |
| `city_development_index` | 93 unique values (range 0.448 – 0.949) |

**Why it matters:**
- One-hot encoding `city` would create **122 extra columns** (curse of dimensionality) and each city has little data.
- There is **redundancy**: 123 cities but only 93 different development-index values → several cities share the same index.

**How we handled it:** Dropped `city` and **kept `city_development_index`** — it already captures the city's development level as a numeric score, which is the information that actually matters for this task.

---

## 5. Class Imbalance in `target`

| Class | Count | Percentage |
|---|---|---|
| 0 (not looking for a job) | 14,378 | **75.1%** |
| 1 (looking for a job) | 4,780 | **24.9%** |

**Why it matters:** A "dumb" model that always predicts `0` would get **75% accuracy** without learning anything. The model may become biased toward the majority class and fail to detect candidates who want to change jobs (the class we actually care about).

**How we handled it:**
- Used `class_weight='balanced'` in both Random Forest and Logistic Regression — automatically gives more weight to the minority class.
- Used **stratified** train/validation split so both classes keep the same 75/25 balance in training and validation.

---

## 6. Irrelevant / Redundant Columns

| Column | Problem | Solution |
|---|---|---|
| `enrollee_id` | Unique for every row (19,158 IDs) → gives no prediction power | Dropped |
| `city` | 123 categories, redundant with the index | Dropped |
| `city_development_index` | Duplicate numeric-ish values, but useful | **Kept** |
| `training_hours` | Ranges 1 – 336, valid | Kept |

---

## 7. Outliers and Skewed Numeric Features

- `training_hours`: mean ≈ 65, but max = **336** and std = 60 → right-skewed distribution, some very high values.
- `city_development_index`: skewed toward high values (median 0.90, min 0.448) — most candidates come from developed cities.
- `experience`: piled up at `>20` (3,286 candidates) — a common "cap" value in surveys.

**Why it matters:** Tree-based models handle this fine, but distance-based models (Logistic Regression) can be pulled by extreme values.

**How we handled it:** Kept values as-is for Random Forest (robust to outliers). `experience` >20 was capped at 21 as a **representative number** instead of keeping text.

---

## 8. Summary of Problems and Solutions

| # | Problem | Impact | Solution |
|---|---|---|---|
| 1 | Missing values (up to 32% per column) | Models can't handle NaN | Fill with mode / "Unknown" |
| 2 | 9 categorical text columns | Model needs numbers | Ordinal + one-hot encoding |
| 3 | Text numbers with symbols (`'<1'`, `'>20'`) | Order/numeric meaning lost | Map to integers |
| 4 | `city` with 123 categories | Dimensionality explosion | Drop, keep `city_development_index` |
| 5 | Imbalanced target (75/25) | Model biased to majority | `class_weight='balanced'` + stratified split |
| 6 | `enrollee_id` (unique IDs) | No predictive value | Drop |
| 7 | Skewed features / outliers | Distorts distance-based models | Capped values; Random Forest robust |

**Final result after preprocessing:**
- 19,158 rows, **0 missing values**, no text columns
- 23 numeric features ready for the model
- Random Forest achieves **~78.6%** validation accuracy vs Logistic Regression **~72.1%**