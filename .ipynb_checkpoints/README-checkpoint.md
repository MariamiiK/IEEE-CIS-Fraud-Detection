# IEEE-CIS Fraud Detection

## Kaggle-ის კონკურსის მოკლე მიმოხილვა

IEEE-CIS Fraud Detection კონკურსი მოიცავს ელექტრონული კომერციის ტრანზაქციების მონაცემებს, რომელთა მიზანია თაღლითური ტრანზაქციების (`isFraud`) გამოვლენა. Dataset შედგება ორი ცხრილისგან: `train_transaction.csv` და `train_identity.csv`, რომლებიც `TransactionID`-ის მეშვეობით უკავშირდება ერთმანეთს. სულ **590,540 სატრენინგო** და **506,691 სატესტო** ტრანზაქცია მოიცავს 434 მახასიათებელს. კლასების შეფარდება არათანაბარია — მხოლოდ **3.5%** (20,663) ტრანზაქცია არის თაღლითური (**27.6:1 შეფარდება**). შეფასება ხდება **ROC-AUC** მეტრიკით, რაც უზრუნველყოფს მოდელის ეფექტურობის გაზომვას კლასების დისბალანსის პირობებში.

---

## ჩემი მიდგომა პრობლემის გადასაჭრელად

პრობლემა გადავწყვიტე მრავალეტაპიანი მიდგომით:

**მონაცემთა დამუშავება:** transaction და identity ცხრილების გაერთიანება `TransactionID`-ით, >90% missing მქონე სვეტების ამოღება, NaN შევსება მედიანით/`unknown`-ით და კატეგორიული ცვლადების Label Encoding.

**Feature Engineering:** ახალი მახასიათებლების შექმნა TransactionDT-დან (საათი, კვირის დღე), TransactionAmt-დან (ლოგარითმი, z-score card1-ის მიხედვით) და email domain-ებიდან.

**Feature Selection:** სამი მიდგომის შედარება — SelectKBest, RF Importance, Correlation.

**Training:** 6 სხვადასხვა მოდელის გატესტვა სხვადასხვა ჰიპერპარამეტრებით, underfitting და overfitting ანალიზი, Optuna-ს საშუალებით ოპტიმიზაცია.

---

## რეპოზიტორიის სტრუქტურა

```
IEEE-CIS-Fraud-Detection/
├── images
├── model_experiment_LogisticRegression.ipynb
├── model_experiment_DecisionTree.ipynb
├── model_experiment_RandomForest.ipynb
├── model_experiment_XGBoost.ipynb
├── model_experiment_LightGBM.ipynb
├── model_experiment_NeuralNetwork.ipynb
├── model_inference.ipynb
└── README.md
```

### ფაილების განმარტება

**`model_experiment_{არქიტექტურა}.ipynb`** — თითოეული მოდელისთვის ცალკე notebook, რომელიც მოიცავს Cleaning, Feature Engineering, Feature Selection და Training სექციებს Heading-ებით გამოყოფილს. ყველა ექსპერიმენტი დალოგილია MLflow-ზე.

**`model_inference.ipynb`** — საუკეთესო მოდელის ჩატვირთვა Model Registry-დან, test set-ზე პროგნოზირება და `submission.csv`-ის გენერირება Kaggle-ზე ასატვირთად.

**`README.md`** — პროექტის სრული დოკუმენტაცია ქართულ ენაზე.

---
### მონაცემთა დამუშავება და Cleaning

გამოვიყენე ერთგვაროვანი Cleaning pipeline ყველა notebook-ში:

```
train + test ჩატვირთვა → merge (TransactionID) → drop_high_missing(0.9) →
feature_engineering() → encode_and_fill(median) → reindex(test→train cols) →
reduce_mem_usage()
```

Merge: Transaction და Identity ცხრილების გაერთიანება TransactionID-ით.

**`Handling Missing Values:`**

- ამოვიღე სვეტები, რომელთა 90%-ზე მეტი ცარიელი იყო (drop_high_missing).
- რიცხვითი მნიშვნელობები შეივსო მედიანით.
- კატეგორიული მნიშვნელობები შეივსო სტრინგით 'unknown'.

 Encoding: კატეგორიული ცვლადების გარდაქმნა მოხდა Label Encoding-ით.

შედეგი: train dataset **590,540 × 432** ზომის DataFrame. Fraud rate: **0.0350**.

## Feature Engineering

### შექმნილი მახასიათებლები

**TransactionDT-დან** :

- `tx_hour` — დღის საათი (0–23)
- `tx_dayofwk` — კვირის დღე (0–6)
- `tx_week` — კვირის ინდექსი
- `tx_month` — თვე (0–11)
- `is_night` — ღამის ტრანზაქციის ბინარული ნიშანი (00:00–05:59), ვინაიდან ღამის ტრანზაქციები უფრო მეტი რისკის მატარებელია

**TransactionAmt-დან:**

- `TransactionAmt_log` — ლოგარითმული სკალა სიმეტრიისთვის
- `TransactionAmt_card1_z` — z-score ერთი და იმავე ბარათის ტრანზაქციებთან შედარებით (გამოთვლა: `(amt - card1_mean) / card1_std`). თაღლითური ტრანზაქციები ხშირად გამოირჩევა ბარათის ჩვეულ ქცევაზე
- `is_round_amount` — "მრგვალი" თანხების ნიშანი (% 1 == 0), გავრცელებულია თაღლითობაში

**Email domain-ებიდან:**

- `P_email_tld`, `R_email_tld` — გამყიდველისა და მყიდველის email domain-ის top-level domain
- `same_email_domain` — ემთხვევა თუ არა P და R email domain-ები (ბინარული ნიშანი)

**ბარათის ვინაობისთვის:**

- `card1_addr1` — card1 + addr1-ის კომბინაციური feature ("თითის ანაბეჭდი" ბარათ-მისამართის წყვილისთვის)

Feature Engineering-ის შემდეგ dataset-ი **434-დან 445 სვეტამდე** გაიზარდა, Feature Selection-მდე კი საბოლოო სამუშაო dataset: **(590,540, 150)** train-ისთვის და **(506,691, 150)** test-ისთვის.

![Top 30 Features by RF Importance](images/feature_importance.png)

### კატეგორიული ცვლადების კოდირება

გამოვიყენე **Label Encoding** ყველა კატეგორიული ცვლადისთვის (`sklearn.preprocessing.LabelEncoder`). კოდირება ხდება ცალ-ცალკე თითოეული feature-ისთვის. NaN კატეგორიული ცვლადებისთვის ჯერ `'unknown'`-ით ვავსებდი, შემდეგ ვასრულებდი encoding-ს. test-ში train-ში არმქონე სვეტები `fill_value=0`-ით ივსება.

### NaN მნიშვნელობების დამუშავება

- **214 სვეტს >50% missing ჰქონდა** (id_24, id_25, id_07, id_08 — 99%-ზე მეტი missing)
- **>90% missing მქონე 12 სვეტი ამოვიღეთ მთლიანად** — `drop_high_missing(threshold=0.9)` — მათი საინფორმაციო ღირებულება დაბალი იყო
- **დარჩენილი numeric სვეტები** შეივსო **მედიანით** (outlier-მდგრადი სტრატეგია)
- **კატეგორიული სვეტები** შეივსო **`'unknown'`-ით**


---

## Feature Selection

გამოვიყენე სამი მიდგომა და შევადარე მათი შედეგები:

### 1. SelectKBest (f_classif)

სტატისტიკური ტესტი, რომელიც ANOVA F-value-ს იყენებს. სწრაფი, მაგრამ მხოლოდ **წრფივ** დამოკიდებულებებს ითვალისწინებს. გამოვიყენეთ Logistic Regression-ისთვის (linear model-ისთვის შესაფერისი). 
**შეირჩა 150 feature.**

### 2. RandomForest Feature Importance (საუკეთესო)

RandomForest მოდელი, რომელიც impurity reduction-ის მიხედვით აფასებს მახასიათებლებს. ითვალისწინებს **არაწრფივ** დამოკიდებულებებს. სიჩქარის გასაზრდელად გამოვიყენეთ **10% stratified subsample** და გავწვრთენით **small RF**. **შეირჩა top 150 feature.** 
გამოყენება: DT, RF, XGBoost, LightGBM, NeuralNetwork.

ყველაზე მნიშვნელოვანი feature-ები RF-ის მიხედვით: **V244, V201, V259, V200, V258, C8, C11, V243, V187, C13** (V-სერიის ანონიმური feature-ები და C-სერიის counting feature-ები დომინირებს).

### 3. Correlation with Target

Pearson კორელაცია target-თან. სწრაფია, მაგრამ მხოლოდ monotonic კავშირებს ხედავს. შედარებისთვის გამოვიყენე.
გამოირიცხა მულტიკოლინეარული მახასიათებლები, რომელთა კორელაცია ერთმანეთთან $> 0.95$ იყო, რაც დაეხმარა მოდელების სტაბილურობას.

### Feature Selection შედეგების შედარება

| მეთოდი | Feature-ები | Overlap KBest ∩ RF | გამოყენება |

| SelectKBest (k=150) | 150 | — | Logistic Regression |
| RF Importance (top 150) | 150 | ~59–62 | DT, RF, XGB, LGBM, NN |
| Correlation (top 150) | 150 | 59–62 | შედარება |

**Overlap KBest ∩ Corr: 150** — ეს ორი მიდგომა ბევრ feature-ს ემთხვევა, ვინაიდან ორივე წრფივ ურთიერთობებს ხედავს. **RF Importance ∩ Corr: ~59** — tree-based მოდელი განსხვავებულ სიგნალებს პოულობს.

---

## Training

### მონაცემების გაყოფა

```
80% Train | 20% Validation
StratifiedKFold (3 ან 5-fold) cross-validation
```

StratifiedKFold კრიტიკულია **27.6:1 class imbalance-ის** გამო, ის უზრუნველყოფს fraud შემთხვევების თანაბარ განაწილებას ყველა fold-ში.

### ტესტირებული მოდელები

| # | მოდელი                 | Notebook                                  | Experiment 

| 1 | LogisticRegression     | model_experiment_LogisticRegression.ipynb | LogisticRegression_Training 
| 2 | DecisionTreeClassifier | model_experiment_DecisionTree.ipynb       | DecisionTree_Training 
| 3 | RandomForestClassifier | model_experiment_RandomForest.ipynb       | RandomForest_Training 
| 4 | XGBClassifier          | model_experiment_XGBoost.ipynb            | XGBoost_Training 
| 5 | LGBMClassifier (GPU)   | model_experiment_LightGBM.ipyn            | LightGBM_Training 
| 6 | PyTorch FraudNet       | model_experiment_NeuralNetwork.ipynb      | NeuralNetwork_Training 



## Logistic Regression

### Hyperparameter ოპტიმიზაცია

**C (Regularization):**

| C | Train AUC | Val AUC | Gap 

| 0.001 | 0.7252 | 0.7487 | -0.0235 
| 0.01 | 0.7253 | **0.7497** | -0.0244 
| 0.1 | 0.7254 | 0.7492 | -0.0239 
| 1.0 | 0.7252 | 0.7493 | -0.0241 
| 10.0 | 0.7252 | 0.7493 | -0.0241 

საუკეთესო C: **0.01**. შენიშვნა: უარყოფითი gap (val > train) განპირობებულია ძლიერი რეგულარიზაციით,
რომელიც ზღუდავს (constrain) მოდელის სირთულეს სატრენინგო მონაცემებზე, რათა გაუმჯობესდეს მისი განზოგადების უნარი

**Penalty:**

| Penalty | Train AUC | Val AUC | Zero-coefs 

| L1 | 0.7251 | 0.7476 | 15 
| L2 | 0.7256 | **0.7494** | 0 

საუკეთესო: **L2**

**Cross-Validation (3-fold):** Fold 1: 0.7328, Fold 2: 0.7329, Fold 3: 0.7209 → **Mean: 0.7289 ± 0.0056**

**Final Model:** C=0.01, L2, lbfgs, class_weight=balanced → **Val AUC: 0.72, CV AUC: 0.7289 ± 0.0056**

**განმარტება:** LR-ს Val AUC ~0.72–0.75 რჩება C-ს ნებისმიერ მნიშვნელობაზე.  ეს Underfitting-ის კლასიკური ნიშანია: 
წრფივი მოდელი (Linear Model) ვერ ახდენს მონაცემებში არსებული კომპლექსური, არაწრფივ კავშირებისა და ინტერაქციების მოდელირებას

---

## Decision Tree

### Overfitting vs Underfitting ანალიზი

| depth | Train AUC | Val AUC | Gap | სტატუსი |

| 1 | 0.6277 | 0.6281 | -0.0004 | underfit |
| 4 | 0.7940 | 0.7961 | -0.0021 | good_fit |
| **8** | **0.8605** | **0.8485** | **0.0120** | **good_fit** |
| 20 | 0.9474 | 0.8548 | 0.0926 | severe_overfit |
| None | 0.9667 | 0.8480 | 0.1187 | severe_overfit |

depth=1 → ძალიან მარტივი, ვერ სწავლობს (underfit). depth=None → ყველა training sample-ი ზეპირდება (severe overfit, gap=0.1187).

### Hyperparameter ოპტიმიზაცია

min_samples_split-ის sweeping (depth=8): **mss=200** → საუკეთესო Val AUC.
criterion: **gini** > entropy
class_weight: **balanced** → უკეთ ახასიათებს imbalanced კლასებს

**Cross-Validation (5-fold):**
Fold 1: 0.8518, Fold 2: 0.8475, Fold 3: 0.8508, Fold 4: 0.8465, Fold 5: 0.8501
**Mean: 0.8493 ± 0.0020**

**Final Model:** depth=8, mss=200, gini, balanced → **Val AUC: 0.861, CV AUC: 0.8493 ± 0.0020**

![DT Overfitting Analysis](images/dt_overfit_analysis.png)
---

## Random Forest

### Baseline

Train: 1.0 | OOB: 0.909 | Val: **0.9238** | Gap: 0.0762

OOB AUC ≈ Val AUC — ადასტურებს bootstrap resampling-ის სისწორეს. Train=1.0 overfitting-ის ნიშანია, მაგრამ OOB/Val ჯერ კიდევ მაღალია

### n_estimators sweep

| n_estimators | OOB AUC | Val AUC | Gap |

| 10 | 0.7992 | 0.8677 | 0.1323 |
| 30 | 0.8702 | 0.9062 | 0.0938 |
| 50 | 0.8893 | 0.9161 | 0.0839 |
| **100** | **0.9090** | **0.9238** | **0.0762** |

### max_depth sweep

| max_depth | Val AUC | Gap |

| 5 | 0.8487 | 0.0022 |
| 10 | 0.8783 | 0.0105 |
| 15 | 0.8984 | 0.0399 |
| **20** | **0.9092** | **0.0714** |
| None | 0.9161 | 0.0839 |

### max_features sweep

| max_features | Val AUC | Gap |

| sqrt | 0.9092 | 0.0714 |
| log2 | 0.9012 | 0.0677 |
| 0.2 | 0.9119 | 0.0755 |
| **0.4** | **0.9137** | **0.0751** |

### class_weight sweep

| class_weight | Val AUC | Gap |

| none | 0.8923 | 0.0210 |
| **balanced** | **0.9137** | **0.0751** |
| balanced_subsample | 0.9132 | 0.0762 |

### Overfitting vs Underfitting

| depth | Train AUC | OOB AUC | Val AUC | Gap | სტატუსი |

| 2 | 0.7637 | 0.7343 | 0.7689 | -0.0052 | underfit |
| 5 | 0.8489 | 0.8457 | 0.8460 | 0.0029 | good_fit |
| 20 | 0.9888 | 0.9029 | 0.9137 | 0.0751 | overfit |
| None | 1.0000 | 0.8962 | 0.9218 | 0.0782 | overfit |

RF-ის bagging-ი ამცირებს variance-ს: depth=None-ზეც კი train/val gap (0.078) გაცილებით მცირეა ვიდრე single DT-ში (0.1187).

**Cross-Validation (3-fold, n=50 per fold):**
Fold 1: 0.8964, Fold 2: 0.8964, Fold 3: 0.9021 → **Mean: 0.8983 ± 0.0027**

**Final Model:** n=100, depth=20, max_features=0.4, balanced → **Final Pipeline Val AUC: 0.9877, OOB: 0.9183, CV: 0.8983 ± 0.0027**

 **შენიშვნა:** Pipeline Val AUC=0.9877 — ეს dataset leakage-ია: Pipeline-ი train+val-ზე train-დება, შემდეგ val-ზე ფასდება. სამართლიანი CV AUC=0.8983.

![RF Overfitting Analysis](images/rf_overfit_analysis.png)
---

## XGBoost

### Baseline

Train: 0.9861 | Val: **0.9512** | Gap: 0.0349 | Best_n: 999

### learning_rate sweep

| lr | Val AUC | Gap |

| 0.30 | 0.9564 | 0.0434 |
| **0.10** | **0.9580** | **0.0386** |
| 0.05 | 0.9512 | 0.0349 |
| 0.01 | 0.9141 | 0.0162 |

### max_depth sweep

| max_depth | Val AUC | Gap |

| 3 | 0.9262 | 0.0179 |
| 5 | 0.9516 | 0.0360 |
| 6 | 0.9580 | 0.0386 |
| 7 | 0.9601 | 0.0395 |
| **9** | **0.9623** | **0.0377** |

### subsample × colsample_bytree

| sub | col | Val AUC | Gap |

| 0.6 | 0.6 | 0.9598 | 0.0402 |
| 0.6 | 0.8 | 0.9611 | 0.0389 |
| **0.8** | **0.6** | **0.9625** | **0.0375** |
| 0.8 | 0.8 | 0.9623 | 0.0377 |
| 1.0 | 1.0 | 0.9552 | 0.0442 |

### scale_pos_weight

| spw     | Val AUC 

| **1.0** | **0.9648** 
| 27.58   | 0.9625 
 
### Overfitting vs Underfitting

| run | Train AUC | Val AUC | Gap | სტატუსი |

| XGB_extreme_underfit | 0.8411 | 0.8380 | 0.0031 | good_fit |
| XGB_underfit | 0.8742 | 0.8726 | 0.0016 | good_fit |
| **XGB_good_fit** | **0.9999** | **0.9647** | **0.0352** | **good_fit** |
| XGB_overfit_deep | 1.0000 | 0.9568 | 0.0432 | overfit |
| XGB_severe_overfit | 1.0000 | 0.9502 | 0.0498 | overfit |

XGBoost-ს built-in regularization (reg_alpha, reg_lambda, subsampling) ამცირებს overfitting-ის ტენდენციას — ძნელია მნიშვნელოვანი gap-ის მიღება.

![XGB Overfitting Analysis](images/xgb_overfit_analysis.png)

### Optuna (20 trials)

Best Val AUC: **0.9629**
Best params: lr=0.1575, depth=9, subsample=0.866, colsample=0.739, reg_alpha=0.001, reg_lambda=0.065, min_child_weight=3

### Cross-Validation (3-fold)

Fold 1: 0.9470, Fold 2: 0.9486, Fold 3: 0.9522 → **Mean: 0.9493 ± 0.0022**

**Final Model:** Optuna best params → **Final Pipeline Val AUC: 0.9999, CV: 0.9493 ± 0.0022**

 **შენიშვნა:** Val AUC=0.9999 — Pipeline leakage (ანალოგიური RF-ის). CV AUC=0.9493 სამართლიანი შეფასებაა.

---

## LightGBM

### Baseline

Train: 0.8416 | Val: **0.8397** | Gap: 0.0019 | Best_n: 1 (early stopping-ით)

GPU: Tesla T4 — LightGBM GPU trainer აჩქარებს training-ს.

### learning_rate sweep

| lr | Val AUC | Gap |

| **0.30** | **0.9542** | **0.0440** |
| 0.10 | 0.8397 | 0.0019 |
| 0.05 | 0.8397 | 0.0019 |
| 0.01 | 0.8731 | 0.0029 |

### num_leaves sweep

| num_leaves | Val AUC | Gap |

| 15 | 0.9408 | 0.0425 |
| 31 | 0.9562 | 0.0432 |
| 63 | 0.9581 | 0.0419 |
| **127** | **0.9592** | **0.0408** |
| 255 | 0.9553 | 0.0447 |

### subsample × colsample_bytree

| sub | col | Val AUC | Gap |

| 0.6 | 0.6 | 0.9603 | 0.0397 |
| 0.6 | 0.8 | 0.9581 | 0.0419 |
| **0.8** | **0.6** | **0.9605** | **0.0395** |
| 0.8 | 0.8 | 0.9590 | 0.0410 |
| 1.0 | 1.0 | 0.9556 | 0.0444 |

### Optuna (20 trials)

Best Val AUC: **0.9645**
Best params: lr=0.1876, num_leaves=183, subsample=0.985, colsample=0.709, reg_alpha=0.00043, reg_lambda=2.53e-05, min_child_samples=90

### Cross-Validation (3-fold)

Fold 1: 0.9497, Fold 2: 0.9503, Fold 3: 0.9519 → **Mean: 0.951 ± 0.001**

**Final Model:** Optuna best params → **Final Val AUC: 1.0, CV: 0.951 ± 0.001**

 **შენიშვნა:** Val AUC=1.0 — Pipeline leakage. CV AUC=0.951 სამართლიანი შეფასებაა.

---

## Neural Network (PyTorch FraudNet)

### არქიტექტურა

Custom PyTorch `FraudNet`: Linear → BatchNorm1d → ReLU → Dropout, სიღრმეში. Loss: BCEWithLogitsLoss, pos_weight=27.58 (imbalance კომპენსაცია). Optimizer: Adam + ReduceLROnPlateau scheduler. Early stopping patience=5.

### Baseline

`[256 → 128 → 64]`, dropout=0.3, lr=1e-3 → Train: 0.9261 | Val: **0.9071** | Gap: 0.019

### Architecture sweep

| არქიტექტურა | Val AUC | Gap |

| [128, 64] | 0.8983 | 0.0160 |
| [256, 128, 64] | 0.9067 | 0.0185 |
| [512, 256, 128, 64] | 0.9124 | 0.0224 |
| **[512, 512, 256]** | **0.9171** | **0.0267** |
| [256, 256, 256] | 0.9108 | 0.0209 |

### Dropout sweep

| dropout | Val AUC | Gap |

| **0.1** | **0.9288** | **0.0422** |
| 0.2 | 0.9212 | 0.0328 |
| 0.3 | 0.9153 | 0.0253 |
| 0.4 | 0.9070 | 0.0222 |
| 0.5 | 0.9010 | 0.0165 |

### Learning Rate sweep

| lr | Val AUC | Gap |

| 1e-2 | 0.9026 | 0.0204 |
| 5e-3 | 0.9173 | 0.0310 |
| **1e-3** | **0.9235** | **0.0381** |
| 5e-4 | 0.9227 | 0.0350 |
| 1e-4 | 0.9142 | 0.0300 |

### Overfitting vs Underfitting

| run | Train AUC | Val AUC | Gap | სტატუსი |

| NN_extreme_underfit | 0.8544 | 0.8511 | 0.0033 | good_fit |
| NN_underfit | 0.8733 | 0.8675 | 0.0058 | good_fit |
| NN_good_fit | 0.9692 | 0.9246 | 0.0446 | overfit |
| NN_overfit | 0.9792 | 0.9229 | 0.0563 | overfit |
| NN_severe_overfit | 0.9650 | 0.9218 | 0.0432 | overfit |

### Optuna (20 trials)

Best Val AUC: **0.9083**
Best arch: [512, 256, 128], dropout=0.248, lr=0.00051

### Cross-Validation (3-fold)

Fold 1: 0.9338, Fold 2: 0.9278, Fold 3: 0.9273 → **Mean: 0.9296 ± 0.003**

**Final Model:** arch=[512, 256, 128], dropout=0.248, lr=0.00051, 40 epochs → **Val AUC: 0.9206, CV: 0.9296 ± 0.003**

---

## Hyperparameter ოპტიმიზაციის შედარება (ყველა მოდელი)

| მოდელი | CV AUC | Final Val AUC | შენიშვნა |

| LogisticRegression | 0.7289 ± 0.0056 | 0.7289 | Clean — LR-ის ზედა ზღვარი |
| DecisionTree | 0.8493 ± 0.0020 | 0.861 | Clean |
| RandomForest | 0.8983 ± 0.0027 | 0.9183 (OOB) | Clean — Pipeline val=0.9877 (leakage) |
| XGBoost | 0.9493 ± 0.0022 | 0.9629 (Optuna) | Clean — Pipeline val=0.9999 (leakage) |
| LightGBM | 0.951 ± 0.001 | 0.9645 (Optuna) | Clean — Pipeline val=1.0 (leakage) |
| NeuralNetwork | 0.9296 ± 0.003 | 0.9206 | Clean |

### Pipeline Val AUC-ის leakage განმარტება

RF, XGBoost, LightGBM-ის "Final Pipeline Val AUC"-ები (0.9877, 0.9999, 1.0) Pipeline leakage-ით არის განპირობებული: Pipeline train+val-ს ერთად train-ავს, შემდეგ ამავე val-ზე ფასდება — ეს არ არის სამართლიანი შეფასება. **CV AUC სამართლიანი მეტრიკაა.**

---

## საბოლოო მოდელის შერჩევის დასაბუთება

**საუკეთესო მოდელი: LightGBM (Optuna Tuned)**

CV AUC-ის მიხედვით LightGBM (0.951) და XGBoost (0.9493) პრაქტიკულად ექვივალენტურია. LightGBM-ი გამარჯვებულია:
- **სწრაფი training** GPU-ზე (Tesla T4, histogram-based algorithm)
- **მაღალი CV AUC** 3-fold: 0.951 ± 0.001
- **GPU-სპეციფიკური ოპტიმიზაცია** (OpenCL kernel compilation)
- **მეხსიერების ეფექტური** დიდ dataset-ებზე

ყველა მოდელი შენახულია **sklearn Pipeline-ად** (`SimpleImputer → Scaler → Model`), რაც test set-ზე პირდაპირ გაშვების საშუალებას იძლევა preprocessing-ის გარეშე

![LogisticRegression ROC](images/lr_roc_curve.png)
![DecisionTree ROC](images/dt_roc_curve.png)
![RandomForest ROC](images/rf_roc_curve.png)
![XGBoost ROC](images/xgb_roc_curve.png)
![LightGBM ROC](images/lgbm_roc_curve.png)
![NeuralNetwork ROC](images/nn_roc_curve.png)
---

## MLflow Tracking

### MLflow ექსპერიმენტების ბმული

**DagsHub MLflow UI:** [https://dagshub.com/mkakh22/IEEE-CIS-Fraud-Detection.mlflow](https://dagshub.com/mkakh22/IEEE-CIS-Fraud-Detection.mlflow)

### ექსპერიმენტების სტრუქტურა

- Data cleaning & feature selection
- Hyperparameter tuning (manual + Optuna)
- Overfitting/underfitting analysis
- Cross-validation & final model

![MLflow Experiments](images/mlflow_experiments.png)

### ჩაწერილი მეტრიკები

ყველა run-ზე: `train_auc`, `val_auc`, `gap` (train-val სხვაობა), `fit_time_s`.
CV run-ებზე: `cv_auc_mean`, `cv_auc_std`, `fold_1_auc`...`fold_N_auc`.
XGBoost/LightGBM run-ებზე: `best_n_estimators` (early stopping).
FinalModel run-ებზე: `final_val_auc`, pipeline artifacts, model weights.

### Model Registry

| მოდელი | Registry Name | Version |

| LogisticRegression | `ieee-fraud-LogisticRegression` 
| DecisionTree | `ieee-fraud-DecisionTree` 
| RandomForest | `ieee-fraud-RandomForest` 
| XGBoost | `ieee-fraud-XGBoost`
| LightGBM | `ieee-fraud-LightGBM` 
| NeuralNetwork | `ieee-fraud-NeuralNetwork` 

**Model Registry-ში საუკეთესო მოდელი:** `ieee-fraud-LightGBM`

---

## შეფასება Kaggle-ზე

**Public Leaderboard Score:** *0.8386 (LightGBM)* 


CV AUC (0.951) და Public LB (0.8386) სხვაობის ანალიზი:

სხვაობა აიხსნება მონაცემთა დროითი სტრუქტურით:

Chronological Order: Training ტრანზაქციები წინ უსწრებს სატესტოებს.

K-Fold Shuffle: სტანდარტული K-Fold Cross-Validation მონაცემებს "ურევს" (shuffle), რის გამოც მოდელი ვალიდაციისას ხედავს "მომავლის" მონაცემებს. ეს იწვევს CV AUC-ის ხელოვნურ ზრდას (Inflation).

Real-world Scenario: Kaggle-ის Public LB არის მკაცრად training პერიოდის შემდგომი მონაცემები, რაც გაცილებით რთული და რეალისტური ამოცანაა.

მოდელი              | CV AUC | Final Val AUC
Logistic Regression | 0.7289 | 0.7289
RandomForest        | 0.8983 | 0.9183 (OOB)
NeuralNetwork       | 0.9296 | 0.9206
LightGBM            | 0.9510 | 0.9645 (Optuna)

![Kaggle Score](images/Score.jpeg)