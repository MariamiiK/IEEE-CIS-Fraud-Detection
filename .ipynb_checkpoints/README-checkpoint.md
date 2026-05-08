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

დროის პარამეტრები (tx_hour, is_night) იმიტომ დავამატე, რომ თაღლითური ქმედებები ხშირად ღამით, ავტომატიზირებული სკრიპტებით ხდება. რაც შეეხება Z-score-ს, თუ ადამიანი ჩვეულებრივ 10 ლარიან პროდუქტს ყიდულობს და უცებ მისი ბარათიდან (card1) 1000 ლარი გადის, ეს გადახრა სტანდარტული განაწილებიდან მოდელმა მაშინვე უნდა შენიშნოს.

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



---

## Feature Selection

გამოვიყენე სამი მიდგომა და შევადარე მათი შედეგები:

### 1. SelectKBest (f_classif)

სტატისტიკური ტესტი, რომელიც ANOVA F-value-ს იყენებს. სწრაფი, მაგრამ მხოლოდ **წრფივ** დამოკიდებულებებს ითვალისწინებს. გამოვიყენეთ Logistic Regression-ისთვის (linear model-ისთვის შესაფერისი). 
**შეირჩა 150 feature.**


### 2. RandomForest Feature Importance (საუკეთესო)

RandomForest მოდელი, რომელიც impurity reduction-ის მიხედვით აფასებს მახასიათებლებს. ითვალისწინებს **არაწრფივ** დამოკიდებულებებს. სიჩქარის გასაზრდელად გამოვიყენეთ **10% stratified subsample** და გავწვრთენით **small RF**. **შეირჩა top 150 feature.** 
ექსპერიმენტებმა აჩვენა, რომ ამაზე მეტი ფუნქციის დატოვება იწვევდა 'Dimensionality Curse'-ს — მოდელი ზედმეტად დეტალიზებულ 'ხმაურს' სწავლობდა და ვეღარ ახდენდა გენერალიზაციას 

გამოყენება: DT, RF, XGBoost, LightGBM, NeuralNetwork.

ყველაზე მნიშვნელოვანი feature-ები RF-ის მიხედვით: **V244, V201, V259, V200, V258, C8, C11, V243, V187, C13** (V-სერიის ანონიმური feature-ები და C-სერიის counting feature-ები დომინირებს).

### 3. Correlation with Target

Pearson კორელაცია target-თან. სწრაფია, მაგრამ მხოლოდ monotonic კავშირებს ხედავს. შედარებისთვის გამოვიყენე.
გამოირიცხა მულტიკოლინეარული მახასიათებლები, რომელთა კორელაცია ერთმანეთთან $> 0.95$ იყო, რაც დაეხმარა მოდელების სტაბილურობას.


---

## Training

### მონაცემების გაყოფა

```
80% Train | 20% Validation
StratifiedKFold (3 ან 5-fold) cross-validation
```

StratifiedKFold კრიტიკულია **27.6:1 class imbalance-ის** გამო, ის უზრუნველყოფს fraud შემთხვევების თანაბარ განაწილებას ყველა fold-ში.

### ტესტირებული მოდელები


| # | მოდელი                | Notebook                                  | Experiment Name                |
|---|-----------------------|-------------------------------------------|--------------------------------|
| 1 | Logistic Regression   | model_experiment_LogisticRegression.ipynb | LogisticRegression_Training    |
| 2 | Decision Tree         | model_experiment_DecisionTree.ipynb       | DecisionTree_Training          |
| 3 | Random Forest         | model_experiment_RandomForest.ipynb       | RandomForest_Training          |
| 4 | XGBoost               | model_experiment_XGBoost.ipynb            | XGBoost_Training               |
| 5 | LightGBM              | model_experiment_LightGBM.ipynb           | LightGBM_Training              |
| 6 | Neural Network        | model_experiment_NeuralNetwork.ipynb      | NeuralNetwork_Training         |



## Logistic Regression

baseline-ზე C=1.0-ით val AUC 0.74 იყო — და რაც C-ს ვცვლიდი, შედეგი თითქმის არ იცვლებოდა. ეს მაშინვე მაჩვენა რომ პრობლემა პარამეტრებში კი არა, მოდელის სიმარტივეში იყო

### Hyperparameter ოპტიმიზაცია
მთავარი აქცენტი C (Inverse Regularization) პარამეტრზე გადავიტანე.
ვცადე C მნიშვნელობები 0.001-დან 10.0-მდე.

მინდოდა მენახა, მოდელი მონაცემების "ხმაურს" ხომ არ მიჰყვებოდა. აღმოჩნდა, რომ C=0.01 (ძლიერი რეგულარიზაცია) საუკეთესო იყო. ამან დამიდასტურა, რომ რადგან მოდელი მარტივია, მას სჭირდება "დამუხრუჭება", რათა ვალიდაციაზე უკეთესი შედეგი აჩვენოს

**C (Regularization):**

| C Value | Train AUC | Validation AUC | Gap (Val - Train) |
|--------|----------|---------------|------------------|
| 0.001  | 0.7252   | 0.7487        | -0.0235          |
| 0.01   | 0.7253   | **0.7497**    | -0.0244          |
| 0.1    | 0.7254   | 0.7492        | -0.0239          |
| 1.0    | 0.7252   | 0.7493        | -0.0241          |
| 10.0   | 0.7252   | 0.7493        | -0.0241          |

საუკეთესო C: **0.01**. შენიშვნა: უარყოფითი gap (val > train) განპირობებულია ძლიერი რეგულარიზაციით,
რომელიც ზღუდავს (constrain) მოდელის სირთულეს სატრენინგო მონაცემებზე, რათა გაუმჯობესდეს მისი განზოგადების უნარი

**Penalty:**

| Penalty | Train AUC | Validation AUC | Zero Coefficients |
|--------|----------|---------------|-------------------|
| L1     | 0.7251   | 0.7476        | 15                |
| L2     | 0.7256   | **0.7494**    | 0                 |

საუკეთესო: **L2**

**Cross-Validation (3-fold):** Fold 1: 0.7328, Fold 2: 0.7329, Fold 3: 0.7209 → **Mean: 0.7289 ± 0.0056**

**Final Model:** C=0.01, L2, lbfgs, class_weight=balanced → **Val AUC: 0.72, CV AUC: 0.7289 ± 0.0056**


---

## Decision Tree

depth=1-დან დავიწყე და თანდათან გავზარდე — depth=8 იყო ის წერტილი სადაც train/val gap ჯერ კიდევ გონივრული იყო.

### Overfitting vs Underfitting ანალიზი

| Max Depth | Train AUC | Validation AUC | Gap   | Status              |
|----------|----------|---------------|------|---------------------|
| 1        | 0.6277   | 0.6281        | -0.0004 | Underfitting        |
| 4        | 0.7940   | 0.7961        | -0.0021 | Good Fit            |
| 8        | **0.8605** | **0.8485**  | 0.0120 |  Best Trade-off   |
| 20       | 0.9474   | 0.8548        | 0.0926 | Overfitting         |
| None     | 0.9667   | 0.8480        | 0.1187 | Severe Overfitting  |

depth=1 → ძალიან მარტივი, ვერ სწავლობს (underfit). depth=None → ყველა training sample-ი ზეპირდება (severe overfit, gap=0.1187).

### Hyperparameter ოპტიმიზაცია

ვცადე: max_depth 1-დან 20-მდე და min_samples_split

როდესაც სიღრმე შეზღუდული არ მქონდა, მოდელმა ტრენინგზე თითქმის 100% აიღო, მაგრამ ვალიდაციაზე ჩავარდა. max_depth=8 აღმოჩნდა ის წერტილი, სადაც ხე საკმარისად ღრმაა კანონზომიერებების საპოვნელად, მაგრამ არა იმ დონეზე, რომ ინდივიდუალური ტრანზაქციების დაზეპირება დაიწყოს

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

საწყის baseline-ზე train AUC=1.0 იყო — overfitting-ის აშკარა ნიშანი, თუმცა OOB და Val 0.92-ზე რჩებოდა, რაც bagging-ის ეფექტს ადასტურებს. შემდეგ ვცადე სხვადასხვა პარამეტრები:

### n_estimators sweep

| Trees (n_estimators) | OOB AUC | Validation AUC | Gap   |
|---------------------|--------|---------------|------|
| 10                  | 0.7992 | 0.8677        | 0.1323 |
| 30                  | 0.8702 | 0.9062        | 0.0938 |
| 50                  | 0.8893 | 0.9161        | 0.0839 |
| 100                 | **0.9090** | **0.9238** | 0.0762 |

ექსპერიმენტმა აჩვენა, რომ 100 ხის შემდეგ შედეგი აღარ იზრდებოდა, მხოლოდ გამოთვლითი დრო იმატებდა, ამიტომ ეფექტურობისთვის აქ გავჩერდი.

### max_depth sweep

| max_features | Validation AUC | Gap   |
|-------------|---------------|------|
| sqrt        | 0.9092        | 0.0714 |
| log2        | 0.9012        | 0.0677 |
| 0.2         | 0.9119        | 0.0755 |
| 0.4         | **0.9137**    | 0.0751 |

### max_features sweep

ჩვეულებრივ sqrt გამოიყენება, მაგრამ მე ვცადე 40% (0.4), რადგან 400-ზე მეტი მახასიათებელი გვქონდა და მინდოდა თითოეულ ხეს მეტი ინფორმაცია ჰქონოდა გადაწყვეტილების მისაღებად. ამან საგრძნობლად გააუმჯობესა AUC

| max_features | Val AUC | Gap   |
|--------------|---------|-------|
| sqrt         | 0.9092  | 0.0714|
| log2         | 0.9012  | 0.0677|
| 0.2          | 0.9119  | 0.0755|
| **0.4**      | **0.9137** | **0.0751** |

### class_weight sweep

| class_weight        | Val AUC | Gap   |
|---------------------|---------|-------|
| none                | 0.8923  | 0.0210|
| **balanced**        | **0.9137** | **0.0751** |
| balanced_subsample  | 0.9132  | 0.0762|

### Overfitting vs Underfitting

| depth | Train AUC | OOB AUC | Val AUC | Gap    | სტატუსი  |
|-------|-----------|---------|---------|--------|-----------|
| 2     | 0.7637    | 0.7343  | 0.7689  | -0.0052| underfit  |
| 5     | 0.8489    | 0.8457  | 0.8460  | 0.0029 | good_fit  |
| 20    | 0.9888    | 0.9029  | 0.9137  | 0.0751 | overfit   |
| None  | 1.0000    | 0.8962  | 0.9218  | 0.0782 | overfit   |

RF-ის bagging-ი ამცირებს variance-ს: depth=None-ზეც კი train/val gap (0.078) გაცილებით მცირეა ვიდრე single DT-ში (0.1187).

**Cross-Validation (3-fold, n=50 per fold):**
Fold 1: 0.8964, Fold 2: 0.8964, Fold 3: 0.9021 → **Mean: 0.8983 ± 0.0027**

**Final Model:** n=100, depth=20, max_features=0.4, balanced → **Final Pipeline Val AUC: 0.9877, OOB: 0.9183, CV: 0.8983 ± 0.0027**


![RF Overfitting Analysis](images/rf_overfit_analysis.png)
---

## XGBoost

baseline-ზე val AUC 0.9512 — უკვე კარგი შედეგი, მაგრამ train=0.9861 gap-ს აჩვენებდა. early stopping best_n=999 იყო, რაც მეტი ტრენინგის საჭიროებაზე მიუთითებდა.

### learning_rate sweep

| Learning Rate | Validation AUC | Gap   |
|--------------|---------------|------|
| 0.30         | 0.9564        | 0.0434 |
| 0.10         | **0.9580**    | 0.0386 |
| 0.05         | 0.9512        | 0.0349 |
| 0.01         | 0.9141        | 0.0162 |

### max_depth sweep

| Max Depth | Validation AUC | Gap   |
|----------|---------------|------|
| 3        | 0.9262        | 0.0179 |
| 5        | 0.9516        | 0.0360 |
| 6        | 0.9580        | 0.0386 |
| 7        | 0.9601        | 0.0395 |
| 9        | **0.9623**    | 0.0377 |

### subsample × colsample_bytree

| sub  | col  | Val AUC | Gap   |
|------|------|---------|-------|
| 0.6  | 0.6  | 0.9598  | 0.0402|
| 0.6  | 0.8  | 0.9611  | 0.0389|
| **0.8** | **0.6** | **0.9625** | **0.0375** |
| 0.8  | 0.8  | 0.9623  | 0.0377|
| 1.0  | 1.0  | 0.9552  | 0.0442|

ამ პარამეტრების 0.7-0.8 მნიშვნელობებზე დაყენებით, მოდელს ვაიძულებდი, რომ ყოველ იტერაციაზე მონაცემების მხოლოდ შემთხვევითი შერჩევით დაემუშავებინა. ეს მეთოდი ხელს უშლის მოდელს კონკრეტულ მახასიათებლებზე დამოკიდებულების ჩამოყალიბებაში და ზრდის მის მდგრადობას ახალი, უცნობი მონაცემების მიმართ

### scale_pos_weight

| spw   | Val AUC |
|-------|---------|
| **1.0** | **0.9648** |
| 27.58 | 0.9625  |
 
### Overfitting vs Underfitting

| run                    | Train AUC | Val AUC | Gap    | სტატუსი  |
|------------------------|-----------|---------|--------|-----------|
| XGB_extreme_underfit   | 0.8411    | 0.8380  | 0.0031 | good_fit  |
| XGB_underfit           | 0.8742    | 0.8726  | 0.0016 | good_fit  |
| **XGB_good_fit**       | **0.9999**| **0.9647** | **0.0352** | **good_fit** |
| XGB_overfit_deep       | 1.0000    | 0.9568  | 0.0432 | overfit   |
| XGB_severe_overfit     | 1.0000    | 0.9502  | 0.0498 | overfit   |

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

baseline val AUC მხოლოდ 0.84 იყო — early stopping პირველ iteration-ზე გაჩერდა (best_n=1), რაც learning rate-ის პრობლემაზე მიუთითებდა. lr=0.30-ზე გადასვლის შემდეგ 0.95-მდე ავიდა.

### learning_rate sweep

| Learning Rate | Validation AUC | Gap   |
|--------------|---------------|------|
| 0.30         | **0.9542**    | 0.0440 |
| 0.10         | 0.8397        | 0.0019 |
| 0.05         | 0.8397        | 0.0019 |
| 0.01         | 0.8731        | 0.0029 |

### num_leaves sweep

| num_leaves | Val AUC | Gap   |
|------------|--------:|------:|
| 15         | 0.9408  | 0.0425 |
| 31         | 0.9562  | 0.0432 |
| 63         | 0.9581  | 0.0419 |
| **127**    | **0.9592** | **0.0408** |
| 255        | 0.9553  | 0.0447 |

### subsample × colsample_bytree

| sub  | col  | Val AUC | Gap   |
|------|------|--------:|------:|
| 0.6  | 0.6  | 0.9603  | 0.0397 |
| 0.6  | 0.8  | 0.9581  | 0.0419 |
| **0.8** | **0.6** | **0.9605** | **0.0395** |
| 0.8  | 0.8  | 0.9590  | 0.0410 |
| 1.0  | 1.0  | 0.9556  | 0.0444 |

### Optuna (20 trials)

Best Val AUC: **0.9645**
Best params: lr=0.1876, num_leaves=183, subsample=0.985, colsample=0.709, reg_alpha=0.00043, reg_lambda=2.53e-05, min_child_samples=90

### Cross-Validation (3-fold)

Fold 1: 0.9497, Fold 2: 0.9503, Fold 3: 0.9519 → **Mean: 0.951 ± 0.001**

**Final Model:** Optuna best params → **Final Val AUC: 1.0, CV: 0.951 ± 0.001**


---

## Neural Network (PyTorch FraudNet)

### არქიტექტურა

Custom PyTorch `FraudNet`: Linear → BatchNorm1d → ReLU → Dropout, სიღრმეში. Loss: BCEWithLogitsLoss, pos_weight=27.58 (imbalance კომპენსაცია). Optimizer: Adam + ReduceLROnPlateau scheduler. Early stopping patience=5.

baseline არქიტექტურა [256→128→64]-ით val AUC 0.907 გამოვიდა — გონივრული საწყისი წერტილი, მაგრამ gap=0.019 მეტი regularization-ის საჭიროებაზე მიუთითებდა.

### Architecture sweep

| არქიტექტურა        | Validation AUC | Gap   |
|-------------------|---------------|------|
| [128, 64]         | 0.8983        | 0.0160 |
| [256, 128, 64]    | 0.9067        | 0.0185 |
| [512, 256, 128, 64] | 0.9124     | 0.0224 |
| [512, 512, 256]   | **0.9171**    | 0.0267 |
| [256, 256, 256]   | 0.9108        | 0.0209 |

დავიწყე პატარა ქსელით, მაგრამ მონაცემების სირთულემ მოითხოვა უფრო ფართო შრეები თავიდანვე (512 ნეირონი), რათა 400+ ფუნქციის კომბინაციები დაეჭირა

### Dropout sweep

| dropout | Val AUC | Gap   |
|---------|--------:|------:|
| **0.1** | **0.9288** | **0.0422** |
| 0.2     | 0.9212  | 0.0328 |
| 0.3     | 0.9153  | 0.0253 |
| 0.4     | 0.9070  | 0.0222 |
| 0.5     | 0.9010  | 0.0165 |

Dropout-ის გარეშე ქსელი სწრაფად დაიწყებდა სატრენინგო მონაცემების დაზეპირებას. 0.1 მნიშვნელობამ მოგვცა საუკეთესო Val AUC (0.9288), თუმცა Optuna-მ საბოლოოდ 0.248 შეარჩია, რადგან კომბინირებულ ოპტიმიზაციაში უკეთ გენერალიზდება

### Learning Rate sweep

| lr    | Val AUC | Gap   |
|-------|--------:|------:|
| 1e-2  | 0.9026  | 0.0204 |
| 5e-3  | 0.9173  | 0.0310 |
| **1e-3** | **0.9235** | **0.0381** |
| 5e-4  | 0.9227  | 0.0350 |
| 1e-4  | 0.9142  | 0.0300 |

### Overfitting vs Underfitting

| run                | Train AUC | Val AUC | Gap   | სტატუსი      |
|--------------------|----------:|--------:|------:|-------------|
| NN_extreme_underfit| 0.8544    | 0.8511  | 0.0033 | good_fit    |
| NN_underfit        | 0.8733    | 0.8675  | 0.0058 | good_fit    |
| NN_good_fit        | 0.9692    | 0.9246  | 0.0446 | overfit     |
| NN_overfit         | 0.9792    | 0.9229  | 0.0563 | overfit     |
| NN_severe_overfit  | 0.9650    | 0.9218  | 0.0432 | overfit     |

### Optuna (20 trials)

Best Val AUC: **0.9083**
Best arch: [512, 256, 128], dropout=0.248, lr=0.00051

### Cross-Validation (3-fold)

Fold 1: 0.9338, Fold 2: 0.9278, Fold 3: 0.9273 → **Mean: 0.9296 ± 0.003**

**Final Model:** arch=[512, 256, 128], dropout=0.248, lr=0.00051, 40 epochs → **Val AUC: 0.9206, CV: 0.9296 ± 0.003**

---

## Hyperparameter ოპტიმიზაციის შედარება (ყველა მოდელი)

| Model              | CV AUC              | Final Validation AUC | შენიშვნა                              |
|-------------------|---------------------|----------------------|------------------------------------|
| LogisticRegression | 0.7289 ± 0.0056    | 0.7289               | Baseline                           |
| Decision Tree      | 0.8493 ± 0.0020    | 0.861                | Stable                             |
| Random Forest      | 0.8983 ± 0.0027    | 0.9183 (OOB)         | Leakage in pipeline validation     |
| XGBoost            | 0.9493 ± 0.0022    | 0.9629               | Strong, slight leakage             |
| LightGBM           | **0.951 ± 0.001**  | **0.9645**           | Best Model                         |
| Neural Network     | 0.9296 ± 0.003     | 0.9206               | Good but less stable               |


---
## საბოლოო მოდელის შერჩევის დასაბუთება

**საუკეთესო მოდელი: LightGBM (Optuna Tuned)**

CV AUC-ის მიხედვით LightGBM (0.951) და XGBoost (0.9493) პრაქტიკულად ექვივალენტურია. 
LightGBM ავარჩიე რამდენიმე მიზეზით: CV AUC-ით XGBoost-ს უსწრებს (0.951 vs 0.9493), GPU-ზე (Tesla T4) მნიშვნელოვნად სწრაფია histogram-based algorithm-ის გამო, და დიდ dataset-ებზე მეხსიერებას ეფექტურად იყენებს.

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


### ჩაწერილი მეტრიკები

ყველა run-ზე train/val AUC და gap ვლოგავდი. CV run-ებზე fold-ების შედეგები ცალ-ცალკე, XGBoost და LightGBM-ზე კი best_n_estimators early stopping-იდან.


### Model Registry

| მოდელი              | Registry Name                         |
|---------------------|---------------------------------------|
| LogisticRegression  | `ieee-fraud-LogisticRegression`       |
| DecisionTree        | `ieee-fraud-DecisionTree`             |
| RandomForest        | `ieee-fraud-RandomForest`             |
| XGBoost             | `ieee-fraud-XGBoost`                  |
| LightGBM            | `ieee-fraud-LightGBM`                 |
| NeuralNetwork       | `ieee-fraud-NeuralNetwork`            |

**Model Registry-ში საუკეთესო მოდელი:** `ieee-fraud-LightGBM`

---

## შეფასება Kaggle-ზე

**Public Leaderboard Score:** *0.8386 (LightGBM)* 


CV AUC (0.951) და Public LB (0.8386) სხვაობის ანალიზი:

სხვაობა აიხსნება იმით, რომ K-Fold shuffle-ით მოდელი ვალიდაციაში "მომავლის" მონაცემებს ხედავს, რაც CV AUC-ს ხელოვნურად ზრდის. Kaggle-ის test set კი მკაცრად training პერიოდის შემდეგია


![Kaggle Score](images/Score.jpeg)