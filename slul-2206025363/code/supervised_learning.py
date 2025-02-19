import pandas as pd
import os
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import  LabelEncoder, StandardScaler
from sklearn.metrics import f1_score
from sklearn import svm
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, DotProduct
import xgboost as xgb

base_path = os.path.dirname(__file__)  
file_name = "Filedata Data Jumlah Penduduk Provinsi DKI Jakarta Berdasarkan Agama.csv"
file_path = os.path.join(base_path, file_name)
df = pd.read_csv(file_path)


#Hapus periode_data
df = df.drop(['periode_data'], axis=1)


#Label Encoding
encoder = LabelEncoder()
object_cols = df.select_dtypes(include=['object']).columns
for col in object_cols:
    df[col] = encoder.fit_transform(df[col]) 
X = df.drop('agama', axis=1)
y = df['agama']


#Hyperparameter Tuning RandomForestClassifier
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# kfold = KFold(n_splits=4, shuffle=True, random_state=42)

# def objective(trial):
#     n_estimators = trial.suggest_int("n_estimators", 100, 500, step=100)
#     max_depth = trial.suggest_int("max_depth", 10, 50, step=10)
#     class_weight = trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample"])

#     model_rf = RandomForestClassifier(
#         n_estimators=n_estimators,
#         max_depth=max_depth,
#         class_weight=class_weight,
#         random_state=42
#     )
#     model_rf.fit(X_train_scaled, y_train)

#     y_pred_rf = model_rf.predict(X_test_scaled)
#     f1_micro_rf = f1_score(y_test, y_pred_rf, average='macro')
    
#     return f1_micro_rf  

# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=100)

# best_params = study.best_params
# print("\nBest parameters found:", best_params)

# final_model = RandomForestClassifier(**best_params, random_state=42)
# final_model.fit(X_train_scaled, y_train)

# y_pred = final_model.predict(X_test_scaled)

# print("\nFinal Model Performance:")
# print(f"f1 macro: {f1_score(y_test, y_pred, average='macro'):.4f}")
# print(f"f1 micro: {f1_score(y_test, y_pred, average='micro'):.4f}")

# Best parameters found: {'n_estimators': 400, 'max_depth': 20, 'class_weight': 'balanced_subsample'}

# Final Model Performance:
# f1 macro: 0.6053
# f1 micro: 0.6070


#Hyperparameter Tuning SVM
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# kfold = KFold(n_splits=4, shuffle=True, random_state=42)

# def objective(trial):
#     C = trial.suggest_int("C", 100, 1000, step=100)  # Start from 1, step of 1
#     kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"])

#     model = svm.SVC(
#         C=C,
#         kernel=kernel
#     )
#     model.fit(X_train_scaled, y_train)

#     y_pred = model.predict(X_test_scaled)
#     f1_micro = f1_score(y_test, y_pred, average='macro')
    
#     return f1_micro  

# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=100)

# best_params = study.best_params
# print("\nBest parameters found:", best_params)

# final_model = svm.SVC(**best_params)
# final_model.fit(X_train_scaled, y_train)

# y_pred = final_model.predict(X_test_scaled)

# print("\nFinal Model Performance:")
# print(f"f1 macro: {f1_score(y_test, y_pred, average='macro'):.4f}")
# print(f"f1 micro: {f1_score(y_test, y_pred, average='micro'):.4f}")

# Best parameters found: {'C': 900, 'kernel': 'linear'}

# Final Model Performance:
# f1 macro: 0.4206
# f1 micro: 0.4465



#Hyperparameter Tuning XGBClassifier
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# kfold = KFold(n_splits=4, shuffle=True, random_state=42)

# def objective(trial):
#     n_estimators = trial.suggest_int("n_estimators", 100, 500, step=100)
#     max_depth = trial.suggest_int("max_depth", 10, 50, step=10)
#     subsample = trial.suggest_int("subsample", 0.5, 1.0)

#     model = xgb.XGBClassifier(
#         n_estimators=n_estimators,
#         max_depth=max_depth,
#         subsample=subsample,
#         random_state=42
#     )
#     model.fit(X_train_scaled, y_train)

#     y_pred = model.predict(X_test_scaled)
#     f1_macro = f1_score(y_test, y_pred, average='macro')
    
#     return f1_macro

# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=20)

# best_params = study.best_params
# print("\nBest parameters found:", best_params)

# final_model = xgb.XGBClassifier(**best_params, random_state=42)
# final_model.fit(X_train_scaled, y_train)

# y_pred = final_model.predict(X_test_scaled)

# print("\nFinal Model Performance:")
# print(f"f1 macro: {f1_score(y_test, y_pred, average='macro'):.4f}")
# print(f"f1 micro: {f1_score(y_test, y_pred, average='micro'):.4f}")

# Best parameters found: {'n_estimators': 100, 'max_depth': 10, 'subsample': 1}

# Final Model Performance:
# f1 macro: 0.6580
# f1 micro: 0.6591



#Train Test Model
f1_macro_mean_rf = [] 
f1_micro_mean_rf = []

f1_macro_mean_xgb = [] 
f1_micro_mean_xgb = []

f1_macro_mean_svm = [] 
f1_micro_mean_svm = []

for i in range(100):
    print(f"Loop ke: {i}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
        
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model_rf = RandomForestClassifier(n_estimators=400, random_state=42, max_depth=20, class_weight='balanced_subsample')
    model_rf.fit(X_train_scaled, y_train)

    y_pred_rf = model_rf.predict(X_test_scaled)
    f1_macro_rf = f1_score(y_test, y_pred_rf, average='macro')
    f1_micro_rf = f1_score(y_test, y_pred_rf, average='micro')
    f1_macro_mean_rf.append(f1_macro_rf)
    f1_micro_mean_rf.append(f1_micro_rf)

    model_xgb = xgb.XGBClassifier(n_estimators=100, max_depth=10, subsample=1)
    model_xgb.fit(X_train_scaled, y_train)

    y_pred_xgb = model_xgb.predict(X_test_scaled)
    f1_macro_xgb = f1_score(y_test, y_pred_xgb, average='macro')
    f1_micro_xgb = f1_score(y_test, y_pred_xgb, average='micro')
    f1_macro_mean_xgb.append(f1_macro_xgb)
    f1_micro_mean_xgb.append(f1_micro_xgb)

    model_svm = svm.SVC(C=900, kernel='linear')
    model_svm.fit(X_train_scaled, y_train)
    
    y_pred_svm = model_svm.predict(X_test_scaled)
    f1_macro_svm = f1_score(y_test, y_pred_svm, average='macro')
    f1_micro_svm = f1_score(y_test, y_pred_svm, average='micro')
    f1_macro_mean_svm.append(f1_macro_svm)
    f1_micro_mean_svm.append(f1_micro_svm)    


print(f"Rata-rata f1 macro dari Random Forest Classifier: {sum(f1_macro_mean_rf)/100:.4f}")
print(f"Rata-rata f1 micro dari Random Forest Classifier: {sum(f1_micro_mean_rf)/100:.4f}")

print(f"Rata-rata f1 macro dari XGBooster Classifier: {sum(f1_macro_mean_xgb)/100:.4f}")
print(f"Rata-rata f1 micro dari XGBooster Classifier: {sum(f1_micro_mean_xgb)/100:.4f}")

print(f"Rata-rata f1 macro dari SVM: {sum(f1_macro_mean_svm)/100:.4f}")
print(f"Rata-rata f1 micro dari SVM: {sum(f1_micro_mean_svm)/100:.4f}")

models = ["Random Forest", "XGBoost", "SVM"]

f1_macro_mean_rf_val = sum(f1_macro_mean_rf) / len(f1_macro_mean_rf)
f1_micro_mean_rf_val = sum(f1_micro_mean_rf) / len(f1_micro_mean_rf)

f1_macro_mean_xgb_val = sum(f1_macro_mean_xgb) / len(f1_macro_mean_xgb)
f1_micro_mean_xgb_val = sum(f1_micro_mean_xgb) / len(f1_micro_mean_xgb)

f1_macro_mean_svm_val = sum(f1_macro_mean_svm) / len(f1_macro_mean_svm)
f1_micro_mean_svm_val = sum(f1_micro_mean_svm) / len(f1_micro_mean_svm)

f1_macro_scores = [f1_macro_mean_rf_val, f1_macro_mean_xgb_val, f1_macro_mean_svm_val]
f1_micro_scores = [f1_micro_mean_rf_val, f1_micro_mean_xgb_val, f1_micro_mean_svm_val]

x = np.arange(len(models))
width = 0.3 

fig, ax = plt.subplots(figsize=(8, 6))
bars1 = ax.bar(x - width/2, f1_macro_scores, width, label="F1 Macro", color="royalblue")
bars2 = ax.bar(x + width/2, f1_micro_scores, width, label="F1 Micro", color="orange")

ax.set_xlabel("Models")
ax.set_ylabel("F1 Score")
ax.set_title("Comparison of F1 Macro and Micro Scores for Different Models")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}", 
            xy=(bar.get_x() + bar.get_width() / 2, height), 
            xytext=(0, 5), 
            textcoords="offset points",
            ha="center", va="bottom", fontsize=10, fontweight="bold")

plt.show()