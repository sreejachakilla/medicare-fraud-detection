# import pandas as pd
# import numpy as np

# # ✅ Fix for deprecated np.bool
# if not hasattr(np, "bool"):
#     np.bool = bool

# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.utils.class_weight import compute_class_weight
# from imblearn.over_sampling import SMOTE
# import shap
# import matplotlib.pyplot as plt
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, BatchNormalization, Input, MultiHeadAttention, LayerNormalization
# from tensorflow.keras.callbacks import EarlyStopping
# import tensorflow_addons as tfa
# import joblib
# import os
# import pickle
# from xgboost import XGBClassifier

# # ---------------------- Create Save Directory ----------------------
# os.makedirs("saved_models", exist_ok=True)

# # ---------------------- Load Dataset ----------------------
# # ✅ Fix UTF-8 decoding issue → fall back to latin1 if needed
# try:
#     df = pd.read_csv("dataset/medicare_5000.csv", encoding="utf-8")
# except UnicodeDecodeError:
#     df = pd.read_csv("dataset/medicare_5000.csv", encoding="latin1")

# df = df.drop(columns=["Provider_ID"])
# df = pd.get_dummies(df)

# # ---------------------- Split X and y ----------------------
# X = df.drop("Fraud", axis=1)
# y = df["Fraud"]

# # ✅ Always overwrite feature columns file safely (use pickle not joblib)
# feature_columns_path = "saved_models/feature_columns.pkl"
# if os.path.exists(feature_columns_path):
#     os.remove(feature_columns_path)

# with open(feature_columns_path, "wb") as f:
#     pickle.dump(X.columns.tolist(), f)

# print(f"✅ Saved feature columns: {len(X.columns)} features → {feature_columns_path}")

# # ---------------------- Balance Dataset with SMOTE ----------------------
# smote = SMOTE(random_state=42)
# X_resampled, y_resampled = smote.fit_resample(X, y)

# # ---------------------- Train-test Split ----------------------
# X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# # ---------------------- Feature Scaling ----------------------
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # Save the scaler
# joblib.dump(scaler, "saved_models/scaler.pkl")

# # ---------------------- Class Weights ----------------------
# classes = np.unique(y_train)
# weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
# class_weights = dict(zip(classes, weights))

# # ---------------------- Decision Tree ----------------------
# dt_model = DecisionTreeClassifier(max_depth=10, min_samples_leaf=5, random_state=42)
# dt_model.fit(X_train, y_train)
# joblib.dump(dt_model, "saved_models/dt_model.pkl")
# print("\n📌 Decision Tree Accuracy:", round(accuracy_score(y_test, dt_model.predict(X_test)) * 100, 2), "%")

# # ---------------------- Random Forest ----------------------
# rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42)
# rf_model.fit(X_train, y_train)
# joblib.dump(rf_model, "saved_models/rf_model.pkl")
# print("\n📌 Random Forest Accuracy:", round(accuracy_score(y_test, rf_model.predict(X_test)) * 100, 2), "%")

# # ---------------------- XGBoost ----------------------
# xgb_model = XGBClassifier(
#     n_estimators=200,
#     learning_rate=0.05,
#     max_depth=6,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     use_label_encoder=False,
#     eval_metric='logloss',
#     random_state=42
# )
# xgb_model.fit(X_train, y_train)
# joblib.dump(xgb_model, "saved_models/xgb_model.pkl")
# print("\n📌 XGBoost Accuracy:", round(accuracy_score(y_test, xgb_model.predict(X_test)) * 100, 2), "%")

# # ---------------------- CNN Model ----------------------
# X_train_cnn = X_train_scaled.reshape(-1, X_train_scaled.shape[1], 1)
# X_test_cnn = X_test_scaled.reshape(-1, X_test_scaled.shape[1], 1)

# cnn_model = Sequential([
#     Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train_cnn.shape[1], 1)),
#     BatchNormalization(),
#     MaxPooling1D(pool_size=2),
#     Dropout(0.3),

#     Conv1D(128, kernel_size=3, activation='relu'),
#     BatchNormalization(),
#     MaxPooling1D(pool_size=2),
#     Dropout(0.3),

#     Conv1D(256, kernel_size=3, activation='relu'),
#     BatchNormalization(),
#     MaxPooling1D(pool_size=2),
#     Dropout(0.3),

#     Flatten(),
#     Dense(128, activation='relu'),
#     Dropout(0.5),
#     Dense(1, activation='sigmoid')
# ])

# cnn_model.compile(
#     optimizer='adam',
#     loss=tfa.losses.SigmoidFocalCrossEntropy(),
#     metrics=['accuracy']
# )

# early_stop = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

# cnn_model.fit(
#     X_train_cnn, y_train,
#     validation_split=0.2,
#     epochs=50,
#     batch_size=32,
#     verbose=1,
#     callbacks=[early_stop],
#     class_weight=class_weights
# )

# cnn_model.save("saved_models/cnn_model.h5")
# cnn_loss, cnn_acc = cnn_model.evaluate(X_test_cnn, y_test)
# print("\n✅ CNN Accuracy:", round(cnn_acc * 100, 2), "%")

# # ---------------------- Transformer Model ----------------------
# inputs = Input(shape=(X_train_cnn.shape[1], 1))
# x = Conv1D(64, 3, activation="relu")(inputs)
# attn_out = MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
# x = LayerNormalization()(attn_out)
# x = Flatten()(x)
# x = Dense(128, activation="relu")(x)
# outputs = Dense(1, activation="sigmoid")(x)

# transformer_model = tf.keras.Model(inputs, outputs)
# transformer_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# transformer_model.fit(
#     X_train_cnn, y_train,
#     validation_split=0.2,
#     epochs=30,
#     batch_size=32,
#     verbose=1
# )

# transformer_model.save("saved_models/transformer_model.h5")
# trans_loss, trans_acc = transformer_model.evaluate(X_test_cnn, y_test)
# print("\n✅ Transformer Accuracy:", round(trans_acc * 100, 2), "%")

# # ---------------------- Hybrid CNN + Transformer + XGBoost ----------------------
# cnn_features_train = cnn_model.predict(X_train_cnn)
# cnn_features_test = cnn_model.predict(X_test_cnn)

# transformer_features_train = transformer_model.predict(X_train_cnn)
# transformer_features_test = transformer_model.predict(X_test_cnn)

# fused_train = np.hstack([cnn_features_train, transformer_features_train])
# fused_test = np.hstack([cnn_features_test, transformer_features_test])

# xgb_hybrid = XGBClassifier(
#     n_estimators=300,
#     learning_rate=0.05,
#     max_depth=6,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     eval_metric="logloss",
#     random_state=42
# )
# xgb_hybrid.fit(fused_train, y_train)
# joblib.dump(xgb_hybrid, "saved_models/xgb_hybrid.pkl")

# hybrid_acc = accuracy_score(y_test, xgb_hybrid.predict(fused_test))
# print("\n🌟 Hybrid CNN+Transformer+XGBoost Accuracy:", round(hybrid_acc * 100, 2), "%")

import pandas as pd
import numpy as np

# ✅ Fix for deprecated np.bool
if not hasattr(np, "bool"):
    np.bool = bool

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, BatchNormalization, Input, MultiHeadAttention, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow_addons as tfa
import joblib
import os
import pickle
from xgboost import XGBClassifier

# ---------------------- Create Save Directory ----------------------
os.makedirs("saved_models", exist_ok=True)

# ---------------------- Load Dataset ----------------------
try:
    df = pd.read_csv("dataset/medicare_5000.csv", encoding="utf-8")
except UnicodeDecodeError:
    df = pd.read_csv("dataset/medicare_5000.csv", encoding="latin1")

df = df.drop(columns=["Provider_ID"])
df = pd.get_dummies(df)

# ---------------------- Split X and y ----------------------
X = df.drop("Fraud", axis=1)
y = df["Fraud"]

# ✅ Save feature names (only original Medicare features)
feature_columns_path = "saved_models/feature_columns.pkl"
if os.path.exists(feature_columns_path):
    os.remove(feature_columns_path)

with open(feature_columns_path, "wb") as f:
    pickle.dump(X.columns.tolist(), f)

print(f"✅ Saved feature columns: {len(X.columns)} features → {feature_columns_path}")

# ---------------------- Balance Dataset with SMOTE ----------------------
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# ---------------------- Train-test Split ----------------------
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# ---------------------- Feature Scaling ----------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, "saved_models/scaler.pkl")

# ---------------------- Class Weights ----------------------
classes = np.unique(y_train)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weights = dict(zip(classes, weights))

# ---------------------- Decision Tree ----------------------
dt_model = DecisionTreeClassifier(max_depth=10, min_samples_leaf=5, random_state=42)
dt_model.fit(X_train, y_train)
joblib.dump(dt_model, "saved_models/dt_model.pkl")
print("\n📌 Decision Tree Accuracy:", round(accuracy_score(y_test, dt_model.predict(X_test)) * 100, 2), "%")

# ---------------------- Random Forest ----------------------
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42)
rf_model.fit(X_train, y_train)
joblib.dump(rf_model, "saved_models/rf_model.pkl")
print("\n📌 Random Forest Accuracy:", round(accuracy_score(y_test, rf_model.predict(X_test)) * 100, 2), "%")

# ---------------------- XGBoost ----------------------
xgb_model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
xgb_model.fit(X_train, y_train)
joblib.dump(xgb_model, "saved_models/xgb_model.pkl")
print("\n📌 XGBoost Accuracy:", round(accuracy_score(y_test, xgb_model.predict(X_test)) * 100, 2), "%")

# ---------------------- CNN Model ----------------------
X_train_cnn = X_train_scaled.reshape(-1, X_train_scaled.shape[1], 1)
X_test_cnn = X_test_scaled.reshape(-1, X_test_scaled.shape[1], 1)

cnn_model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train_cnn.shape[1], 1)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    Conv1D(128, kernel_size=3, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    Conv1D(256, kernel_size=3, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

cnn_model.compile(
    optimizer='adam',
    loss=tfa.losses.SigmoidFocalCrossEntropy(),
    metrics=['accuracy']
)

early_stop = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

cnn_model.fit(
    X_train_cnn, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    verbose=1,
    callbacks=[early_stop],
    class_weight=class_weights
)

cnn_model.save("saved_models/cnn_model.h5")
cnn_loss, cnn_acc = cnn_model.evaluate(X_test_cnn, y_test)
print("\n✅ CNN Accuracy:", round(cnn_acc * 100, 2), "%")

# ---------------------- Transformer Model ----------------------
inputs = Input(shape=(X_train_cnn.shape[1], 1))
x = Conv1D(64, 3, activation="relu")(inputs)
attn_out = MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
x = LayerNormalization()(attn_out)
x = Flatten()(x)
x = Dense(128, activation="relu")(x)
outputs = Dense(1, activation="sigmoid")(x)

transformer_model = tf.keras.Model(inputs, outputs)
transformer_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

transformer_model.fit(
    X_train_cnn, y_train,
    validation_split=0.2,
    epochs=30,
    batch_size=32,
    verbose=1
)

transformer_model.save("saved_models/transformer_model.h5")
trans_loss, trans_acc = transformer_model.evaluate(X_test_cnn, y_test)
print("\n✅ Transformer Accuracy:", round(trans_acc * 100, 2), "%")

# ---------------------- Hybrid CNN + Transformer + XGBoost ----------------------
cnn_features_train = cnn_model.predict(X_train_cnn)
cnn_features_test = cnn_model.predict(X_test_cnn)

transformer_features_train = transformer_model.predict(X_train_cnn)
transformer_features_test = transformer_model.predict(X_test_cnn)

# ✅ Combine original Medicare features + CNN + Transformer outputs
fused_train = np.hstack([X_train_scaled, cnn_features_train, transformer_features_train])
fused_test = np.hstack([X_test_scaled, cnn_features_test, transformer_features_test])

# ✅ Update feature names for SHAP
hybrid_feature_names = X.columns.tolist() + ["CNN_output", "Transformer_output"]
with open("saved_models/hybrid_feature_columns.pkl", "wb") as f:
    pickle.dump(hybrid_feature_names, f)

xgb_hybrid = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)
xgb_hybrid.fit(fused_train, y_train)
joblib.dump(xgb_hybrid, "saved_models/xgb_hybrid.pkl")

hybrid_acc = accuracy_score(y_test, xgb_hybrid.predict(fused_test))
print("\n🌟 Hybrid CNN+Transformer+XGBoost Accuracy:", round(hybrid_acc * 100, 2), "%")
