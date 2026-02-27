# from flask import Flask, render_template, request, redirect, url_for, flash
# import pandas as pd
# import joblib
# import tensorflow as tf
# import numpy as np
# import pickle

# # ✅ Fix for NumPy / SHAP np.bool issue
# if not hasattr(np, "bool"):
#     np.bool = bool  # Use built-in bool

# import shap
# import os
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import traceback

# # ---------------------- Configuration ----------------------
# app = Flask(__name__)
# app.secret_key = 'your_secret_key_here'

# # Paths
# MODEL_PATHS = {
#     'cnn': 'saved_models/cnn_model.h5',
#     'transformer': 'saved_models/transformer_model.h5',
#     'hybrid': 'saved_models/xgb_hybrid.pkl',
#     'rf': 'saved_models/rf_model.pkl',
#     'dt': 'saved_models/dt_model.pkl',
#     'xgb': 'saved_models/xgb_model.pkl',
#     'scaler': 'saved_models/scaler.pkl',
#     'features': 'saved_models/feature_columns.pkl'
# }
# LOG_FILE = 'logs/predictions.csv'

# # ---------------------- Load Models ----------------------
# print("🔍 Loading models ...")

# rf_model = joblib.load(MODEL_PATHS['rf'])
# dt_model = joblib.load(MODEL_PATHS['dt'])
# xgb_model = joblib.load(MODEL_PATHS['xgb'])
# xgb_hybrid = joblib.load(MODEL_PATHS['hybrid'])
# scaler = joblib.load(MODEL_PATHS['scaler'])

# # ✅ Load features with pickle (not joblib)
# with open(MODEL_PATHS['features'], "rb") as f:
#     feature_columns = pickle.load(f)

# # ✅ CNN model (compile=False skips missing loss function issue)
# try:
#     cnn_model = tf.keras.models.load_model(
#         MODEL_PATHS['cnn'],
#         compile=False
#     )
#     print("✅ CNN model loaded successfully")
# except Exception as e:
#     cnn_model = None
#     print(f"❌ Failed to load CNN model: {e}")

# # ✅ Transformer model
# try:
#     transformer_model = tf.keras.models.load_model(
#         MODEL_PATHS['transformer'],
#         compile=False
#     )
#     print("✅ Transformer model loaded successfully")
# except Exception as e:
#     transformer_model = None
#     print(f"❌ Failed to load Transformer model: {e}")

# print("✅ All models loaded!")

# # ---------------------- Routes ----------------------
# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         input_data = request.form.to_dict()
#         if not input_data:
#             flash("⚠️ No input received.")
#             return redirect(url_for('index'))

#         df = pd.DataFrame([input_data])
#         df_encoded = pd.get_dummies(df)

#         # 🔑 Align with training features
#         df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)

#         # Tree model predictions
#         rf_pred = int(rf_model.predict(df_encoded)[0])
#         dt_pred = int(dt_model.predict(df_encoded)[0])
#         xgb_pred = int(xgb_model.predict(df_encoded)[0])

#         # Scale data for deep models
#         df_scaled = scaler.transform(df_encoded)

#         cnn_pred = transformer_pred = hybrid_pred = None
#         cnn_input = df_scaled.reshape(1, df_scaled.shape[1], 1)

#         # CNN Prediction
#         if cnn_model is not None:
#             cnn_prob = cnn_model.predict(cnn_input)[0][0]
#             cnn_pred = int(cnn_prob > 0.5)

#         # Transformer Prediction
#         if transformer_model is not None:
#             transformer_prob = transformer_model.predict(cnn_input)[0][0]
#             transformer_pred = int(transformer_prob > 0.5)

#         # Hybrid Prediction (fused CNN + Transformer features -> XGB)
#         if cnn_model is not None and transformer_model is not None:
#             cnn_features = cnn_model.predict(cnn_input)
#             transformer_features = transformer_model.predict(cnn_input)
#             fused_features = np.hstack([cnn_features, transformer_features])
#             hybrid_pred = int(xgb_hybrid.predict(fused_features)[0])

#         # Save log
#         save_prediction_log(input_data, rf_pred, dt_pred, xgb_pred,
#                             cnn_pred, transformer_pred, hybrid_pred)

#         # ---------------------- SHAP EXPLAINERS ----------------------
#         shap_rf = generate_shap_plot(rf_model, df_encoded, "rf")
#         shap_dt = generate_shap_plot(dt_model, df_encoded, "dt")
#         shap_xgb = generate_shap_plot(xgb_model, df_encoded, "xgb")

#         shap_cnn = generate_shap_deep_plot(cnn_model, cnn_input, "cnn") if cnn_model else None
#         shap_transformer = generate_shap_deep_plot(transformer_model, cnn_input, "transformer") if transformer_model else None
#         shap_hybrid = generate_shap_plot(xgb_hybrid, pd.DataFrame(fused_features), "hybrid") if (cnn_model and transformer_model) else None

#         return render_template(
#             'result.html',
#             rf=rf_pred,
#             dt=dt_pred,
#             xgb=xgb_pred,
#             cnn=cnn_pred,
#             transformer=transformer_pred,
#             hybrid=hybrid_pred,
#             shap_rf=shap_rf,
#             shap_dt=shap_dt,
#             shap_xgb=shap_xgb,
#             shap_cnn=shap_cnn,
#             shap_transformer=shap_transformer,
#             shap_hybrid=shap_hybrid
#         )

#     except Exception as e:
#         print("❌ Prediction Error:", traceback.format_exc())
#         flash(f"Prediction failed: {e}")
#         return redirect(url_for('index'))

# # ---------------------- SHAP Plot (Tree Models) ----------------------
# def generate_shap_plot(model, input_data, model_name):
#     try:
#         explainer = shap.TreeExplainer(model)
#         shap_values = explainer.shap_values(input_data)

#         os.makedirs('static', exist_ok=True)
#         plot_path = f'static/shap_{model_name}.png'

#         plt.figure()
#         shap.summary_plot(shap_values, input_data, show=False)
#         plt.savefig(plot_path, bbox_inches="tight")
#         plt.close()
#         return plot_path
#     except Exception as e:
#         print(f"⚠️ SHAP generation failed for {model_name}: {e}")
#         return None

# # ---------------------- SHAP Plot (Deep Models: CNN/Transformer) ----------------------
# def generate_shap_deep_plot(model, input_data, model_name):
#     try:
#         explainer = shap.DeepExplainer(model, input_data)
#         shap_values = explainer.shap_values(input_data)

#         os.makedirs('static', exist_ok=True)
#         plot_path = f'static/shap_{model_name}.png'

#         plt.figure()
#         shap.summary_plot(shap_values, input_data.squeeze(), show=False)
#         plt.savefig(plot_path, bbox_inches="tight")
#         plt.close()
#         return plot_path
#     except Exception as e:
#         print(f"⚠️ SHAP Deep generation failed for {model_name}: {e}")
#         return None

# # ---------------------- Logging ----------------------
# def save_prediction_log(data, rf, dt, xgb, cnn, transformer, hybrid):
#     row = data.copy()
#     row.update({
#         'rf': rf,
#         'dt': dt,
#         'xgb': xgb,
#         'cnn': cnn,
#         'transformer': transformer,
#         'hybrid': hybrid
#     })
#     df = pd.DataFrame([row])
#     os.makedirs('logs', exist_ok=True)

#     if os.path.exists(LOG_FILE):
#         df.to_csv(LOG_FILE, mode='a', header=False, index=False)
#     else:
#         df.to_csv(LOG_FILE, mode='w', header=True, index=False)

# # ---------------------- Run App ----------------------
# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, render_template, request, redirect, url_for, flash, session
import pandas as pd
import joblib
import tensorflow as tf
import numpy as np
import pickle
import shap
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import traceback
import mysql.connector
import json

# ✅ Fix for NumPy / SHAP np.bool issue
if not hasattr(np, "bool"):
    np.bool = bool

# ---------------------- Configuration ----------------------
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # session management

MODEL_PATHS = {
    'cnn': 'saved_models/cnn_model.h5',
    'transformer': 'saved_models/transformer_model.h5',
    'hybrid': 'saved_models/xgb_hybrid.pkl',
    'rf': 'saved_models/rf_model.pkl',
    'dt': 'saved_models/dt_model.pkl',
    'xgb': 'saved_models/xgb_model.pkl',
    'scaler': 'saved_models/scaler.pkl',
    'features': 'saved_models/feature_columns.pkl',
    'hybrid_features': 'saved_models/hybrid_feature_columns.pkl'
}

MODEL_DISPLAY = {
    "rf": "Random Forest",
    "dt": "Decision Tree",
    "xgb": "XGBoost",
    "cnn": "CNN",
    "transformer": "Transformer",
    "hybrid": "Hybrid (CNN+Transformer→XGB)"
}

# ---------------------- DB Connection ----------------------
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",        # change if needed
        password="",        # change if needed
        database="fraud_detection"
    )

# ---------------------- Load Models ----------------------
print("🔍 Loading models ...")
rf_model = joblib.load(MODEL_PATHS['rf'])
dt_model = joblib.load(MODEL_PATHS['dt'])
xgb_model = joblib.load(MODEL_PATHS['xgb'])
xgb_hybrid = joblib.load(MODEL_PATHS['hybrid'])
scaler = joblib.load(MODEL_PATHS['scaler'])

with open(MODEL_PATHS['features'], "rb") as f:
    feature_columns = pickle.load(f)
if "Fraud" in feature_columns:
    feature_columns = [c for c in feature_columns if c != "Fraud"]

# ✅ Load hybrid feature columns
try:
    with open(MODEL_PATHS['hybrid_features'], "rb") as f:
        hybrid_feature_columns = pickle.load(f)
except:
    hybrid_feature_columns = None

try:
    cnn_model = tf.keras.models.load_model(MODEL_PATHS['cnn'], compile=False)
    print("✅ CNN model loaded successfully")
except Exception as e:
    cnn_model = None
    print(f"❌ CNN load failed: {e}")

try:
    transformer_model = tf.keras.models.load_model(MODEL_PATHS['transformer'], compile=False)
    print("✅ Transformer model loaded successfully")
except Exception as e:
    transformer_model = None
    print(f"❌ Transformer load failed: {e}")

print("✅ All models loaded!")

# ---------------------- Routes ----------------------
@app.route('/')
def home():
    return redirect(url_for('register'))  # ✅ fixed BuildError

# ✅ Registration (with duplicate check)
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        existing_user = cursor.fetchone()

        if existing_user:
            flash("⚠️ Username already exists. Please choose a different one.")
            cursor.close()
            conn.close()
            return redirect(url_for('register'))

        cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, password))
        conn.commit()
        cursor.close()
        conn.close()

        flash("✅ Registration successful! Please log in.")
        return redirect(url_for('login'))

    return render_template('reg.html')

# ✅ Login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username == "admin" and password == "admin123":
            session['user'] = 'admin'
            session['role'] = 'admin'
            flash("✅ Admin login successful!")
            return redirect(url_for('admin_dashboard'))

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE username=%s AND password=%s", (username, password))
        user = cursor.fetchone()
        cursor.close()
        conn.close()

        if user:
            session['user'] = username
            session['role'] = 'user'
            flash("✅ User login successful!")
            return redirect(url_for('dashboard'))
        else:
            flash("❌ Invalid username or password. Try again.")

    return render_template('login.html')

# ✅ Logout
@app.route('/logout')
def logout():
    session.pop('user', None)
    session.pop('role', None)
    flash("✅ Logged out successfully.")
    return redirect(url_for('login'))

# ✅ User Dashboard
@app.route('/dashboard')
def dashboard():
    if 'user' not in session or session.get('role') != 'user':
        flash("⚠️ Please login as User.")
        return redirect(url_for('login'))
    return render_template('index.html', MODEL_DISPLAY=MODEL_DISPLAY)

# ✅ Admin Dashboard
@app.route('/admin')
def admin_dashboard():
    if 'user' not in session or session.get('role') != 'admin':
        flash("⚠️ Please login as Admin.")
        return redirect(url_for('login'))

    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute("SELECT id, username, model, prediction FROM predictions ORDER BY id DESC LIMIT 20")
        logs = cursor.fetchall()

        cursor.execute("SELECT COUNT(*) AS total, COALESCE(SUM(prediction), 0) AS frauds FROM predictions")
        stats = cursor.fetchone()

        cursor.execute("SELECT id, username, created_at FROM users ORDER BY id DESC")
        users = cursor.fetchall()

        cursor.close()
        conn.close()

        total = stats['total'] if stats['total'] else 0
        frauds = stats['frauds'] if stats['frauds'] else 0
        fraud_rate = round((frauds / total) * 100, 2) if total > 0 else 0.0
        fraud_counts = {0: total - frauds, 1: frauds}

        return render_template(
            "admin_dashboard.html",
            logs=logs,
            fraud_rate=fraud_rate,
            total=total,
            fraud_counts=fraud_counts,
            users=users
        )
    except Exception as e:
        print("❌ Dashboard Error:", traceback.format_exc())
        flash("Dashboard failed to load.")
        return redirect(url_for('login'))

# ✅ Download Users CSV
@app.route('/download_users')
def download_users():
    if 'user' not in session or session.get('role') != 'admin':
        flash("⚠️ Unauthorized access. Please login as Admin.")
        return redirect(url_for('login'))

    try:
        conn = get_db_connection()
        df = pd.read_sql("SELECT id, username, created_at FROM users", conn)
        conn.close()

        file_path = "static/users.csv"
        df.to_csv(file_path, index=False)

        flash("✅ Users list exported successfully.")
        return redirect(url_for('static', filename='users.csv'))
    except Exception as e:
        print("❌ Download Users Error:", e)
        flash("⚠️ Failed to export users.")
        return redirect(url_for('admin_dashboard'))

# ✅ Download Predictions CSV
@app.route('/download_predictions')
def download_predictions():
    if 'user' not in session or session.get('role') != 'admin':
        flash("⚠️ Unauthorized access. Please login as Admin.")
        return redirect(url_for('login'))

    try:
        conn = get_db_connection()
        df = pd.read_sql("SELECT id, username, model, prediction, user_input FROM predictions", conn)
        conn.close()

        file_path = "static/predictions.csv"
        df.to_csv(file_path, index=False)

        flash("✅ Predictions exported successfully.")
        return redirect(url_for('static', filename='predictions.csv'))
    except Exception as e:
        print("❌ Download Predictions Error:", e)
        flash("⚠️ Failed to export predictions.")
        return redirect(url_for('admin_dashboard'))

# ---------------------- Prediction ----------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'user' not in session:
            flash("⚠️ Please login first.")
            return redirect(url_for('login'))

        input_data = request.form.to_dict()
        selected_model = input_data.pop("selected_model", None)

        if not input_data or not selected_model:
            flash("⚠️ Please provide input and select a model.")
            return redirect(url_for('dashboard'))

        df = pd.DataFrame([input_data])
        df_encoded = pd.get_dummies(df)
        df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)
        df_encoded = df_encoded.loc[:, feature_columns]

        df_scaled = scaler.transform(df_encoded)
        cnn_input = df_scaled.reshape(1, df_scaled.shape[1], 1)

        prediction, shap_plot = None, None

        if selected_model == "rf":
            prediction = int(rf_model.predict(df_encoded)[0])
            shap_plot = generate_shap_tree_plot(rf_model, df_encoded, "rf")

        elif selected_model == "dt":
            prediction = int(dt_model.predict(df_encoded)[0])
            shap_plot = generate_shap_tree_plot(dt_model, df_encoded, "dt")

        elif selected_model == "xgb":
            prediction = int(xgb_model.predict(df_encoded)[0])
            shap_plot = generate_shap_tree_plot(xgb_model, df_encoded, "xgb")

        elif selected_model == "cnn":
            if not cnn_model:
                flash("⚠️ CNN model not available.")
                return redirect(url_for('dashboard'))
            prob = float(cnn_model.predict(cnn_input)[0][0])
            prediction = int(prob > 0.5)
            shap_plot = generate_shap_deep_plot(cnn_model, cnn_input, "cnn")

        elif selected_model == "transformer":
            if not transformer_model:
                flash("⚠️ Transformer model not available.")
                return redirect(url_for('dashboard'))
            prob = float(transformer_model.predict(cnn_input)[0][0])
            prediction = int(prob > 0.5)
            shap_plot = generate_shap_deep_plot(transformer_model, cnn_input, "transformer")

        elif selected_model == "hybrid":
            if not (cnn_model and transformer_model):
                flash("⚠️ Hybrid requires both CNN and Transformer models.")
                return redirect(url_for('dashboard'))

            cnn_features = cnn_model.predict(cnn_input)
            transformer_features = transformer_model.predict(cnn_input)

            # Use Medicare + CNN + Transformer features
            fused_features = np.hstack([df_scaled, cnn_features, transformer_features])

            if hybrid_feature_columns:
                fused_df = pd.DataFrame(fused_features, columns=hybrid_feature_columns)
            else:
                fused_df = pd.DataFrame(fused_features)

            prediction = int(xgb_hybrid.predict(fused_features)[0])
            shap_plot = generate_shap_tree_plot(xgb_hybrid, fused_df, "hybrid")

        else:
            flash(f"⚠️ Selected model not available. (Received: {selected_model})")
            return redirect(url_for('dashboard'))

        prediction_label = "Fraud" if prediction == 1 else "Non-Fraud"
        save_prediction_log(session['user'], input_data, selected_model, prediction)

        return render_template(
            'result.html',
            model_used=MODEL_DISPLAY[selected_model],
            prediction=prediction_label,
            shap_plot=shap_plot
        )

    except Exception as e:
        print("❌ Prediction Error:", traceback.format_exc())
        flash(f"Prediction failed: {e}")
        return redirect(url_for('dashboard'))

# ---------------------- SHAP Plot Functions ----------------------
def generate_shap_tree_plot(model, input_data, model_name):
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_data)

        os.makedirs('static', exist_ok=True)
        plot_path = f'static/shap_{model_name}.png'

        plt.figure()
        if isinstance(shap_values, list):
            shap.summary_plot(shap_values[0], input_data, feature_names=input_data.columns, show=False)
        else:
            shap.summary_plot(shap_values, input_data, feature_names=input_data.columns, show=False)

        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()
        return plot_path
    except Exception as e:
        print(f"⚠️ SHAP tree generation failed for {model_name}: {e}")
        return None

def generate_shap_deep_plot(model, input_data, model_name):
    try:
        background = np.repeat(input_data, 20, axis=0)
        explainer = shap.GradientExplainer(model, background)
        shap_values = explainer.shap_values(input_data)

        os.makedirs('static', exist_ok=True)
        plot_path = f'static/shap_{model_name}.png'

        plt.figure()
        flat_sv = np.array(shap_values).reshape(len(shap_values), -1)
        flat_x = input_data.reshape(input_data.shape[0], -1)

        shap.summary_plot(flat_sv, flat_x, feature_names=feature_columns, show=False)
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()
        return plot_path
    except Exception as e:
        print(f"⚠️ SHAP deep generation failed for {model_name}: {e}")
        return None

# ---------------------- Logging ----------------------
def save_prediction_log(username, data, model, pred):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO predictions (username, user_input, model, prediction) VALUES (%s, %s, %s, %s)",
            (username, json.dumps(data), model, pred)
        )
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print("⚠️ DB Log Save Error:", e)

# ---------------------- Run ----------------------
if __name__ == '__main__':
    app.run(debug=True)
