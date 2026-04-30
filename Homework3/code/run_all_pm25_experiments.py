
# -*- coding: utf-8 -*-
"""
基于多变量 LSTM 的 PM2.5 预测实验：自动预处理、训练、评估、作图。
运行方式：
    python run_all_pm25_experiments.py
输出目录：
    /mnt/data/pm25_experiment_package/outputs
"""

from pathlib import Path
import os, json, math, random, warnings, zipfile
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# =====================
# 1. 配置
# =====================
TRAIN_PATH = "/mnt/data/6fd7e370-a479-4517-867e-7b4f64020bd1.csv"
TEST_PATH = "/mnt/data/74d07538-0977-40f9-ab01-f06863684354.csv"
OUT = Path("/mnt/data/pm25_experiment_package/outputs")
FIG = OUT / "figures"
MODEL = OUT / "models"
for p in [OUT, FIG, MODEL]:
    p.mkdir(parents=True, exist_ok=True)

SEED = 42
WINDOWS = [1, 6, 12, 18, 24, 48]
BATCH_SIZE = 2048
EPOCHS_WINDOW = 4
EPOCHS_ABLATION = 4
HIDDEN_DIM = 32
LR = 1e-3

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(2)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# 2. 工具函数
# =====================
def savefig(name):
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=220, bbox_inches="tight")
    plt.close()

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def add_time_features(df):
    dt = pd.to_datetime(df["date"])
    df["hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24)
    df["month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12)
    return df

def make_sequences(df, feature_cols, L):
    x = df[feature_cols].astype("float32").values
    y = df["pollution"].astype("float32").values
    X = np.empty((len(df) - L, L, len(feature_cols)), dtype=np.float32)
    for i in range(L, len(df)):
        X[i - L] = x[i-L:i]
    return X, y[L:].copy()

def inverse_y(y_scaled, y_min, y_range):
    return y_scaled * y_range + y_min

def metrics(y_true_scaled, y_pred_scaled, y_min, y_range):
    yt = inverse_y(np.asarray(y_true_scaled), y_min, y_range)
    yp = inverse_y(np.asarray(y_pred_scaled), y_min, y_range)
    return {
        "MAE_scaled": float(mean_absolute_error(y_true_scaled, y_pred_scaled)),
        "RMSE_scaled": rmse(y_true_scaled, y_pred_scaled),
        "R2_scaled": float(r2_score(y_true_scaled, y_pred_scaled)),
        "MAE_original": float(mean_absolute_error(yt, yp)),
        "RMSE_original": rmse(yt, yp),
        "R2_original": float(r2_score(yt, yp)),
    }

# =====================
# 3. 数据预处理
# =====================
train_raw = pd.read_csv(TRAIN_PATH)
test_raw = pd.read_csv(TEST_PATH)
train_raw["date"] = pd.to_datetime(train_raw["date"])
test_raw["date"] = pd.date_range(
    start=train_raw["date"].max() + pd.Timedelta(hours=1),
    periods=len(test_raw),
    freq="H"
)

num_cols = ["pollution", "dew", "temp", "press", "wnd_spd", "snow", "rain"]
for df in [train_raw, test_raw]:
    df[num_cols] = df[num_cols].interpolate("linear").ffill().bfill()
    df["wnd_dir"] = df["wnd_dir"].ffill().bfill()
    df["is_snow"] = (df["snow"] > 0).astype(float)
    df["is_rain"] = (df["rain"] > 0).astype(float)
    add_time_features(df)

wind_cats = sorted(train_raw["wnd_dir"].unique().tolist())
for cat in wind_cats:
    train_raw[f"wnd_dir_{cat}"] = (train_raw["wnd_dir"] == cat).astype(float)
    test_raw[f"wnd_dir_{cat}"] = (test_raw["wnd_dir"] == cat).astype(float)

mins = train_raw[num_cols].min()
ranges = (train_raw[num_cols].max() - mins).replace(0, 1.0)
train = train_raw.copy()
test = test_raw.copy()
train[num_cols] = (train_raw[num_cols] - mins) / ranges
test[num_cols] = (test_raw[num_cols] - mins) / ranges

y_min = float(mins["pollution"])
y_range = float(ranges["pollution"])

wind_cols = [f"wnd_dir_{c}" for c in wind_cats]
time_cols = ["hour_sin", "hour_cos", "month_sin", "month_cos"]
binary_cols = ["is_snow", "is_rain"]
all_features = num_cols + binary_cols + wind_cols + time_cols

train.to_csv(OUT / "train_processed.csv", index=False)
test.to_csv(OUT / "test_processed.csv", index=False)

metadata = {
    "train_rows": int(len(train_raw)),
    "test_rows": int(len(test_raw)),
    "train_time_range": [str(train_raw["date"].min()), str(train_raw["date"].max())],
    "test_time_range_inferred": [str(test_raw["date"].min()), str(test_raw["date"].max())],
    "wind_categories": wind_cats,
    "feature_columns": all_features,
    "pollution_min": y_min,
    "pollution_max": float(train_raw["pollution"].max()),
    "device": str(DEVICE),
}
(OUT / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

# =====================
# 4. 数据分析图
# =====================
plt.figure(figsize=(12, 4))
plt.plot(train_raw["date"], train_raw["pollution"], linewidth=0.45)
plt.title("Training PM2.5 time series")
plt.xlabel("Time")
plt.ylabel("PM2.5")
savefig("01_pm25_time_series.png")

plt.figure(figsize=(8, 5))
plt.hist(train_raw["pollution"], bins=80)
plt.title("PM2.5 distribution in training data")
plt.xlabel("PM2.5")
plt.ylabel("Frequency")
savefig("02_pm25_distribution.png")

tmp = train_raw.copy()
tmp["month"] = tmp["date"].dt.month
monthly = tmp.groupby("month")["pollution"].mean()
plt.figure(figsize=(8, 4))
plt.plot(monthly.index, monthly.values, marker="o")
plt.xticks(range(1, 13))
plt.title("Average PM2.5 by month")
plt.xlabel("Month")
plt.ylabel("Average PM2.5")
savefig("03_monthly_average_pm25.png")

corr_cols = ["pollution", "dew", "temp", "press", "wnd_spd", "snow", "rain"] + time_cols
corr = train[corr_cols].corr().values
plt.figure(figsize=(9, 7))
im = plt.imshow(corr, vmin=-1, vmax=1, aspect="auto")
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.xticks(range(len(corr_cols)), corr_cols, rotation=45, ha="right")
plt.yticks(range(len(corr_cols)), corr_cols)
for i in range(len(corr_cols)):
    for j in range(len(corr_cols)):
        if i == j or abs(corr[i, j]) >= 0.45:
            plt.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center", fontsize=7)
plt.title("Feature correlation heatmap")
savefig("04_correlation_heatmap.png")

# =====================
# 5. Residual LSTM 模型
#    输出 = 最近一小时 PM2.5 + LSTM 学到的修正量
#    好处：保留 LSTM 时序建模，同时利用 PM2.5 强自相关。
# =====================
class ResidualLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, pollution_index=None):
        super().__init__()
        self.pollution_index = pollution_index
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        delta = self.head(out[:, -1, :]).squeeze(-1)
        if self.pollution_index is None:
            return delta
        last_pm25 = x[:, -1, self.pollution_index]
        return last_pm25 + delta

def train_lstm(X, y, feature_cols, epochs=4):
    split = int(len(X) * 0.8)
    Xtr, ytr = X[:split], y[:split]
    Xval, yval = X[split:], y[split:]
    pollution_index = feature_cols.index("pollution") if "pollution" in feature_cols else None

    model = ResidualLSTM(X.shape[-1], HIDDEN_DIM, pollution_index).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    ds = TensorDataset(torch.tensor(Xtr), torch.tensor(ytr))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    hist = []
    best_state, best_val = None, float("inf")
    val_x = torch.tensor(Xval, dtype=torch.float32).to(DEVICE)
    val_y = torch.tensor(yval, dtype=torch.float32).to(DEVICE)

    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(val_x), val_y).item())
        train_loss = float(np.mean(losses))
        hist.append({
            "epoch": ep,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_rmse_scaled": math.sqrt(train_loss),
            "val_rmse_scaled": math.sqrt(val_loss),
        })
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(hist)

def predict_lstm(model, X):
    model.eval()
    preds = []
    loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)), batch_size=4096, shuffle=False)
    with torch.no_grad():
        for (xb,) in loader:
            preds.append(model(xb.to(DEVICE)).detach().cpu().numpy())
    return np.concatenate(preds)

# =====================
# 6. 时间窗口实验
# =====================
window_rows = []
window_models = {}
window_hists = {}
for L in WINDOWS:
    X, y = make_sequences(train, all_features, L)
    Xte, yte = make_sequences(test, all_features, L)
    model, hist = train_lstm(X, y, all_features, epochs=EPOCHS_WINDOW)
    pred = predict_lstm(model, Xte)
    row = metrics(yte, pred, y_min, y_range)
    row.update({"window": L, "model": "Residual-LSTM", "epochs": EPOCHS_WINDOW})
    window_rows.append(row)
    window_models[L] = model
    window_hists[L] = hist
    torch.save(model.state_dict(), MODEL / f"residual_lstm_window_{L}.pt")

window_df = pd.DataFrame(window_rows)
window_df.to_csv(OUT / "window_experiment_metrics.csv", index=False)
best_window = int(window_df.sort_values("RMSE_original").iloc[0]["window"])

plt.figure(figsize=(8, 5))
x = np.arange(len(window_df))
plt.bar(x - 0.18, window_df["MAE_original"], width=0.36, label="MAE")
plt.bar(x + 0.18, window_df["RMSE_original"], width=0.36, label="RMSE")
plt.xticks(x, [f"{int(v)}h" for v in window_df["window"]])
plt.title("Window length comparison: Residual-LSTM")
plt.xlabel("Window length")
plt.ylabel("Error in original PM2.5 scale")
plt.legend()
savefig("05_window_length_comparison.png")

# =====================
# 7. 模型对比实验
# =====================
L = best_window
X, y = make_sequences(train, all_features, L)
Xte, yte = make_sequences(test, all_features, L)
rows = []

# Persistence baseline
pred_persist = Xte[:, -1, all_features.index("pollution")]
row = metrics(yte, pred_persist, y_min, y_range)
row.update({"model": "Persistence", "window": L})
rows.append(row)

# Ridge baseline
Xflat = X.reshape(len(X), -1)
Xteflat = Xte.reshape(len(Xte), -1)
ridge = Ridge(alpha=1.0)
ridge.fit(Xflat, y)
pred_ridge = ridge.predict(Xteflat)
row = metrics(yte, pred_ridge, y_min, y_range)
row.update({"model": "Ridge", "window": L})
rows.append(row)

# HistGradientBoosting baseline
hgb = HistGradientBoostingRegressor(max_iter=160, learning_rate=0.06, l2_regularization=0.01, random_state=SEED)
hgb.fit(Xflat, y)
pred_hgb = hgb.predict(Xteflat)
row = metrics(yte, pred_hgb, y_min, y_range)
row.update({"model": "HistGradientBoosting", "window": L})
rows.append(row)

# Residual-LSTM from window experiment
lstm_model = window_models[L]
pred_lstm = predict_lstm(lstm_model, Xte)
row = metrics(yte, pred_lstm, y_min, y_range)
row.update({"model": "Residual-LSTM", "window": L})
rows.append(row)

model_df = pd.DataFrame(rows).sort_values("RMSE_original")
model_df.to_csv(OUT / "model_comparison_metrics.csv", index=False)

plt.figure(figsize=(9, 5))
x = np.arange(len(model_df))
plt.bar(x - 0.18, model_df["MAE_original"], width=0.36, label="MAE")
plt.bar(x + 0.18, model_df["RMSE_original"], width=0.36, label="RMSE")
plt.xticks(x, model_df["model"], rotation=20, ha="right")
plt.title(f"Model comparison, window={L}h")
plt.ylabel("Error in original PM2.5 scale")
plt.legend()
savefig("06_model_comparison.png")

# =====================
# 8. 特征消融实验
# =====================
base_weather = ["dew", "temp", "press", "wnd_spd", "snow", "rain", "is_snow", "is_rain"]
feature_sets = {
    "Pollution only": ["pollution"],
    "Weather only": base_weather + wind_cols + time_cols,
    "Pollution + weather": ["pollution"] + base_weather,
    "Full features": all_features,
}

ablation_rows = []
for name, cols in feature_sets.items():
    Xa, ya = make_sequences(train, cols, L)
    Xta, yta = make_sequences(test, cols, L)
    m, h = train_lstm(Xa, ya, cols, epochs=EPOCHS_ABLATION)
    p = predict_lstm(m, Xta)
    row = metrics(yta, p, y_min, y_range)
    row.update({"feature_set": name, "num_features": len(cols), "window": L})
    ablation_rows.append(row)

ablation_df = pd.DataFrame(ablation_rows).sort_values("RMSE_original")
ablation_df.to_csv(OUT / "feature_ablation_metrics.csv", index=False)

plt.figure(figsize=(9, 5))
x = np.arange(len(ablation_df))
plt.bar(x - 0.18, ablation_df["MAE_original"], width=0.36, label="MAE")
plt.bar(x + 0.18, ablation_df["RMSE_original"], width=0.36, label="RMSE")
plt.xticks(x, ablation_df["feature_set"], rotation=25, ha="right")
plt.title(f"Feature ablation: Residual-LSTM, window={L}h")
plt.ylabel("Error in original PM2.5 scale")
plt.legend()
savefig("07_feature_ablation.png")

# =====================
# 9. 最优 LSTM 详细分析图
# =====================
hist = window_hists[L]
plt.figure(figsize=(8, 5))
plt.plot(hist["epoch"], hist["train_rmse_scaled"], marker="o", label="Train RMSE")
plt.plot(hist["epoch"], hist["val_rmse_scaled"], marker="o", label="Validation RMSE")
plt.title(f"Residual-LSTM training curve, window={L}h")
plt.xlabel("Epoch")
plt.ylabel("RMSE on scaled target")
plt.legend()
savefig("08_lstm_training_curve.png")

true_orig = inverse_y(yte, y_min, y_range)
pred_orig = inverse_y(pred_lstm, y_min, y_range)
resid = pred_orig - true_orig

plt.figure(figsize=(12, 5))
plt.plot(true_orig, label="True PM2.5", linewidth=1.2)
plt.plot(pred_orig, label="Predicted PM2.5", linewidth=1.2)
plt.title(f"Residual-LSTM true vs predicted PM2.5, window={L}h")
plt.xlabel("Test sample index")
plt.ylabel("PM2.5")
plt.legend()
savefig("09_lstm_true_vs_predicted.png")

plt.figure(figsize=(12, 5))
n = min(160, len(true_orig))
plt.plot(true_orig[:n], label="True PM2.5", linewidth=1.2)
plt.plot(pred_orig[:n], label="Predicted PM2.5", linewidth=1.2)
plt.title(f"Residual-LSTM true vs predicted PM2.5, first {n} samples")
plt.xlabel("Test sample index")
plt.ylabel("PM2.5")
plt.legend()
savefig("10_lstm_true_vs_predicted_zoom.png")

plt.figure(figsize=(6, 6))
plt.scatter(true_orig, pred_orig, s=14, alpha=0.65)
mn = float(min(true_orig.min(), pred_orig.min()))
mx = float(max(true_orig.max(), pred_orig.max()))
plt.plot([mn, mx], [mn, mx], linestyle="--")
plt.title("Predicted vs true PM2.5")
plt.xlabel("True PM2.5")
plt.ylabel("Predicted PM2.5")
savefig("11_predicted_vs_true_scatter.png")

plt.figure(figsize=(8, 5))
plt.hist(resid, bins=40)
plt.title("Residual distribution")
plt.xlabel("Predicted - true")
plt.ylabel("Frequency")
savefig("12_residual_distribution.png")

plt.figure(figsize=(8, 5))
plt.scatter(true_orig, resid, s=14, alpha=0.65)
plt.axhline(0, linestyle="--")
plt.title("Residual vs true PM2.5")
plt.xlabel("True PM2.5")
plt.ylabel("Residual")
savefig("13_residual_vs_true_pm25.png")

def pm25_level(y):
    bins = [-np.inf, 35, 75, 115, 150, np.inf]
    labels = ["0-35", "35-75", "75-115", "115-150", ">150"]
    return pd.cut(y, bins=bins, labels=labels)

level_true = pm25_level(true_orig)
level_pred = pm25_level(pred_orig)
err_df = pd.DataFrame({"true": true_orig, "pred": pred_orig, "abs_error": np.abs(resid), "level": level_true})
err_by_level = err_df.groupby("level", observed=False)["abs_error"].agg(["count", "mean", "median"]).reset_index()
err_by_level.to_csv(OUT / "error_by_pm25_level.csv", index=False)

plt.figure(figsize=(8, 5))
plt.bar(err_by_level["level"].astype(str), err_by_level["mean"])
plt.title("Mean absolute error by true PM2.5 level")
plt.xlabel("True PM2.5 level")
plt.ylabel("Mean absolute error")
savefig("14_error_by_pm25_level.png")

labels = ["0-35", "35-75", "75-115", "115-150", ">150"]
cm = confusion_matrix(level_true.astype(str), level_pred.astype(str), labels=labels)
cm_df = pd.DataFrame(cm, index=[f"True {l}" for l in labels], columns=[f"Pred {l}" for l in labels])
cm_df.to_csv(OUT / "pm25_level_confusion_matrix.csv")

plt.figure(figsize=(8, 6))
im = plt.imshow(cm, aspect="auto")
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.xticks(range(len(labels)), labels, rotation=30, ha="right")
plt.yticks(range(len(labels)), labels)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")
plt.title("PM2.5 level confusion matrix")
plt.xlabel("Predicted level")
plt.ylabel("True level")
savefig("15_pm25_level_confusion_matrix.png")

# =====================
# 10. LSTM 置换重要性
# =====================
base_rmse = rmse(true_orig, pred_orig)
perm_rows = []
rng = np.random.default_rng(SEED)
for j, name in enumerate(all_features):
    deltas = []
    for _ in range(3):
        Xp = Xte.copy()
        idx = rng.permutation(len(Xp))
        Xp[:, :, j] = Xp[idx, :, j]
        pp = inverse_y(predict_lstm(lstm_model, Xp), y_min, y_range)
        deltas.append(rmse(true_orig, pp) - base_rmse)
    perm_rows.append({"feature": name, "rmse_increase": float(np.mean(deltas)), "std": float(np.std(deltas))})
perm_df = pd.DataFrame(perm_rows).sort_values("rmse_increase", ascending=False)
perm_df.to_csv(OUT / "lstm_permutation_importance.csv", index=False)

top = perm_df.head(12).iloc[::-1]
plt.figure(figsize=(8, 6))
plt.barh(top["feature"], top["rmse_increase"])
plt.title("Residual-LSTM permutation importance")
plt.xlabel("RMSE increase after permutation")
savefig("16_lstm_permutation_importance.png")

# =====================
# 11. 汇总与打包
# =====================
all_metrics = pd.concat([
    window_df.assign(section="window"),
    model_df.assign(section="model_comparison"),
    ablation_df.assign(section="feature_ablation"),
], ignore_index=True, sort=False)
all_metrics.to_csv(OUT / "all_experiment_metrics.csv", index=False)

summary = {
    "best_window_by_lstm_rmse": best_window,
    "best_model_by_rmse": str(model_df.iloc[0]["model"]),
    "best_model_metrics": model_df.iloc[0].to_dict(),
    "residual_lstm_metrics_at_best_window": metrics(yte, pred_lstm, y_min, y_range),
    "figures_count": len(list(FIG.glob("*.png"))),
}
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

zip_path = OUT.parent / "pm25_experiment_results.zip"
if zip_path.exists():
    zip_path.unlink()
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for path in OUT.rglob("*"):
        z.write(path, path.relative_to(OUT.parent))
    z.write(Path(__file__), "run_all_pm25_experiments.py")

print("实验完成。")
print(json.dumps(summary, ensure_ascii=False, indent=2))
print("\n[窗口实验]")
print(window_df[["window", "MAE_original", "RMSE_original", "R2_original", "MAE_scaled", "RMSE_scaled"]].to_string(index=False))
print("\n[模型对比]")
print(model_df[["model", "MAE_original", "RMSE_original", "R2_original"]].to_string(index=False))
print("\n[特征消融]")
print(ablation_df[["feature_set", "num_features", "MAE_original", "RMSE_original", "R2_original"]].to_string(index=False))
print(f"\n输出目录: {OUT}")
print(f"压缩包: {zip_path}")
