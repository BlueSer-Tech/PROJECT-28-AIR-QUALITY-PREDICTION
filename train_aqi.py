import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
import joblib
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

# Đọc dữ liệu từ tệp CSV
data = pd.read_csv("HCM_moi.csv", na_values='=')

# # Tách features và target
X = data[['day', 'month', 'year', 'PM25']]  # Features
y = data['_pm25']  # Target AQI thay vì AQI Range

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
x_train_v, x_val, y_train_v, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

# Xây dựng pipeline cho MLPRegressor
mlp_regressor = Pipeline([
    ('scaler', StandardScaler()),  # Chuẩn hóa dữ liệu
    ('mlp', MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam', random_state=0, max_iter=1000))  # Mô hình MLP
])

best_val_err = float('inf')
best_alpha = None
best_hls = None

alphas = [0.001, 0.01, 0.1, 1, 10, 20, 100]
hidden_layer_sizes = [1, 10, 20, 50, 70, 100]

for alpha in alphas:
    for hls in hidden_layer_sizes:
        print(f"alpha = {alpha}, hidden_layer_sizes = {hls}")
        mlp_regressor.set_params(mlp__alpha=alpha, mlp__hidden_layer_sizes=hls)
        mlp_regressor.fit(x_train_v, y_train_v)
        val_err = mean_squared_error(y_val, mlp_regressor.predict(x_val))

        if val_err < best_val_err:
            best_val_err = val_err
            best_alpha = alpha
            best_hls = hls

print(f"Best alpha: {best_alpha}, Best hidden_layer_sizes: {best_hls}")

# Huấn luyện lại mô hình với toàn bộ dữ liệu huấn luyện và các tham số tốt nhất
mlp_regressor.set_params(mlp__alpha=best_alpha, mlp__hidden_layer_sizes=best_hls)
# Huấn luyện mô hình MLPRegressor
mlp_regressor.fit(X_train, y_train)


# Đánh giá mô hình MLPRegressor trên tập kiểm tra
y_pred_mlp = mlp_regressor.predict(X_test)
print("MLPRegressor:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred_mlp))

# Lưu mô hình MLPRegressor
joblib.dump(mlp_regressor, "mlp_regressor.pkl")


# Xây dựng pipeline cho RandomForestRegressor
rf_regressor = Pipeline([
    ('scaler', StandardScaler()),  # Chuẩn hóa dữ liệu
    ('rf', RandomForestRegressor(n_estimators=100, random_state=0))  # Mô hình Random Forest
])

best_val_err_rf = float('inf')
best_n_estimators = None
best_max_depth = None
best_min_samples_split = None

n_estimators_values = [50, 100, 150, 200]  # Số lượng cây quyết định
max_depth_values = [10, 20, 30, 40]  # Độ sâu tối đa mà mỗi cây quyết định
min_samples_split_values = [2, 5, 10, 20]  # Số lượng mẫu tối thiểu mà một nút

for n_estimators in n_estimators_values:
    for max_depth in max_depth_values:
        for min_samples_split in min_samples_split_values:
            print(f"n_estimators = {n_estimators}, max_depth = {max_depth}, min_samples_split = {min_samples_split}")
            rf_regressor.set_params(rf__n_estimators=n_estimators, rf__max_depth=max_depth, rf__min_samples_split=min_samples_split)
            rf_regressor.fit(x_train_v, y_train_v)
            val_err_rf = mean_squared_error(y_val, rf_regressor.predict(x_val))  # Sử dụng mean_squared_error để tính toán lỗi

            if val_err_rf < best_val_err_rf:
                best_val_err_rf = val_err_rf
                best_n_estimators = n_estimators
                best_max_depth = max_depth
                best_min_samples_split = min_samples_split

print(f"Best n_estimators: {best_n_estimators}, Best max_depth: {best_max_depth}, Best min_samples_split: {best_min_samples_split}")

# Huấn luyện lại mô hình Random Forest Regressor với toàn bộ dữ liệu huấn luyện và các tham số tốt nhất
rf_regressor.set_params(rf__n_estimators=best_n_estimators, rf__max_depth=best_max_depth, rf__min_samples_split=best_min_samples_split)
rf_regressor.fit(X_train, y_train)

# Đánh giá mô hình RandomForestRegressor trên tập kiểm tra
y_pred_rf = rf_regressor.predict(X_test)
print("\nRandom Forest Regressor:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred_rf))

# Tính toán accuracy với ngưỡng thay đổi
thresholds = [50, 100, 150]  # Danh sách các ngưỡng
for threshold in thresholds:
    y_pred_rf_discrete = ['Tốt' if pred <= threshold else 'Kém' for pred in y_pred_rf]
    y_test_discrete = ['Tốt' if label <= threshold else 'Kém' for label in y_test]
    accuracy_rf = accuracy_score(y_test_discrete, y_pred_rf_discrete)
    print("Accuracy with threshold", threshold, ":", accuracy_rf)

# Lưu mô hình RandomForestRegressor
joblib.dump(rf_regressor, "rf_regressor.pkl")

# Model SVR
svr_model = Pipeline(steps=[
    ('scaler', StandardScaler()),  # Chuẩn hóa dữ liệu
    ('svr', SVR(kernel='rbf'))  # Mô hình SVR
])

# Các giá trị của tham số C và gamma cần thử
C_values = [0.1, 1, 10, 20, 50, 100]
gamma_values = [0.0001, 0.001, 0.01, 0.1, 1]

best_val_err = float('inf')
best_C = None
best_gamma = None

for C in C_values:
    for gamma in gamma_values:
        print(f"C = {C}, gamma = {gamma}")
        svr_model.set_params(svr__C=C, svr__gamma=gamma)
        svr_model.fit(x_train_v, y_train_v)

        # Đánh giá sai số trên tập validation
        val_err = mean_squared_error(y_val, svr_model.predict(x_val))

        # Lưu lại các giá trị tốt nhất
        if val_err < best_val_err:
            best_val_err = val_err
            best_C = C
            best_gamma = gamma

print(f"Best C: {best_C}, Best gamma: {best_gamma}")

# Huấn luyện lại mô hình với các tham số tốt nhất trên toàn bộ dữ liệu huấn luyện
svr_model.set_params(svr__C=best_C, svr__gamma=best_gamma)
svr_model.fit(X_train, y_train)

# Lưu mô hình đã được huấn luyện
joblib.dump(svr_model, "svm_regressor_model.pkl")



# Chuẩn bị mô hình GradientBoostingRegressor
gb_regressor_model = Pipeline(steps=[
    ('scaler', StandardScaler()),  # Chuẩn hóa dữ liệu
    ('gb_regressor', GradientBoostingRegressor(random_state=0))  # Mô hình GradientBoostingRegressor
])

# Tinh chỉnh tham số
params_grid_gb = {
    'gb_regressor__n_estimators': [50, 100, 150],
    'gb_regressor__learning_rate': [0.05, 0.1, 0.2],
    'gb_regressor__max_depth': [3, 5, 7]
}

# Tạo GridSearchCV cho GradientBoostingRegressor
gb_grid_search = GridSearchCV(estimator=gb_regressor_model, param_grid=params_grid_gb, cv=3, n_jobs=-1)

# Tiến hành tinh chỉnh
gb_grid_search.fit(x_train_v, y_train_v)

# Lấy ra các tham số tốt nhất
best_gb_regressor_model = gb_grid_search.best_estimator_
best_gb_regressor_params = gb_grid_search.best_params_

# Huấn luyện lại mô hình GradientBoostingRegressor với các tham số tốt nhất
best_gb_regressor_model.fit(X_train, y_train)

# Lưu mô hình GradientBoostingRegressor
joblib.dump(best_gb_regressor_model, "gb_regressor_model.pkl")

# Đánh giá mô hình GradientBoostingRegressor trên tập kiểm tra
y_pred_gb_regressor = best_gb_regressor_model.predict(X_test)
mse_gb_regressor = mean_squared_error(y_test, y_pred_gb_regressor)
print("Mean Squared Error (GradientBoostingRegressor):", mse_gb_regressor)

