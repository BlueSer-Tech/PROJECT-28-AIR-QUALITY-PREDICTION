import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
import joblib
from sklearn.metrics import classification_report

# Đọc dữ liệu từ tệp CSV
data2017 = pd.read_csv("weather_data_[2017].csv", sep="\t")
data2018 = pd.read_csv("weather_data_[2018].csv", sep="\t")
data2019 = pd.read_csv("weather_data_[2019].csv", sep="\t")
data = pd.concat([data2017, data2018, data2019], ignore_index=True, axis=0)

# Xếp loại thời tiết
list_norain = ['Clear', 'Cloudy', 'Mist', 'Sunny', 'Partly cloudy', 'Thundery outbreaks possible']
list_rain = ['Light drizzle', 'Light rain', 'Light rain shower', 'Patchy light drizzle', 'Patchy light rain',
             'Patchy light rain with thunder', 'Patchy rain possible', 'Heavy rain', 'Heavy rain at times',
             'Moderate or heavy rain shower', 'Moderate rain', 'Moderate rain at times', 'Overcast', 'Torrential rain shower']

y = []
for w in data.weather:
    if w in list_norain:
        y += ['norain']
    elif w in list_rain:
        y += ['rain']
    else:
        y += ['unknown']
y.pop(0)  # Điều chỉnh do dự đoán lệch mục tiêu

x = data.drop(8511)  # Điều chỉnh do dự đoán lệch mục tiêu

# Chia dữ liệu
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, stratify=y, random_state=0)
x_train_v, x_val, y_train_v, y_val = train_test_split(x_train, y_train, test_size=0.2, stratify=y_train, random_state=0)

class ColStd(BaseEstimator, TransformerMixin):
    def fit(self, X_df, y=None):
        return self

    def transform(self, X_df, y=None):
        df = X_df.copy()
        df['time'] = df['time'].apply(lambda x: 'sang' if 6 <= int(x.split(':')[0]) <= 15 else 'toi')
        df['month'] = df['month'].apply(lambda x: 'mua' if x in [5, 6, 7, 8, 9, 10, 11] else 'kho')
        df['direction'] = df['direction'].apply(lambda x: x[1:] if len(x) == 3 else x)
        df['weather'] = df['weather'].apply(lambda w: 'norain' if w in list_norain else ('rain' if w in list_rain else 'unknown'))
        return df

# phân tách dữ liệu so sánh và dữ liệu không dùng đến
nume_cols = ['temperature', 'feelslike', 'wind', 'gust', 'cloud', 'humidity', 'pressure']
cate_cols = ['time', 'month', 'direction', 'weather']


nume_trans = Pipeline(steps=[
    ('imputer1', SimpleImputer(missing_values=np.nan, strategy='mean')),
    ('imputer2', SimpleImputer(missing_values=0, strategy='mean'))
])

cate_trans = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(dtype=int, handle_unknown='ignore'))
])

cols_trans = ColumnTransformer(transformers=[
    ('nume', nume_trans, nume_cols),
    ('cate', cate_trans, cate_cols)
])

preprocess_pipeline = Pipeline(steps=[
    ('colstd', ColStd()),
    ('coltrans', cols_trans),
    ('std', StandardScaler())
])

# Chuẩn bị mô hình MLPClassifier
mlpclf_model = Pipeline(steps=[
    ('pre', preprocess_pipeline),
    ('mlpclf', MLPClassifier(hidden_layer_sizes=(20), activation='tanh', solver='adam', random_state=0, max_iter=1000))
])

# Chuẩn bị mô hình Random Forest Classifier
rf_model = Pipeline(steps=[
    ('pre', preprocess_pipeline),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=0))
])

# Tinh chỉnh tham số
best_val_err = float('inf')
best_alpha = None
best_hls = None

alphas = [0.001, 0.01, 0.1, 1, 10, 20, 100]
hidden_layer_sizes = [1, 10, 20, 50, 70, 100]

for alpha in alphas:
    for hls in hidden_layer_sizes:
        print(f"alpha = {alpha}, hls = {hls}")
        mlpclf_model.set_params(mlpclf__alpha=alpha, mlpclf__hidden_layer_sizes=(hls))
        mlpclf_model.fit(x_train_v, y_train_v)
        val_err = 100 * (1 - mlpclf_model.score(x_val, y_val))

        if val_err < best_val_err:
            best_val_err = val_err
            best_alpha = alpha
            best_hls = hls

print(f"Best alpha: {best_alpha}, Best hidden_layer_sizes: {best_hls}")

# Huấn luyện lại mô hình với toàn bộ dữ liệu huấn luyện và các tham số tốt nhất
mlpclf_model.set_params(mlpclf__alpha=best_alpha, mlpclf__hidden_layer_sizes=(best_hls))
# Huấn luyện mô hình MLPClassifier
mlpclf_model.fit(x_train, y_train)

# Lưu mô hình MLPClassifier
joblib.dump(mlpclf_model, "mlpclf_model.pkl")



# Tinh chỉnh tham số
best_val_err_rf = float('inf')
best_n_estimators = None
best_max_depth = None
best_min_samples_split = None

n_estimators_values = [50, 100, 150, 200] #số lượng cây quyết định
max_depth_values = [10, 20, 30, 40] #độ sâu tối đa mà mỗi cây quyết định
min_samples_split_values = [2, 5, 10, 20] #số lượng mẫu tối thiểu mà một nút

for n_estimators in n_estimators_values:
    for max_depth in max_depth_values:
        for min_samples_split in min_samples_split_values:
            print(f"n_estimators = {n_estimators}, max_depth = {max_depth}, min_samples_split = {min_samples_split}")
            rf_model.set_params(rf__n_estimators=n_estimators, rf__max_depth=max_depth, rf__min_samples_split=min_samples_split)
            rf_model.fit(x_train_v, y_train_v)
            val_err_rf = 100 * (1 - rf_model.score(x_val, y_val))

            if val_err_rf < best_val_err_rf:
                best_val_err_rf = val_err_rf
                best_n_estimators = n_estimators
                best_max_depth = max_depth
                best_min_samples_split = min_samples_split

print(f"Best n_estimators: {best_n_estimators}, Best max_depth: {best_max_depth}, Best min_samples_split: {best_min_samples_split}")

# Huấn luyện lại mô hình Random Forest Classifier với toàn bộ dữ liệu huấn luyện và các tham số tốt nhất
rf_model.set_params(rf__n_estimators=best_n_estimators, rf__max_depth=best_max_depth, rf__min_samples_split=best_min_samples_split)
rf_model.fit(x_train, y_train)

# Lưu mô hình Random Forest Classifier
joblib.dump(rf_model, "rf_model.pkl")


svmclf_model = Pipeline(steps=[
    ('pre', preprocess_pipeline),
    ('svm', SVC(kernel = 'rbf'))
])
c = [0.1,1,10, 20, 50,100]
gamma = [0.0001,0.001,0.01,0.1,1]
train_errs1 = []
val_errs1 = []
iter = []
best_val_err = float('inf'); best_C = None; best_gamma=None
for C in c:
    for g in gamma:
        iter +=[f"C={C}, g={g}"]
        print(iter[-1])
        svmclf_model.set_params(svm__C=C,svm__gamma=g)
        svmclf_model.fit(x_train_v,y_train_v)
        train_err =100*(1 - svmclf_model.score(x_train_v,y_train_v))
        val_err =100*(1 - svmclf_model.score(x_val,y_val))
        train_errs1 += [train_err]
        val_errs1 +=[val_err]
        if ((val_err < best_val_err)):
            best_val_err = val_err
            best_C = C
            best_gamma =g

for i in range(len(train_errs1)):
    print(f"{iter[i]} \n train_err :{train_errs1[i]}\t val_err:{val_errs1[i]}")
    print()
print(f"best C: {best_C}")
print(f"best gamma: {best_gamma}")

#điểm của mô hình svmclf khi áp dụng tham số tốt nhất dự đoán trên tập tập x_train_v
svmclf_model.set_params(svm__C=best_C,svm__gamma=best_gamma)
svmclf_model.fit(x_train_v,y_train_v)

joblib.dump(svmclf_model, "svmclf_model.pkl")



# Chuẩn bị mô hình GradientBoostingClassifier
gb_model = Pipeline(steps=[
    ('pre', preprocess_pipeline),
    ('gb', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0))
])

# Tinh chỉnh tham số
params_grid_gb = {
    'gb__n_estimators': [50, 100, 150],
    'gb__learning_rate': [0.05, 0.1, 0.2],
    'gb__max_depth': [3, 5, 7]
}

from sklearn.model_selection import GridSearchCV

# Tạo GridSearchCV cho GradientBoostingClassifier
gb_grid_search = GridSearchCV(estimator=gb_model, param_grid=params_grid_gb, cv=3, n_jobs=-1)

# Tiến hành tinh chỉnh
gb_grid_search.fit(x_train_v, y_train_v)

# Lấy ra các tham số tốt nhất
best_gb_model = gb_grid_search.best_estimator_
best_gb_params = gb_grid_search.best_params_

# Huấn luyện lại mô hình GradientBoostingClassifier với các tham số tốt nhất
best_gb_model.fit(x_train, y_train)

# Lưu mô hình GradientBoostingClassifier
joblib.dump(best_gb_model, "gb_model.pkl")

# Đánh giá mô hình GradientBoostingClassifier trên tập kiểm tra
y_pred_gb = best_gb_model.predict(x_test)
print("GradientBoostingClassifier:")
print(classification_report(y_test, y_pred_gb))

# Đánh giá mô hình MLPClassifier trên tập kiểm tra
y_pred_mlp = mlpclf_model.predict(x_test)
print("MLPClassifier:")
print(classification_report(y_test, y_pred_mlp))

y_pred = svmclf_model.predict(x_val)
print("SVM:")
print(classification_report(y_val, y_pred))

# Đánh giá mô hình Random Forest Classifier trên tập kiểm tra
y_pred_rf = rf_model.predict(x_test)
print("\nRandom Forest Classifier:")
print(classification_report(y_test, y_pred_rf))
