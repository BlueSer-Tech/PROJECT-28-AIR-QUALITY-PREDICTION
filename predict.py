import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
import re
import openpyxl
import random
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import os
import time

scope = ['https://spreadsheets.google.com/feeds',
         'https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name('credentials.json', scope)
client = gspread.authorize(creds)


sheet = client.open_by_url('https://docs.google.com/spreadsheets/d/10mzqlmS9ZYW2YNOnDS2ioAcgYpwbsScwSQ406sYovxk/edit?hl=vi#gid=0')
# Hoặc
# sheet = client.open_by_url('YOUR_SPREADSHEET_URL')

# Lấy bảng theo tên
worksheet = sheet.worksheet("Sheet 1")

lst_global=["0","1","2","3","4"]

start_time = datetime.now()

# Thiết lập ngưỡng thời gian tối đa cho vòng lặp (ví dụ: 1 giờ)
max_duration = timedelta(minutes=30)

while datetime.now() - start_time < max_duration:
    lst = []
    lst1 = []
    lst2 = []

    # Thực hiện yêu cầu GET
    r = requests.get('https://www.iqair.com/vi/vietnam/ho-chi-minh-city')

    # Phân tích cú pháp HTML
    soup = BeautifulSoup(r.content, 'html.parser')
    dataTable = soup.find('table', class_='aqi-overview-detail__main-pollution-table')
    contentdataTable = dataTable.find_all('td')
    weather = soup.find('div', class_='weather__detail')
    contentWeather = weather.find_all('td')

    for i in range(1, len(contentWeather), 2):
        lst.append(contentWeather[i].text)

    for i in contentdataTable:
        lst.append(i.text)

    # PM= soup.find("span", class_="mat-tooltip-trigger pollutant-concentration-value")
    PM_Table = soup.find('span', class_='mat-tooltip-trigger pollutant-concentration-value').text

    r2 = requests.get('https://www.worldweatheronline.com/ho-chi-minh-city-weather/vn.aspx')

    # Phân tích cú pháp HTML
    soup2 = BeautifulSoup(r2.content, 'html.parser')
    dataTable2 = soup2.find('div', class_='ws-details')
    contentdataTable2 = dataTable2.find_all('div', class_="ws-details-item")

    for j in range(1, len(contentdataTable2)):
        lst2.append(contentdataTable2[j].text)

    # print(lst2[1][103:])

    dataTable3 = soup2.find('div', class_='weather-widget-temperature')
    contentdataTable3 = dataTable3.find_all('p', class_="feels")

    feels = contentdataTable3[0].text

    # x = datetime.now()
    # realTime=x.strftime("%Y/%m/%d - %H:%M:%S")
    time_delta = timedelta(days=1)

    # Thêm khoảng thời gian vào thời gian hiện tại để có thời gian trong tương lai
    future_time = datetime.now() + time_delta

    # Chuyển đổi thời gian trong tương lai thành chuỗi định dạng
    future_time_str = future_time.strftime("%Y/%m/%d - %H:%M:%S")
    data = [future_time_str, lst[0], lst[1], lst[2], lst[3], lst[4], PM_Table, lst[5], lst[6][1:3], lst2[1][47:49],
            feels[41:43]]
    print(data)
    # Gửi dữ liệu vào hàng tiếp theo trong cột A
    next_row = len(worksheet.col_values(1)) + 1
    for i, item in enumerate(data):
        worksheet.update_cell(next_row, i + 1, item)
    time.sleep(30)


scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name('skilful-ethos-420017-4f2f492e9611.json', scope)
client = gspread.authorize(creds)

# Mở bảng tính
sheet2 = client.open('Test28032024').sheet1  # Thay 'Tên bảng tính' bằng tên bảng tính của bạn

# Đọc dữ liệu từ bảng tính
data = sheet2.get_all_records()
num_rows = 10

gg_sheet = pd.DataFrame(data)
gg_sheet['Date'] = pd.to_datetime(gg_sheet['Ngày giờ'])
gg_sheet['Day'] = gg_sheet['Date'].dt.day
gg_sheet['Month'] = gg_sheet['Date'].dt.month
gg_sheet['Year'] = gg_sheet['Date'].dt.year

gg_sheet['time'] = gg_sheet['Date'].dt.strftime('%H:%M')
gg_sheet['hour'] = gg_sheet['Date'].dt.hour

def process_column(sheet, column_name):
    sheet[column_name] = pd.to_numeric(sheet[column_name].str.replace(r'[^\d.]', '', regex=True), errors='coerce').astype(float)


# Xử lý các cột
columns_to_process = ['Nhiệt độ', 'Độ ẩm', 'Gió', 'Áp suất','Độ che phủ mây','Feels']
for column in columns_to_process:
    process_column(gg_sheet, column)

columns_to_fill = [ 'Độ che phủ mây','Feels']
for column in columns_to_fill:
    gg_sheet[column] = gg_sheet[column].fillna(gg_sheet[column].mean())

shuffled_df = gg_sheet.sample(frac=1, random_state=42)

# Chọn ngẫu nhiên 10 dòng từ DataFrame gg_sheet
random_selection = shuffled_df.head(10)

# Lưu các dòng đã chọn vào DataFrame mới
selected_data = pd.DataFrame(random_selection)

# Xác định lớp cho máy biến áp tùy chỉnh
class ColStd(BaseEstimator, TransformerMixin):
    def fit(self, X_df, y=None):
        return self

    def transform(self, X_df, y=None):
        df = X_df.copy()
        df['time'] = df['time'].apply(lambda x: self.extract_time(x))
        df['month'] = df['month'].apply(lambda x: 'mùa' if x in [5, 6, 7, 8, 9, 10, 11] else 'khô')
        df['direction'] = df['direction'].apply(lambda x: x[1:] if len(x) == 3 else x)
        df['weather'] = df['weather'].apply(lambda w: 'không mưa' if w in list_norain else ('mưa' if w in list_rain else 'không xác định'))
        return df

    def extract_time(self, x):
        match = re.findall(r'(\d{1,2}):\d{2}', x)
        if match:
            hour = int(match[0])
            return 'sáng' if 6 <= hour <= 15 else 'tối'
        else:
            return 'thời gian không hợp lệ'

# Định nghĩa list_norain và list_rain
list_norain = ['Clear', 'Cloudy', 'Mist', 'Sunny', 'Partly cloudy', 'Thundery outbreaks possible']
list_rain = ['Light drizzle','Light rain','Light rain shower', 'Patchy light drizzle', 'Patchy light rain',
             'Patchy light rain with thunder','Patchy rain possible','Heavy rain','Heavy rain at times',
             'Moderate or heavy rain shower','Moderate rain', 'Moderate rain at times', 'Overcast','Torrential rain shower']

weather_translation = {
    'Clear': 'Trời quang đãng',
    'Cloudy': 'Âm u',
    'Mist': 'Sương mù',
    'Sunny': 'Nắng',
    'Partly cloudy': 'Trời nhiều mây',
    'Thundery outbreaks possible': 'Có khả năng có dông',
    'Light drizzle': 'Mưa phùn nhẹ',
    'Light rain': 'Mưa nhẹ',
    'Light rain shower': 'Mưa nhẹ',
    'Patchy light drizzle': 'Mưa phùn nhẹ',
    'Patchy light rain': 'Mưa nhẹ',
    'Patchy light rain with thunder': 'Mưa nhẹ kèm sấm sét',
    'Patchy rain possible': 'Có khả năng mưa nhỏ',
    'Heavy rain': 'Mưa lớn',
    'Heavy rain at times': 'Mưa lớn lúc nào',
    'Moderate or heavy rain shower': 'Mưa vừa hoặc mưa lớn',
    'Moderate rain': 'Mưa vừa',
    'Moderate rain at times': 'Mưa vừa lúc nào',
    'Overcast': 'Trời âm u',
    'Torrential rain shower': 'Mưa lớn'
}

weather_mapping = {
    'norain': 'Không mưa',   # không mưa
    'rain': 'Mưa',          # mưa
}
new_data = {
    'time': selected_data['time'],
    'month': selected_data['Month'],
    'temperature': selected_data['Nhiệt độ'],
    'feelslike': selected_data['Feels'],
    'wind': selected_data['Gió'],
    'gust': selected_data['Gió'],
    'cloud': selected_data['Độ che phủ mây'],
    'humidity': selected_data['Độ ẩm'],
    'pressure': selected_data['Áp suất'],
    'direction': ['']*num_rows,
    'weather': ['']*num_rows
}

new_data1 = pd.DataFrame({
    'day': selected_data['Day'],
    'month': selected_data['Month'],
    'year': selected_data['Year'],
    'PM25' : selected_data['Nồng độ PM2.5']
})

# Chọn các tính năng cần thiết để dự đoán
X_new = new_data1[['day','month','year','PM25']]

# Chức năng phân loại phạm vi AQI
def aqi_range(x):
    if 0 <= x <= 50:
        return "Tốt"
    elif 51 <= x <= 100:
        return "Trung bình"
    elif 101 <= x <= 150:
        return "Không tốt cho các nhóm nhạy cảm"
    elif 151 <= x <= 200:
        return "Không lành mạnh"
    elif 201 <= x <= 300:
        return "Rất không tốt"
    elif 301 <= x <= 500:
        return "Nguy hiểm"

# Tải các mô hình đã được huấn luyện
mlp_Classifier = joblib.load("mlpclf_model.pkl")
rf_Classifier = joblib.load("rf_model.pkl")
mlp_Classifier1 = joblib.load("mlpclf_model1.pkl")
rf_Classifier1 = joblib.load("rf_model1.pkl")
svc = joblib.load("svmclf_model.pkl")
gb_Classifier = joblib.load("gb_model.pkl")
svc1 = joblib.load("svmclf_model1.pkl")
gb_Classifier1 = joblib.load("gb_model1.pkl")
mlp_regressor = joblib.load("mlp_regressor.pkl")
rf_regressor = joblib.load("rf_regressor.pkl")
svr = joblib.load("svm_regressor_model.pkl")
gb_regressor = joblib.load("gb_regressor_model.pkl")

# Xử lý trước dữ liệu
preprocessed_data = pd.DataFrame(new_data)

# Dự đoán thời tiết mới từ dữ liệu Dự đoán thời tiết MLPClassifier, Random Forest Classifier, SVM, GradientBoostingClassifier
y_pred_mlp = mlp_Classifier.predict(preprocessed_data)
y_pred_rf = rf_Classifier.predict(preprocessed_data)
y_pred_svc = gb_Classifier.predict(preprocessed_data)
y_pred_gb = svc.predict(preprocessed_data)

y_pred_mlp_vietnamese = [weather_mapping[label] for label in y_pred_mlp]
y_pred_rf_vietnamese = [weather_mapping[label] for label in y_pred_rf]
y_pred_svmclf_vietnamese = [weather_mapping[label] for label in y_pred_svc]
y_pred_gb_vietnamese = [weather_mapping[label] for label in y_pred_gb]

# In kết quả dự đoán ra màn hình
print("Dự đoán thời tiết (MLPClassifier):", y_pred_mlp_vietnamese)
print("Dự đoán thời tiết (Random Forest Classifier):",y_pred_rf_vietnamese)
print("Dự đoán thời tiết (SVM):",y_pred_svmclf_vietnamese)
print("Dự đoán thời tiết (GradientBoostingClassifier):", y_pred_gb_vietnamese)

# Dự đoán thời tiết mới từ dữ liệu Dự đoán tình trạng thời tiết mới bằng model MLPClassifier, Random Forest Classifier, SVM, GradientBoostingClassifier
y_pred_mlp1 = mlp_Classifier1.predict(preprocessed_data)
y_pred_rf1 = rf_Classifier1.predict(preprocessed_data)
y_pred_svmclf1 = svc1.predict(preprocessed_data)
y_pred_gb1 = gb_Classifier1.predict(preprocessed_data)

y_pred_mlp_vietnamese1 = [weather_translation[label] for label in y_pred_mlp1]
y_pred_rf_vietnamese1 = [weather_translation[label] for label in y_pred_rf1]
y_pred_svmclf_vietnamese1 = [weather_translation[label] for label in y_pred_svmclf1]
y_pred_gb_vietnamese1 = [weather_translation[label] for label in y_pred_gb1]

# In kết quả dự đoán ra màn hình
print("Dự đoán tình trạng thời tiết (MLPClassifier):", y_pred_mlp_vietnamese1)
print("Dự đoán tình trạng thời tiết (Random Forest Classifier):", y_pred_rf_vietnamese1)
print("Dự đoán tình trạng thời tiết (SVM):", y_pred_svmclf_vietnamese1)
print("Dự đoán tình trạng thời tiết (GradientBoostingClassifier):", y_pred_gb_vietnamese1)
print('-'*50)

# Dự đoán chất lượng không khí bằng MLPRegressor, RandomForestRegressor, SVR, Gradient Boosting Regressor
predictions_mlp = mlp_regressor.predict(X_new)
predictions_rf = rf_regressor.predict(X_new)
predictions_svr = svr.predict(X_new)
predictions_gb = gb_regressor.predict(X_new)

# In kết quả dự báo từ cả hai mô hình và trạng thái của chất lượng không khí
print("Predictions from MLP Regressor:")
print(predictions_mlp)
print("AQI Range:")
print([aqi_range(x) for x in predictions_mlp])
print('-'*50)
print("\nPredictions from Random Forest Regressor:")
print(predictions_rf)
print("AQI Range:")
print([aqi_range(x) for x in predictions_rf])
print('-'*50)
print("\nPredictions from SVR:")
print(predictions_svr)
print("AQI Range:")
print([aqi_range(x) for x in predictions_svr])
print('-'*50)
print("\nPredictions from Gradient Boosting Regressor:")
print(predictions_gb)
print("AQI Range:")
print([aqi_range(x) for x in predictions_gb])


scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
credentials = ServiceAccountCredentials.from_json_keyfile_name('skilful-ethos-420017-4f2f492e9611.json', scope)
client = gspread.authorize(credentials)
sheet_url = "https://docs.google.com/spreadsheets/d/1iITV4GTIEL5UbP16qZaju6XygTHRr2iMwBCHj0-2dwg/edit#gid=0"
sheet1 = client.open_by_url(sheet_url).sheet1


cell_list = sheet1.range('A2:F11')

# Xóa nội dung của từng ô trong danh sách
for cell in cell_list:
    cell.value = ''

# Cập nhật nội dung của các ô trong danh sách
sheet1.update_cells(cell_list)

existing_data_range = sheet1.get_all_values()

# Chuyển dữ liệu hiện có xuống một hàng
for row_idx in range(len(existing_data_range), 1, -1):
    for col_idx in range(1, 5):  # Cập nhật cho 4 cột
        cell_value = sheet1.cell(row_idx, col_idx).value
        sheet1.update_cell(row_idx + 1, col_idx, cell_value)

current_time = datetime.now().strftime("%Y-%m-%d")

def future_date(days=0):
    current_date = datetime.now().strftime("%Y-%m-%d")
    future_date = datetime.strptime(current_date, "%Y-%m-%d") + timedelta(days=days)
    return future_date.strftime("%Y-%m-%d")


# Tạo từ điển để lưu trữ ngày tiếp theo cho mỗi giá trị Day
next_day_time_dict = {}

for i in range(len(new_data1)):
    day_value = gg_sheet['Day'].iloc[i]
    if day_value in next_day_time_dict:
        future_time = next_day_time_dict[day_value]  # Sử dụng thời gian của ngày tiếp theo đã lưu trữ nếu đã tồn tại
    else:
        future_date = datetime.now() + timedelta(days=len(next_day_time_dict) + 1)  # Tạo một ngày tiếp theo mới nếu chưa tồn tại
        future_time = future_date.strftime("%Y-%m-%d")  # Chuyển đổi ngày tiếp theo thành chuỗi thời gian
        next_day_time_dict[day_value] = future_time
    sheet1.update_cell(2 + i, 1, current_time)
    sheet1.update_cell(2 + i, 2, future_time)

for i in range(len(y_pred_rf_vietnamese)):
    sheet1.update_cell(2 + i, 3, y_pred_rf_vietnamese[i])

for i in range(len(y_pred_rf_vietnamese1)):
    sheet1.update_cell(2 + i, 4, y_pred_rf_vietnamese1[i])
#
for i in range(len(predictions_rf)):
    sheet1.update_cell(2 + i, 5, predictions_rf[i])
    sheet1.update_cell(2 + i, 6, aqi_range(predictions_rf[i]))
