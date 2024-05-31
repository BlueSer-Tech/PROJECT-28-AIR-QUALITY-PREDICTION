# # beauty_soup.py

# from bs4 import BeautifulSoup
# from urllib.request import urlopen

# url = "https://www.iqair.com/vi/vietnam/ho-chi-minh-city"
# page = urlopen(url)
# html = page.read().decode("utf-8")
# soup = BeautifulSoup(html, "html.parser")
# re=soup.get_text()
# r=soup.find_all("aqi-value__value")
# for i in r:
#     print(i)

# To run this, download the BeautifulSoup zip file
# http://www.py4e.com/code3/bs4.zip
# and unzip it in the same directory as this file

# import urllib.request, urllib.parse, urllib.error
# from bs4 import BeautifulSoup
# import ssl

# # Ignore SSL certificate errors
# ctx = ssl.create_default_context()
# ctx.check_hostname = False
# ctx.verify_mode = ssl.CERT_NONE

# url = "https://www.iqair.com/vi/vietnam/ho-chi-minh-city"
# html = urllib.request.urlopen(url, context=ctx).read()
# soup = BeautifulSoup(html, 'html.parser')

# # Retrieve all of the anchor tags
# tags = soup('a')
# for tag in tags:
#     print(tag.get('class'))

import requests
from bs4 import BeautifulSoup
from datetime import datetime,timedelta
import os
import gspread
from oauth2client.service_account import ServiceAccountCredentials
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
max_duration = timedelta(minutes=5)

while datetime.now() - start_time < max_duration:
    lst=[]
    lst1=[]
    lst2=[]

    # Making a GET request
    r = requests.get('https://www.iqair.com/vi/vietnam/ho-chi-minh-city')

    # Parsing the HTML
    soup = BeautifulSoup(r.content, 'html.parser')
    dataTable = soup.find('table', class_='aqi-overview-detail__main-pollution-table')
    contentdataTable = dataTable.find_all('td')
    weather = soup.find('div', class_='weather__detail')
    contentWeather = weather.find_all('td')


    for i in range(1,len(contentWeather),2):
        lst.append(contentWeather[i].text)

    for i in contentdataTable: 
        lst.append(i.text)


    # PM= soup.find("span", class_="mat-tooltip-trigger pollutant-concentration-value")
    PM_Table = soup.find('span', class_='mat-tooltip-trigger pollutant-concentration-value').text


    r2 = requests.get('https://www.worldweatheronline.com/ho-chi-minh-city-weather/vn.aspx')

    # Parsing the HTML
    soup2 = BeautifulSoup(r2.content, 'html.parser')
    dataTable2 = soup2.find('div', class_='ws-details')
    contentdataTable2 = dataTable2.find_all('div', class_="ws-details-item")



    for j in range(1,len(contentdataTable2)):
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
    data = [future_time_str, lst[0], lst[1], lst[2], lst[3], lst[4], PM_Table, lst[5], lst[6][1:3], lst2[1][47:49],feels[41:43]]
    print(data)
    # Gửi dữ liệu vào hàng tiếp theo trong cột A
    next_row = len(worksheet.col_values(1)) + 1
    for i, item in enumerate(data):
        worksheet.update_cell(next_row, i + 1, item)
    time.sleep(10)