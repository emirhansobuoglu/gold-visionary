import time
from fileinput import close
from operator import index
from pydoc import classname
from types import NoneType
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from datetime import datetime
from dateutil.relativedelta import relativedelta
import requests
import pandas as pd
from bs4 import BeautifulSoup

driver = webdriver.Chrome()
driver.maximize_window()
URL = "https://tr.investing.com/currencies/gau-try-historical-data"
datas = []


def close_popup(second):
    try:
        close_button = WebDriverWait(driver, second).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "svg[data-test='sign-up-dialog-close-button']"))
        )
        close_button.click()
        print("Pop-up kapatıldı.")
    except:
        print("Pop-up bulunamadı.")


driver.get(URL)
time.sleep(5)
close_popup(15)
scroll_amount = 10
pause_time = 0.05

thisDay = datetime.now()
thisDate = thisDay.strftime("%d.%m.%Y")
print(thisDate)
for _ in range(75):
    driver.execute_script(f"window.scrollBy(0, {scroll_amount});")
    time.sleep(pause_time)

close_popup(3)
try:
    dateBox = WebDriverWait(driver, 20).until(
        EC.element_to_be_clickable((By.XPATH, f"//div[contains(text(), '- {thisDate}')]"))
    )
    dateBox.click()
    time.sleep(3)
    firstDateBox = WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.XPATH, '(//input[@type="date" and @max="2024-11-13"])[1]'))
    )
    firstDateBox.click()
    for i in range(700):
        firstDateBox.send_keys(Keys.ARROW_UP)
    time.sleep(1)
    apply_button = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "//span[text()='Uygula']")))
    apply_button.click()
    print("Butona başarıyla tıklandı.")
except:
    print("Butona tıklanamadı:")
URL2FirstDate = "2013/04/10"
URL2EndDate = "2011/01/01"
date_format = "%Y/%m/%d"

firstDate = datetime.strptime(URL2FirstDate, date_format)
endDate = datetime.strptime(URL2EndDate, date_format)

while firstDate >= endDate:

    URL2 = f"https://altin.in/arsiv/{firstDate.year}/{firstDate.month:02d}/{firstDate.day:02d}"
    driver.get(URL2)
    try:
        goldData2 = driver.find_element(By.XPATH, "//li[@title='Gram Altın - Alış']")
        print(f"Çekilen Veri: {goldData2.text} - Tarih: {firstDate.strftime(date_format)}")
    except Exception as e:
        print(f"Veri çekme hatası - Tarih: {firstDate.strftime(date_format)}")

    firstDate -= relativedelta(days=1)
time.sleep(5)