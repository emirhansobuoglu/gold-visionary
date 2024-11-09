import time

import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By

driver = webdriver.Chrome()
driver.set_window_size(1920, 1080)
driver.set_window_position(0, 0)
URL = "https://tr.investing.com/currencies/gau-try-historical-data"
driver.get(URL)

time.sleep(5)
goldData = driver.find_elements(By.CSS_SELECTOR,".datatable_cell__LJp3C")
for td in goldData:
   if "%" not in td.text:
      print(td.text)