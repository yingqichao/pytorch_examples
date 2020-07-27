from selenium import webdriver
from bs4 import BeautifulSoup
import time
import re

word, word_dict, num_downloaded = [], set(), 0
name = ['realDonaldTrump','Mike_Pence','CNN','SecondLady']
url = 'https://twitter.com/search?q=(from%3A'+name[2]+')%20until%3A2020-07-27%20since%3A2020-01-01%20-filter%3Areplies&src=typed_query&f=live'
#创建一个浏览器对象
driver = webdriver.Chrome('C:\Program Files (x86)\Google\Chrome\Application\chromedriver.exe')
#获取网页
driver.get(url)
#等待加载
time.sleep(5)
def gundong(n):
    driver.execute_script("window.scrollBy(0,"+str(n)+")")
    time.sleep(7)

def get_twitter():
    global num_downloaded
    try:
        #读取html文件
        page=BeautifulSoup(driver.page_source,'html5lib')
        #搜索目标信息
        # mu_mes = page.find('div',{'style':re.compile('position: relative; min-height: ')})
        # class="css-901oao r-hkyrab r-1qd0xha r-a023e6 r-16dba41 r-ad9z0x r-bcqeeo r-bnwqim r-qvutc0"
        mu_meses = page.find_all('div',{'class':re.compile('css-901oao r-hkyrab r-1qd0xha r-a023e6 r-16dba41 r-ad9z0x r-bcqeeo r-bnwqim r-qvutc0')})
        for mu_mes in mu_meses:
            mes_links = mu_mes.find_all('span',{'class':"css-901oao css-16my406 r-1qd0xha r-ad9z0x r-bcqeeo r-qvutc0"})
        # mes_links = mu_mes.find('span',{'class':"css-901oao css-16my406 r-1qd0xha r-ad9z0x r-bcqeeo r-qvutc0"})

            temp = ''
            #输出
            for mes_link in mes_links:
                temp += mes_link.get_text()
            if(temp not in word_dict):
                word.append(temp)
                word_dict.add(temp)
                print(temp)
                num_downloaded += 1
    except AttributeError as e:
        print('Error Occured')
    finally:
        print('Downloaded: '+str(num_downloaded))

# //*[@id="react-root"]/div/div/div[2]/main/div/div/div/div[1]/div/div[2]/div/div/section/div/div/div/div[2]/div/div/article/div/div/div/div[2]/div[2]/div[2]/div[1]/div/span
# /html/body/div/div/div/div[2]/main/div/div/div/div[1]/div/div[2]/div/div/section/div/div/div/div[2]/div/div/article/div/div/div/div[2]/div[2]/div[2]/div[1]/div
try:
    for i in range(10):
        n = i*2000
        gundong(n)
        get_twitter()
        print(i)
#关闭浏览器
#driver.close()
except AttributeError as e:
    print('Error Occurred!')
finally:
    with open('../mydataset/twitter_'+name[2]+'_train.txt', "w") as f:
        for i in range(len(word)):
            f.write(word[i])
            f.write('\n')