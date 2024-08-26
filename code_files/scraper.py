import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

def scrape_wikipedia(url):
    # Setup Chrome driver
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')  # Run headless Chrome for no GUI
    options.add_argument('--disable-gpu')
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    try:
        driver.get(url)
    except:
        return "Invalid URL"
    
    time.sleep(2)
    
    # Scrape the title
    try:
        title = driver.find_element(By.ID, "firstHeading").text
        print(f"Title: {title}")    
    except:
        title = ""
    
    try:
        # Scrape the content
        content = driver.find_element(By.ID, "bodyContent").text
        # print(content)
        
        # Save to a text file
        # output_file = f"{title}.txt"
        # with open(output_file, 'w', encoding='utf-8') as file:
        #     file.write(f"Title: {title}\n")
        #     file.write(content)
        # print(f"Content saved to {output_file}")
        
        driver.quit()
        return title + content
    except:
        driver.quit()
        return "Invalid URL"
    

# Example usage
url = "https://en.wikipedia.org/wiki/Mahatma_Gandhi"
# url = "hts://en.wikipedia.org/wiki/Bugatti"
# url = "https://www.google.com/search?q=financial+pages&oq=financial+pages&gs_lcrp=EgZjaHJvbWUyBggAEEUYOdIBCDQ2OTdqMGo3qAIAsAIA&sourceid=chrome&ie=UTF-8"
# scrape_wikipedia(url)

# print(scrape_wikipedia(url))
