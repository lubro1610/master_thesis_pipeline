# For the harvest stage, pandas is used to create a dataframe and export to CSV. 
# Time is used for sleep(0), or pauses inbetween actions. It gives the domain time to load, and it decreases the chance of being blocked by any anti-bot measures.
# os is used to create file paths and directiories, as seen in os.path.join and os.makedirs.
# Selenium is one of the most popular web scraping tools, and allows for programmatic control of a web browser.
# Selenium contains various modules, such as webdriver, service, options, and several others used for the harvesting stage.
# Webdriver, service, and options are used to set up the browser and control it.
# By was used to choose HTML-elements in the Norges Bank html-code, which was particularly useful for speaker names, dates etc. found in <header> tags.
# WebDriverWait and expected_conditions were used to wait for certain elements to load before the script continued, which was key, seeing as Norges Bank uses dynamic loading.
# webdriver_manager automatically downloads the correct version of ChromeDriver, which is a key aspect of using Selenium with Chrome.

# NB: If anyone is reading this, you may notice that the script uses target year 1999. Before coming across various contraints making the 2006-2025 scope the better fit,
# I intended to harvest from 1999, which is why the target year reflects this. Obviously its not that efficient, but its a minor detail and I didnt mind it staying this way.
import pandas as pd
import time
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# One of the first issues I encountered in the harvesting & extraction pipeline was the cookie banner, which was hidden in a Shadow DOM.
# This complicated the procedure, as the dynamic loading of the page meant that the cookie banner would appear at somewhat random times.
# In addition, it meant that Selenium had to interact with the Shadow DOM in order to continue the harvesting process, which was a new concept for me.
# This was not an element I was much familiar with beforehand, but after some research I was able to find a solution.
# The solution was to use JavaScript to pierce the Shadow DOM and click the "Accept all" button, which allowed the automated harvesting process to continue. 
# After inspecting the code of the cookie banner, I found that the acceptance button had a unique data-testid attribute, which made bypassing it with Javascript feasible.
def handle_cookie_banner(driver):
    """
    Specifically targets the Usercentrics Shadow DOM cookie banner.
    """
    time.sleep(4) # First the program waits 4 seconds for the banner to load.
    
    try:
        script = """
        const root = document.querySelector('#usercentrics-root'); # First select the root element in order to target the Shadow DOM.
        if (root && root.shadowRoot) {
            const acceptBtn = root.shadowRoot.querySelector('button[data-testid="uc-accept-all-button"]'); # Then I select the acceptance button using its unique attribute.
            if (acceptBtn) { # If the button is found, it is clicked, which allows the harvesting process to continue.
                acceptBtn.click(); 
                return "SUCCESS";
            }
        }
        return "NOT_FOUND";
        """
        result = driver.execute_script(script)
        
        if result == "SUCCESS":
            time.sleep(2)
        else:
            standard_btn = driver.find_elements(By.ID, "coiPage-1-buttons-button-1") # In case the Shadow DOM method fails, the program checks for a standard banner as a backup.
            if standard_btn:
                standard_btn[0].click()
    except Exception:
        pass

def harvest_norges_bank_urls(targets, target_year=1999): # Main harvesting function. Takes dictionary of cetegories and URLs, and target year for stopping point.
    chrome_options = Options() 
    chrome_options.add_argument("--window-size=1920,1080")
    # chrome_options.add_argument("--headless") # Uncomment to run in background
    
    output_dir = os.path.join("02_Data", "urls") # Creates a file path for output directory, which stores the URLs in a structured CSV format. This is very useful for the next stage.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for category, base_url in targets.items(): # Iterates through the categories and their corresponding URLs, which are defined in the main function.
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options) 
        print(f"\ncategory={category}")
        driver.get(base_url) 
        
        collected_data = [] # This list stores the collected data.
        seen_urls = set() # This set is used to track URLs already retrieved, which prevents duplicates.
        
        try:
            handle_cookie_banner(driver) # Cookie banner is repeatedly checked for and bypassed if it appears.

            searching = True # This controls the main harvesting loop, and continues until target year reached/no more results.
            last_count = 0 # Tracks number of articles found in last iteration. If last count is same as current count, no new articles are loading and we break.
            
            while searching:
                # 1. Wait for articles to load using the class found earlier
                try:
                    WebDriverWait(driver, 15).until(
                        EC.presence_of_element_located((By.CLASS_NAME, "news-hit__link")) # Waits for presence of links, indicating the page has loaded enough to start harvest.
                    )
                except Exception:
                    print(f"timeout_no_articles={category}") # If no more links appear, assume end is reached and break.
                    break

                # 2. Extract links currently on page
                articles = driver.find_elements(By.CLASS_NAME, "news-hit__link") 
                
                # Check if we are actually getting new results
                if len(articles) == last_count:
                    print(f"no_new_items={category}")
                    break
                
                last_count = len(articles) # Update last count to current number of articles for next run of loop.

                for article in articles: # Iterate through the articles found on the page, extract URL, title, and year if possible.
                    url = article.get_attribute('href') 
                    if not url or url in seen_urls: # If URL is empty or seen, skip.
                        continue

                    try:
                        # Extract Year from URL
                        year = None # Tries to extract year from URL using tag such as <time>. 
                        parts = url.split('/') # Backup method in case <time> fails. Splits URL into parts and looks for 4-digit numbers that could be years.
                        for p in parts: 
                            if p.isdigit() and len(p) == 4: # If part is a 4-digit number, assume its the year and break.
                                year = int(p) 
                                break
                        
                        # Skip entries where year could not be determined or is beyond 2025
                        if year is None or year > 2025: # If year found seems unrealistic, skip. Had a case where year became something like 8000, so this is for protection.
                            continue
                        
                        seen_urls.add(url) # Add URL to seen to prevent duplicates.
                        collected_data.append({ # Stores collected data, useful for extraction step. 
                            'date': f"Year: {year}", 
                            'url': url, 
                            'title': article.text.strip(), 
                            'category': category 
                        })
                        
                        # Stop if we hit the target year
                        if year < target_year:
                            print(f"target_year_reached={year}")
                            searching = False
                            break
                    except Exception:
                        continue

                if not searching: break

                # 3. Handle 'Show more hits', which is a button that loads more results without changing the URL. 
                try:
                    load_more_btn = WebDriverWait(driver, 10).until( # Waits and checks for the "show more hits" button. 
                        EC.element_to_be_clickable((By.CLASS_NAME, "news__show-more")) # If it appears, it is clikced to load more results.
                    )
                    
                    if load_more_btn:
                        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", load_more_btn) # Scrolls to the button to ensure it is in view.
                        time.sleep(1) # Short pause to ensure button is interactable.
                        driver.execute_script("arguments[0].click();", load_more_btn) # Uses javascript when clicking the button as it is more reliable for such elements.
                        print(f"[{category}] links={len(collected_data)}") 
                        time.sleep(3) # Wait for page to update
                    else:
                        print(f"button_not_found={category}")
                        break
                except Exception:
                    print(f"end_of_archive={category}")
                    break

        finally:
            driver.quit() 
        
        if collected_data: # If data was collected throughout the process, it is saved to a CSV file in output directories. 
            df = pd.DataFrame(collected_data) 
            filename = os.path.join(output_dir, f"urls_{category.lower().replace(' ', '_')}.csv") 
            df.to_csv(filename, index=False)
            print(f"saved: {filename}")
            print(f"rows={len(df)}")

if __name__ == "__main__":
    JOBS_TO_DO = { # Defines the categories the program is to harvest, and their corresponding URLs. 
        "Press Releases": "https://www.norges-bank.no/en/news-events/news/?selectedFacets[Type]=11520", # Takes the program to the press release section of the news-page.
        "Speeches": "https://www.norges-bank.no/en/news-events/news/?selectedFacets[Type]=11532" # Goes to the speeches section. 
    }
    
    harvest_norges_bank_urls(JOBS_TO_DO, target_year=1999)