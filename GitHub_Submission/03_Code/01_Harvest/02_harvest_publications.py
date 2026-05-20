# While 01_harvest_news.py focused on harvesting press releases and speeches from the news section, this script is dedicated to harvesting URLs from the publications section.
# This section contains the more comprehensive strategic reports, and are published with lower frequency, which makes harvesting slightly less prone for error.
# In terms of package structure, this program is very similar to the news harvesting program.
import pandas as pd
import time
import os
import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# This function is used to handle the cookie banner that appears, and is encapsulated in a Shadow DOM, which sort of hinders interaction.
# Notes in 01_harvest_news.py explain the process of piercing the Shadow DOM, which was one of the first significant hurdles I encountered in the programatic pipeline for the thesis.
def handle_cookie_banner(driver):
    """Pierces the Shadow DOM to accept cookies."""
    try:
        time.sleep(4)
        script = """
        const root = document.querySelector('#usercentrics-root');
        if (root && root.shadowRoot) {
            const acceptBtn = root.shadowRoot.querySelector('button[data-testid="uc-accept-all-button"]');
            if (acceptBtn) { acceptBtn.click(); return "SUCCESS"; }
        }
        return "NOT_FOUND";
        """
        driver.execute_script(script)
    except Exception: # Contrary to the 01_harvest_news.py, I silently pass on any exceptions here, as the backup didnt really provide any value.
        pass

def get_year_from_time_tag(article_link_element): # Extracts the year from the <time> tag associated with a given article.
    """
    Finds the year by locating the <time> tag associated with the link.
    Uses the 'datetime' attribute for perfect accuracy.
    """
    try:
        # Norges Bank usually places the <time> tag as a sibling or in a parent container.
        # look for the nearest 'news-hit__date' class relative to the link.
        parent_hit = article_link_element.find_element(By.XPATH, "./ancestor::*[contains(@class, 'news-hit')]")
        time_tag = parent_hit.find_element(By.CLASS_NAME, "news-hit__date")
        
        # Pull from 'datetime' attribute (such as '2025-11-12T09:30:00')
        dt_attr = time_tag.get_attribute("datetime")
        if dt_attr:
            return int(dt_attr[:4])
        
        # While this approach for finding the exact publication date is more robust than the implementation in 01_harvest_news.py, I include a fallback.
        # In some cases the <time> tag may not be present, which is why I also include a regex-based fallback that looks for 4-digit numbers. 
        date_text = time_tag.text.strip() 
        match = re.search(r'(\d{4})', date_text) # Fallback to regex if datetime attribute is missing
        if match:
            return int(match.group(1))
    except Exception:
        pass
    return None

# The overall setup of this function is very similar to the one in 01_harvest_news.py, with minor adjustments.

def harvest_publications(targets, target_year=1999):
    chrome_options = Options() 
    chrome_options.add_argument("--window-size=1920,1080")
    
    output_dir = os.path.join("02_Data", "urls") 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for category, base_url in targets.items(): 
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options) 
        print(f"\ncategory={category}")
        driver.get(base_url)
        
        collected_data = []
        seen_urls = set()
        try:
            handle_cookie_banner(driver)
            
            # Initial pause to let the JS load the first batch
            time.sleep(3)
            
            searching = True
            last_count = 0
            
            while searching:
                try:
                    # Wait for results to be visible
                    WebDriverWait(driver, 20).until(
                        EC.visibility_of_element_located((By.CLASS_NAME, "news-hit__link"))
                    )
                except Exception:
                    print(f"[{category}] no_items_visible")
                    break

                articles = driver.find_elements(By.CLASS_NAME, "news-hit__link")
                
                # If the number of articles hasn't increased, we might be at the end
                if len(articles) == last_count:
                    break
                last_count = len(articles)

                # Process the newly found articles
                for article in articles:
                    url = article.get_attribute('href')
                    title = article.text.strip()
                    
                    if not url or url in seen_urls:
                        continue

                    # EXTRACT YEAR using the <time> tag logic found
                    year = get_year_from_time_tag(article)
                    
                    # If year is still None, we may have a problem with the DOM structure
                    if year is None:
                        # Final desperate fallback to regex in URL/Title
                        matches = re.findall(r'\d{4}', url + title)
                        valid = [int(m) for m in matches if 1990 <= int(m) <= 2026]
                        year = max(valid) if valid else 2026

                    # If year is none or over 2025, we skip as it may be an error or a publication outside the analysis' range/scope.
                    # Had some bugs with this in the early stages, pulling random years mistakenly from URL for instnace 
                    if year > 2025:
                        continue
                        # Append the data to the collection.
                    seen_urls.add(url)
                    collected_data.append({
                        'date': f"Year: {year}", 
                        'url': url, 
                        'title': title,
                        'category': category
                    })
                    
                    # Check if we hit the 1999 target
                    if year < target_year:
                        print(f"[{category}] target_year_reached={year}")
                        searching = False
                        break

                if not searching:
                    break

                # Handle 'Show more hits' button
                try:
                    load_more_btn = WebDriverWait(driver, 10).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, "button.news__show-more")) # Here we use By.CSS_SELECTOR instead of By.CLASS_NAME as in 01_harvest_news.py.
                    )                                                                           # This approach is slightly more robust, as the by class name appraoch can match to unwanted elements.
                    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", load_more_btn)
                    time.sleep(1)
                    driver.execute_script("arguments[0].click();", load_more_btn)
                    print(f"[{category}] collected={len(collected_data)}")
                    time.sleep(4) # Allow content to populate
                except Exception:
                    print(f"[{category}] end_of_archive")
                    break

        finally:
            driver.quit()
        
        if collected_data:
            df = pd.DataFrame(collected_data)
            filename = os.path.join(output_dir, f"urls_pubs_{category.lower().replace(' ', '_')}.csv")
            df.to_csv(filename, index=False)
            print(f"saved: {filename}")
            print(f"rows={len(df)}")

if __name__ == "__main__":
    PUB_TARGETS = {
        "Monetary Policy Report": "https://www.norges-bank.no/en/news-events/publications/?selectedFacets[Type]=11404&skip=0", # Targets the monetary policy report/MPR.
        "Financial Stability": "https://www.norges-bank.no/en/news-events/publications/?selectedFacets[Type]=11403&skip=0", # The financial stability report/FSR. 
        "Bank Lending Survey": "https://www.norges-bank.no/en/news-events/publications/?selectedFacets[Type]=68544&skip=0" # The bank lending survey/BLS.
    }
    
    harvest_publications(PUB_TARGETS, target_year=1999) # Again, the 1999 target is just an artifact from the initial stages where I was building the pipeline schematic as I went. 
                                                        # See "NB" in 01_harvest_news.py for detail.