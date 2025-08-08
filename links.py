from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import traceback

def scrape_youtube_channel_selenium(channel_url, max_videos=5, start_position=0):
    """
    使用Selenium爬取YouTube频道视频页面的视频链接
    
    Args:
        channel_url: 频道视频页面的URL
        max_videos: 最大爬取视频数量
        start_position: 起始位置（用于多次爬取时避免重复）
    
    Returns:
        视频链接列表
    """
    # 设置Chrome选项
    chrome_options = Options()
    # 如果要看到浏览器界面，可以注释掉下面这行
    # chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    # 添加更多选项以避免检测
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option("useAutomationExtension", False)
    
    # 初始化浏览器
    driver = webdriver.Chrome(options=chrome_options)
    # 修改 navigator.webdriver 标志，使其更难被检测
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    
    all_video_links = []
    
    try:
        # 访问频道页面
        driver.get(channel_url)
        print(f"正在爬取频道: {channel_url}")
        # 增加等待时间
        time.sleep(5)  # 等待页面完全加载
        
        # 如果指定了起始位置，先滚动到该位置
        if start_position > 0:
            print(f"🔄 滚动到起始位置: {start_position}")
            for i in range(start_position // 10):  # 每10个视频滚动一次
                driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
                time.sleep(2)
        
        # 持续滚动直到找到足够的视频或者没有新视频加载
        last_count = 0
        scroll_attempts = 0
        max_scroll_attempts = 10  # 设置最大滚动尝试次数
        
        while len(all_video_links) < max_videos and scroll_attempts < max_scroll_attempts:
            # 打印当前页面高度
            height = driver.execute_script("return document.documentElement.scrollHeight")
            print(f"当前页面高度: {height}, 尝试滚动次数: {scroll_attempts+1}")
            
            # 滚动到页面底部以加载更多视频
            driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
            time.sleep(3)  # 增加等待时间，给页面更多加载时间
            
            # 尝试不同的选择器来查找视频元素
            try:
                # 尝试方法1：通过视频标题链接
                video_elements = driver.find_elements(By.CSS_SELECTOR, "a#video-title")
                print(f"使用方法1找到 {len(video_elements)} 个视频元素")
                
                if not video_elements:
                    # 尝试方法2：通过缩略图链接
                    video_elements = driver.find_elements(By.CSS_SELECTOR, "a.yt-simple-endpoint.ytd-thumbnail")
                    print(f"使用方法2找到 {len(video_elements)} 个视频元素")
                
                if not video_elements:
                    # 尝试方法3：任何可能包含视频链接的a标签
                    video_elements = driver.find_elements(By.XPATH, "//a[contains(@href, '/watch?v=')]")
                    print(f"使用方法3找到 {len(video_elements)} 个视频元素")
                
                # 提取视频链接
                new_links_found = 0
                for element in video_elements:
                    href = element.get_attribute('href')
                    if href and '/watch?v=' in href and href not in all_video_links:
                        all_video_links.append(href)
                        new_links_found += 1
                        print(f"找到视频 ({len(all_video_links)}): {href}")
                        
                        # 如果达到最大视频数量，停止爬取
                        if len(all_video_links) >= max_videos:
                            break
                
                print(f"本次滚动找到 {new_links_found} 个新视频链接")
                
                # 检查是否有新视频加载
                if len(video_elements) == last_count and new_links_found == 0:
                    scroll_attempts += 1
                    print(f"没有新视频加载，尝试再次滚动... ({scroll_attempts}/{max_scroll_attempts})")
                else:
                    scroll_attempts = 0  # 如果找到新内容，重置尝试计数
                
                last_count = len(video_elements)
                
            except Exception as e:
                print(f"查找视频元素时出错: {e}")
                traceback.print_exc()
                scroll_attempts += 1
        
        if not all_video_links:
            print("尝试直接从页面源代码中提取视频链接...")
            page_source = driver.page_source
            import re
            # 尝试从页面源代码中提取视频ID
            video_ids = re.findall(r'"/watch\?v=([^"]+)"', page_source)
            for video_id in set(video_ids):  # 使用set去重
                video_url = f"https://www.youtube.com/watch?v={video_id}"
                if video_url not in all_video_links:
                    all_video_links.append(video_url)
                    print(f"从源代码找到视频: {video_url}")
                    if len(all_video_links) >= max_videos:
                        break
        
    except Exception as e:
        print(f"爬取时出错: {e}")
        traceback.print_exc()
    finally:
        # 获取页面截图以便于调试
        try:
            driver.save_screenshot("youtube_debug.png")
            print("已保存页面截图到 youtube_debug.png")
        except:
            pass
        
        # 关闭浏览器
        driver.quit()
    
    return all_video_links

def save_links_to_file(links, filename='ted_videos.txt'):
    """将链接保存到文本文件"""
    with open(filename, 'w', encoding='utf-8') as file:
        for link in links:
            file.write(f"{link}\n")
    print(f"已将 {len(links)} 个视频链接保存到 {filename}")

def main():
    channel_url = "https://www.youtube.com/@TED/videos"
    output_file = input("请输入保存结果的文件名 (默认为ted_videos.txt): ") or "ted_videos.txt"
    max_videos = int(input("请输入要爬取的最大视频数量: ") or "5")
    
    print(f"开始爬取TED频道视频链接...")
    video_links = scrape_youtube_channel_selenium(channel_url, max_videos)
    
    if video_links:
        print(f"总共找到 {len(video_links)} 个视频链接")
        save_links_to_file(video_links, output_file)
    else:
        print("没有找到任何视频链接")

if __name__ == "__main__":
    main()