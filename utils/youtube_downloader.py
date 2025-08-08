import yt_dlp
import os
from pathlib import Path
from typing import List, Dict
import pandas as pd
from tqdm import tqdm

class YouTubeDownloader:
    def __init__(self, output_dir: str = "downloads"):
        """
        初始化YouTube下载器
        
        Args:
            output_dir: 下载视频的目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # yt-dlp配置
        self.ydl_opts = {
            'format': 'best[height<=720]',  # 下载720p或更低质量
            'outtmpl': str(self.output_dir / '%(id)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
            'extractaudio': False,
            'audioformat': 'mp3',
        }
    
    def download_video(self, url: str) -> str:
        """
        下载单个视频
        
        Args:
            url: YouTube视频URL
            
        Returns:
            下载的视频文件路径
        """
        try:
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                video_id = info['id']
                ext = info['ext']
                video_path = self.output_dir / f"{video_id}.{ext}"
                return str(video_path)
        except Exception as e:
            print(f"下载失败 {url}: {e}")
            return None
    
    def download_from_csv(self, csv_path: str, max_videos: int = None) -> List[str]:
        """
        从CSV文件下载视频
        
        Args:
            csv_path: 包含URL的CSV文件路径
            max_videos: 最大下载视频数量
            
        Returns:
            下载的视频文件路径列表
        """
        # 读取CSV文件
        df = pd.read_csv(csv_path)
        
        # 提取唯一的URL（去掉时间戳参数）
        urls = []
        for url in df['URL']:
            # 清理URL，去掉时间戳参数
            clean_url = url.split('&t=')[0] if '&t=' in url else url
            if clean_url not in urls:
                urls.append(clean_url)
        
        if max_videos:
            urls = urls[:max_videos]
        
        print(f"准备下载 {len(urls)} 个视频...")
        
        downloaded_paths = []
        for url in tqdm(urls, desc="下载视频"):
            video_path = self.download_video(url)
            if video_path:
                downloaded_paths.append(video_path)
        
        print(f"成功下载 {len(downloaded_paths)} 个视频")
        return downloaded_paths
    
    def download_from_list(self, urls: List[str], max_videos: int = None) -> List[str]:
        """
        从URL列表下载视频
        
        Args:
            urls: YouTube视频URL列表
            max_videos: 最大下载视频数量
            
        Returns:
            下载的视频文件路径列表
        """
        if max_videos:
            urls = urls[:max_videos]
        
        print(f"准备下载 {len(urls)} 个视频...")
        
        downloaded_paths = []
        for url in tqdm(urls, desc="下载视频"):
            video_path = self.download_video(url)
            if video_path:
                downloaded_paths.append(video_path)
        
        print(f"成功下载 {len(downloaded_paths)} 个视频")
        return downloaded_paths 