# utils/smart_downloader.py
import yt_dlp
import os
import time
from pathlib import Path
import re
from datetime import datetime, timedelta

# 尝试导入browser_cookie3以自动获取cookies
try:
    import browser_cookie3
    COOKIES_AVAILABLE = True
    print("✅ 检测到browser_cookie3，将自动使用浏览器cookies")
except ImportError:
    COOKIES_AVAILABLE = False
    print("⚠️ 未安装browser_cookie3，可能遇到YouTube下载限制")

class SmartVideoDownloader:
    """智能视频下载器：只下载指定时间段前后的视频片段"""
    
    def __init__(self, output_dir="downloads"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.cookies_jar = self._get_browser_cookies()
    
    def _get_browser_cookies(self):
        """尝试从浏览器获取YouTube cookies"""
        if not COOKIES_AVAILABLE:
            return None
        
        try:
            # 尝试从Chrome获取cookies
            print("🍪 正在从Chrome浏览器获取YouTube cookies...")
            cookies = browser_cookie3.chrome(domain_name='youtube.com')
            if cookies:
                print("✅ 成功获取Chrome cookies")
                return cookies
        except Exception as e:
            print(f"⚠️ Chrome cookies获取失败: {e}")
        
        try:
            # 尝试从Edge获取cookies
            print("🍪 正在从Edge浏览器获取YouTube cookies...")
            cookies = browser_cookie3.edge(domain_name='youtube.com')
            if cookies:
                print("✅ 成功获取Edge cookies")
                return cookies
        except Exception as e:
            print(f"⚠️ Edge cookies获取失败: {e}")
        
        try:
            # 尝试从Firefox获取cookies
            print("🍪 正在从Firefox浏览器获取YouTube cookies...")
            cookies = browser_cookie3.firefox(domain_name='youtube.com')
            if cookies:
                print("✅ 成功获取Firefox cookies")
                return cookies
        except Exception as e:
            print(f"⚠️ Firefox cookies获取失败: {e}")
        
        print("❌ 无法获取浏览器cookies，将尝试无认证下载")
        return None
    
    def _get_ydl_opts(self, output_template, segment_mode=False):
        """获取yt-dlp配置选项，包括cookies支持"""
        base_opts = {
            'format': 'best[height<=720]',  # 限制分辨率以提高速度
            'outtmpl': output_template,
            'quiet': False,
            'no_warnings': False,
            'progress_hooks': [self._progress_hook],
            # 增强的反检测选项
            'extractor_args': {
                'youtube': {
                    'player_client': ['android', 'web'],
                    'skip': ['hls', 'dash']
                }
            },
            # 网络选项
            'retries': 3,
            'timeout': 30,
            'socket_timeout': 30,
            # 用户代理
            'headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
        }
        
        # 如果有cookies，添加到配置中
        if self.cookies_jar:
            try:
                # 创建临时cookies文件
                cookies_file = self.output_dir / "temp_cookies.txt"
                with open(cookies_file, 'w') as f:
                    f.write("# Netscape HTTP Cookie File\n")
                    for cookie in self.cookies_jar:
                        if hasattr(cookie, 'domain') and 'youtube' in cookie.domain:
                            f.write(f"{cookie.domain}\tTRUE\t{cookie.path}\t{'TRUE' if cookie.secure else 'FALSE'}\t{cookie.expires or 0}\t{cookie.name}\t{cookie.value}\n")
                
                base_opts['cookiefile'] = str(cookies_file)
                print(f"🍪 使用cookies文件: {cookies_file}")
            except Exception as e:
                print(f"⚠️ Cookies配置失败: {e}")
        
        return base_opts
    
    def parse_timecode(self, timecode):
        """解析时间码 (HH:MM:SS.mmm) 为秒数"""
        parts = timecode.split(':')
        if len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        return 0
    
    def _find_local_videos(self):
        """查找downloads文件夹中的所有视频文件"""
        video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv', '*.webm']
        local_videos = []
        
        for extension in video_extensions:
            local_videos.extend(self.output_dir.glob(extension))
        
        # 按修改时间排序，最新的在前
        local_videos.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        if local_videos:
            print(f"   找到 {len(local_videos)} 个本地视频文件:")
            for i, video in enumerate(local_videos[:5], 1):  # 只显示前5个
                file_size = video.stat().st_size / (1024*1024)  # MB
                print(f"   {i}. {video.name} ({file_size:.1f}MB)")
            if len(local_videos) > 5:
                print(f"   ... 还有 {len(local_videos)-5} 个文件")
        
        return local_videos
    
    def download_video_segment(self, url, start_timecode, end_timecode, 
                              buffer_seconds=60, max_duration=300):
        """
        下载视频片段或完整视频 - 支持本地文件
        
        Args:
            url: YouTube URL 或本地文件路径
            start_timecode: 开始时间码 (HH:MM:SS.mmm) - 下载完整视频时忽略
            end_timecode: 结束时间码 (HH:MM:SS.mmm) - 下载完整视频时忽略
            buffer_seconds: 前后缓冲时间（秒） - 下载完整视频时忽略
            max_duration: 最大下载时长（秒），0表示下载完整视频
        """
        print(f"🎬 开始处理视频: {url}")
        
        # 检查是否使用本地视频文件
        if url.startswith("downloads/") or not url.startswith("http"):
            return self._process_local_video_file(url, start_timecode, end_timecode, buffer_seconds, max_duration)
        
        # 判断是否下载完整视频
        if max_duration == 0:
            print(f"   模式: 下载完整视频")
            return self._download_full_video(url)
        else:
            print(f"   模式: 下载视频片段")
            print(f"   时间段: {start_timecode} - {end_timecode}")
            return self._download_video_segment_with_cutting(url, start_timecode, end_timecode, buffer_seconds, max_duration)
    
    def _download_full_video(self, url):
        """下载完整视频 - 优先使用本地文件"""
        # 提取视频ID
        video_id = self._extract_video_id(url)
        if not video_id:
            print("❌ 无法提取视频ID")
            return None
        
        # 生成输出文件名
        output_filename = f"{video_id}_full_video.mp4"
        output_path = self.output_dir / output_filename
        
        # 优先检查本地是否已有对应的文件
        if output_path.exists():
            print(f"✅ 找到本地文件: {output_filename}")
            return str(output_path)
        
        # 🔧 修复：检查是否有对应视频ID的其他格式文件
        print("🔍 检查本地视频文件...")
        local_videos = self._find_local_videos()
        for video_file in local_videos:
            # 检查文件名是否包含当前视频ID
            if video_id in video_file.name:
                print(f"✅ 找到匹配的本地视频: {video_file.name}")
                return str(video_file)
        
        print(f"📥 本地无对应视频({video_id})，开始从网络下载...")
        
        # 配置yt-dlp选项 - 支持cookies和反检测
        ydl_opts = self._get_ydl_opts(str(output_path).replace('.mp4', '_temp.%(ext)s'))

        try:
            print("   步骤: 下载完整视频...")
            temp_video_path = None
            
            # 添加重试机制
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        ydl.download([url])
                        
                        # 查找下载的临时文件
                        temp_files = list(self.output_dir.glob(f"{video_id}_full_video_temp.*"))
                        if temp_files:
                            temp_video_path = temp_files[0]
                            print(f"   ✅ 完整视频下载成功: {temp_video_path.name}")
                            break  # 下载成功，跳出重试循环
                        else:
                            print("   ❌ 未找到下载的视频文件")
                            retry_count += 1
                            if retry_count < max_retries:
                                print(f"   🔄 第 {retry_count} 次重试...")
                                time.sleep(2)  # 等待2秒后重试
                            continue
                            
                except Exception as e:
                    retry_count += 1
                    print(f"   ❌ 下载失败 (尝试 {retry_count}/{max_retries}): {e}")
                    if retry_count < max_retries:
                        print(f"   🔄 等待5秒后重试...")
                        time.sleep(5)  # 等待5秒后重试
                    else:
                        print(f"   ❌ 所有重试都失败了")
                        break
            
            # 重命名临时文件为最终文件
            if temp_video_path and temp_video_path.exists():
                final_path = str(temp_video_path).replace('_temp.', '.')
                temp_video_path.rename(final_path)
                print(f"   ✅ 完整视频保存成功: {Path(final_path).name}")
                return final_path
            else:
                print("   ❌ 视频文件处理失败")
                return None
                        
        except Exception as e:
            print(f"❌ 下载失败: {e}")
            return None
    
    def _download_video_segment_with_cutting(self, url, start_timecode, end_timecode, buffer_seconds, max_duration):
        """下载并切割视频片段"""
        # 解析时间码
        start_seconds = self.parse_timecode(start_timecode)
        end_seconds = self.parse_timecode(end_timecode)
        
        # 计算下载范围（前后各加buffer_seconds）
        download_start = max(0, start_seconds - buffer_seconds)
        download_end = end_seconds + buffer_seconds
        
        # 限制最大时长
        actual_duration = download_end - download_start
        if actual_duration > max_duration:
            # 如果超过最大时长，优先保留目标时间段
            target_duration = end_seconds - start_seconds
            remaining_time = max_duration - target_duration
            buffer_seconds = remaining_time // 2
            download_start = max(0, start_seconds - buffer_seconds)
            download_end = min(download_end, download_start + max_duration)
        
        print(f"   下载范围: {download_start:.1f}s - {download_end:.1f}s (时长: {download_end - download_start:.1f}s)")
        
        # 提取视频ID
        video_id = self._extract_video_id(url)
        if not video_id:
            print("❌ 无法提取视频ID")
            return None
        
        # 生成输出文件名
        output_filename = f"{video_id}_segment_{start_seconds:.0f}_{end_seconds:.0f}.mp4"
        output_path = self.output_dir / output_filename
        
        # 如果文件已存在，跳过下载
        if output_path.exists():
            print(f"✅ 文件已存在: {output_filename}")
            return str(output_path)
        
        # 配置yt-dlp选项 - 支持cookies和反检测
        ydl_opts = self._get_ydl_opts(str(output_path).replace('.mp4', '_temp.%(ext)s'), segment_mode=True)

        try:
            # 第一步：下载完整视频（带重试）
            print("   步骤1: 下载完整视频...")
            temp_video_path = None
            
            # 添加重试机制
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        ydl.download([url])
                        
                        # 查找下载的临时文件
                        temp_files = list(self.output_dir.glob(f"{video_id}_segment_*_temp.*"))
                        if temp_files:
                            temp_video_path = temp_files[0]
                            print(f"   ✅ 完整视频下载成功: {temp_video_path.name}")
                            break  # 下载成功，跳出重试循环
                        else:
                            print("   ❌ 未找到下载的视频文件")
                            retry_count += 1
                            if retry_count < max_retries:
                                print(f"   🔄 第 {retry_count} 次重试...")
                                time.sleep(2)  # 等待2秒后重试
                            continue
                            
                except Exception as e:
                    retry_count += 1
                    print(f"   ❌ 下载失败 (尝试 {retry_count}/{max_retries}): {e}")
                    if retry_count < max_retries:
                        print(f"   🔄 等待5秒后重试...")
                        time.sleep(5)  # 等待5秒后重试
                    else:
                        print(f"   ❌ 所有重试都失败了")
                        break
            
            if not temp_video_path or not temp_video_path.exists():
                print("   ❌ 视频下载失败")
                return None
            
            # 第二步：使用ffmpeg切割视频片段
            print("   步骤2: 切割视频片段...")
            import subprocess
            
            # 构建ffmpeg命令
            ffmpeg_cmd = [
                'ffmpeg',
                '-i', str(temp_video_path),
                '-ss', str(download_start),
                '-t', str(download_end - download_start),
                '-c', 'copy',
                '-avoid_negative_ts', 'make_zero',
                '-y',  # 覆盖输出文件
                str(output_path)
            ]
            
            try:
                result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=120)
                
                if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
                    print(f"   ✅ 视频切割成功: {output_filename}")
                    # 删除临时文件
                    if temp_video_path.exists():
                        temp_video_path.unlink()
                    return str(output_path)
                else:
                    print(f"   ❌ 视频切割失败: {result.stderr}")
                    return None
                    
            except subprocess.TimeoutExpired:
                print("   ❌ 视频切割超时")
                return None
            except FileNotFoundError:
                print("   ⚠️ 未找到ffmpeg，将使用完整视频文件")
                # 如果ffmpeg不可用，返回完整视频路径
                if temp_video_path and temp_video_path.exists():
                    final_path = str(temp_video_path).replace('_temp.', '.')
                    temp_video_path.rename(final_path)
                    print(f"   ✅ 使用完整视频: {Path(final_path).name}")
                    return final_path
                return None
            finally:
                # 清理临时文件
                if temp_video_path and temp_video_path.exists():
                    try:
                        temp_video_path.unlink()
                    except:
                        pass
                        
        except Exception as e:
            print(f"❌ 下载失败: {e}")
            return None
    
    def _extract_video_id(self, url):
        """从YouTube URL中提取视频ID"""
        patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]+)',
            r'youtube\.com/watch\?.*v=([a-zA-Z0-9_-]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    
    def _progress_hook(self, d):
        """下载进度回调"""
        if not isinstance(d, dict):
            return
        
        status = d.get('status')
        if status == 'downloading':
            # 安全地获取下载进度信息
            downloaded = d.get('downloaded_bytes', 0)
            total = d.get('total_bytes') or d.get('total_bytes_estimate', 0)
            
            if total and total > 0:
                percent = downloaded / total * 100
                print(f"   下载进度: {percent:.1f}%")
            elif downloaded > 0:
                print(f"   已下载: {downloaded / 1024 / 1024:.1f}MB")
                
        elif status == 'finished':
            print("   下载完成，正在处理...")
    
    def download_from_benchmark_file(self, benchmark_file, max_videos=1):
        """从benchmark文件下载视频 - 支持完整视频下载"""
        print(f"📋 从benchmark文件读取: {benchmark_file}")
        
        downloaded_videos = []
        
        with open(benchmark_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 跳过标题行
        for i, line in enumerate(lines[1:], 1):
            if i > max_videos:
                break
                
            parts = line.strip().split(',')
            if len(parts) >= 1:  # 只需要URL
                url = parts[0]
                start_timecode = parts[1] if len(parts) > 1 else '00:00:00.000'
                end_timecode = parts[2] if len(parts) > 2 else '99:59:59.999'
                
                print(f"\n📥 处理第 {i} 个视频:")
                video_path = self.download_video_segment(url, start_timecode, end_timecode, max_duration=0)
                
                if video_path:
                    downloaded_videos.append({
                        'url': url,
                        'start_timecode': start_timecode,
                        'end_timecode': end_timecode,
                        'video_path': video_path
                    })
        
        print(f"\n✅ 完成下载 {len(downloaded_videos)} 个完整视频")
        return downloaded_videos
    
    def _process_local_video_file(self, url, start_timecode, end_timecode, buffer_seconds, max_duration):
        """处理本地视频文件 - 根据max_duration决定是否处理完整视频"""
        print(f"📂 使用本地视频文件: {url}")
        
        # 检查文件是否存在
        video_path = Path(url)
        if not video_path.exists():
            print(f"❌ 本地视频文件不存在: {url}")
            return None
        
        print(f"   源视频: {video_path.name}")
        
        # 根据max_duration决定处理模式
        if max_duration == 0:
            print(f"   模式: 处理完整视频")
            print(f"   ✅ 使用完整视频文件进行处理")
        else:
            print(f"   模式: 处理视频片段")
            print(f"   时间段信息: {start_timecode} - {end_timecode} (仅用于记录，实际处理完整视频)")
            print(f"   ⚠️ 注意: 本地文件模式下仍处理完整视频，时间段参数仅供参考")
        
        # 直接返回完整视频路径，不进行切割
        return str(video_path) 