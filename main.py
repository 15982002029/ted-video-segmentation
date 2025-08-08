#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TED视频分割处理程序 - 完整端到端解决方案
集成爬虫+处理+安全扩展功能
"""

import argparse
import sys
import os
import shutil
import csv
from pathlib import Path
from datetime import datetime
from typing import Set, List, Tuple
import subprocess
import time

# 导入现有的处理模块
from video_segmentation_pipeline import VideoSegmentationPipeline

# 导入爬虫功能
try:
    from links import scrape_youtube_channel_selenium, save_links_to_file
    CRAWLER_AVAILABLE = True
except ImportError:
    CRAWLER_AVAILABLE = False
    print("⚠️  警告：爬虫模块不可用，将跳过爬虫功能")

class SafeVideoProcessor:
    """安全视频处理器 - 支持备份、重复检测、增量处理"""
    
    def __init__(self):
        self.backup_dir = Path("backups")
        self.backup_dir.mkdir(exist_ok=True)
        self.processed_videos_file = Path("processed_videos.txt")
        self.processed_videos = self._load_processed_videos()
    
    def _load_processed_videos(self) -> Set[str]:
        """从filtered_ted_segments.txt加载已处理的视频ID列表"""
        processed = set()
        segments_file = Path("filtered_ted_segments.txt")
        if segments_file.exists():
            with open(segments_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('URL') and line:  # 跳过标题行和空行
                        # 从CSV行中提取URL（第一列）
                        url = line.split(',')[0] if ',' in line else line
                        if url.startswith('http'):
                            video_id = self._extract_video_id(url)
                            if video_id:
                                processed.add(video_id)
        return processed
    
    def _extract_video_id(self, url: str) -> str:
        """从YouTube URL中提取视频ID"""
        if 'youtube.com/watch?v=' in url:
            return url.split('watch?v=')[1].split('&')[0]
        elif 'youtu.be/' in url:
            return url.split('youtu.be/')[1].split('?')[0]
        return url  # 如果不是YouTube链接，返回原URL
    
    def create_backup(self, files_to_backup: List[str]) -> str:
        """创建备份"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"backup_{timestamp}"
        backup_path = self.backup_dir / backup_name
        backup_path.mkdir(exist_ok=True)
        
        print(f"🛡️  创建备份: {backup_path}")
        for file_path in files_to_backup:
            if Path(file_path).exists():
                shutil.copy2(file_path, backup_path / Path(file_path).name)
                print(f"   ✅ 备份: {file_path}")
        
        return str(backup_path)
    
    def check_duplicates(self, input_file: str) -> Tuple[List[str], List[str]]:
        """检查重复视频，返回新视频和重复视频列表"""
        new_videos = []
        duplicate_videos = []
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                url = line.strip()
                if url:
                    video_id = self._extract_video_id(url)
                    if video_id in self.processed_videos:
                        duplicate_videos.append(url)
                    else:
                        new_videos.append(url)
        
        return new_videos, duplicate_videos
    
    def create_filtered_input(self, input_file: str, new_videos: List[str]) -> str:
        """创建过滤后的输入文件"""
        filtered_input = f"temp_filtered_{Path(input_file).name}"
        with open(filtered_input, 'w', encoding='utf-8') as f:
            for url in new_videos:
                f.write(f"{url}\n")
        return filtered_input
    
    def update_processed_videos(self, processed_urls: List[str]):
        """更新已处理视频记录（现在主要依赖filtered_ted_segments.txt，但保持兼容性）"""
        for url in processed_urls:
            video_id = self._extract_video_id(url)
            self.processed_videos.add(video_id)
        
        # 可选：仍然保存到processed_videos.txt以保持兼容性
        with open(self.processed_videos_file, 'w', encoding='utf-8') as f:
            for video_id in sorted(self.processed_videos):
                f.write(f"{video_id}\n")
        
        print(f"📝 已处理视频记录已更新（主要数据源：filtered_ted_segments.txt）")

class VideoCrawler:
    """视频爬虫类"""
    
    def __init__(self):
        if not CRAWLER_AVAILABLE:
            raise ImportError("爬虫模块不可用，请确保links.py文件存在")
    
    def _extract_video_id(self, url: str) -> str:
        """从YouTube URL中提取视频ID"""
        if 'youtube.com/watch?v=' in url:
            return url.split('watch?v=')[1].split('&')[0]
        elif 'youtu.be/' in url:
            return url.split('youtu.be/')[1].split('?')[0]
        return ""
    
    def crawl_ted_videos(self, max_videos: int = 10, output_file: str = "scraped_ted_videos.txt") -> str:
        """爬取TED视频链接，直到获得足够的新视频"""
        print(f"📡 开始爬取TED视频链接...")
        print(f"   目标新视频数量: {max_videos}")
        print(f"   输出文件: {output_file}")
        
        try:
            # 从filtered_ted_segments.txt获取已处理的视频ID列表
            processed_videos = set()
            segments_file = Path("filtered_ted_segments.txt")
            if segments_file.exists():
                with open(segments_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('URL') and line:  # 跳过标题行和空行
                            # 从CSV行中提取URL（第一列）
                            url = line.split(',')[0] if ',' in line else line
                            if url.startswith('http'):
                                video_id = self._extract_video_id(url)
                                if video_id:
                                    processed_videos.add(video_id)
            
            print(f"📊 已处理视频数: {len(processed_videos)}")
            if processed_videos:
                print(f"   已处理的视频ID示例: {list(processed_videos)[:3]}{'...' if len(processed_videos) > 3 else ''}")
            
            # 持续爬取直到获得足够的新视频
            all_video_links = []
            new_video_links = []
            crawl_attempts = 0
            max_crawl_attempts = 5  # 最多尝试5次爬取
            
            while len(new_video_links) < max_videos and crawl_attempts < max_crawl_attempts:
                crawl_attempts += 1
                print(f"\n🔄 第 {crawl_attempts} 次爬取尝试...")
                
                # 爬取视频链接，使用不同的起始位置
                channel_url = "https://www.youtube.com/@TED/videos"
                batch_size = max(10, max_videos - len(new_video_links))  # 动态调整批次大小
                start_position = (crawl_attempts - 1) * 20  # 每次爬取从不同位置开始
                video_links = scrape_youtube_channel_selenium(channel_url, batch_size, start_position)
                
                if not video_links:
                    print("❌ 本次爬取没有获得任何视频链接")
                    break
                
                # 检查新视频
                batch_new_videos = 0
                for video_url in video_links:
                    if video_url not in all_video_links:  # 避免重复添加
                        all_video_links.append(video_url)
                        
                        # 提取视频ID并检查是否已处理
                        video_id = self._extract_video_id(video_url)
                        if video_id and video_id not in processed_videos:
                            new_video_links.append(video_url)
                            batch_new_videos += 1
                            print(f"🆕 发现新视频 ({len(new_video_links)}/{max_videos}): {video_id}")
                            
                            if len(new_video_links) >= max_videos:
                                break
                        else:
                            print(f"⏭️  跳过已处理视频: {video_id}")
                
                print(f"📊 本次爬取: 获得 {len(video_links)} 个视频，其中 {batch_new_videos} 个新视频")
                
                if len(new_video_links) >= max_videos:
                    break
                
                # 如果还没达到目标，等待一下再继续
                if crawl_attempts < max_crawl_attempts:
                    print(f"⏳ 还需要 {max_videos - len(new_video_links)} 个新视频，继续爬取...")
                    time.sleep(2)  # 短暂等待
            
            if new_video_links:
                # 保存新视频到文件
                save_links_to_file(new_video_links, output_file)
                print(f"✅ 成功获得 {len(new_video_links)} 个新视频链接")
                return output_file
            else:
                print("❌ 经过多次爬取，没有找到足够的新视频")
                return None
                
        except Exception as e:
            print(f"❌ 爬取失败: {e}")
            return None
    
    def create_manual_links_file(self, filename: str) -> bool:
        """创建手动链接文件"""
        sample_links = [
            "https://www.youtube.com/watch?v=uEOK3fk45Rg",
            "https://www.youtube.com/watch?v=p8ReF00JP5w", 
            "https://www.youtube.com/watch?v=eIRtcspIH4U"
        ]
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                for link in sample_links:
                    f.write(f"{link}\n")
            
            print(f"✅ 已创建示例链接文件: {filename}")
            print(f"📊 包含 {len(sample_links)} 个示例链接")
            return True
        except Exception as e:
            print(f"❌ 创建示例文件失败: {e}")
            return False

def get_user_input(prompt: str, default: str = "") -> str:
    """获取用户输入"""
    if default:
        user_input = input(f"{prompt} (默认: {default}): ").strip()
        return user_input if user_input else default
    else:
        return input(f"{prompt}: ").strip()

def get_user_int_input(prompt: str, default: int) -> int:
    """获取用户整数输入"""
    while True:
        try:
            user_input = input(f"{prompt} (默认: {default}): ").strip()
            if not user_input:
                return default
            return int(user_input)
        except ValueError:
            print("❌ 请输入有效的数字")

def run_command(command: str, description: str) -> bool:
    """运行命令并显示进度"""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"{'='*60}")
    print(f"命令: {command}")
    print()
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=False)
        print(f"✅ {description} 完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} 失败: {e}")
        return False

def check_file_exists(filepath: str, description: str) -> bool:
    """检查文件是否存在"""
    if os.path.exists(filepath):
        file_size = os.path.getsize(filepath)
        print(f"✅ {description}: {filepath} (大小: {file_size} 字节)")
        return True
    else:
        print(f"❌ {description}不存在: {filepath}")
        return False

def main():
    """主函数 - 完整的端到端处理流程"""
    parser = argparse.ArgumentParser(description='TED视频分割处理程序 - 完整端到端解决方案（爬虫+处理+安全扩展）')
    
    # 基础参数
    parser.add_argument('--input', type=str, help='输入文件路径（包含YouTube链接）')
    parser.add_argument('--output', type=str, default='filtered_ted_segments.txt', help='输出文件路径')
    parser.add_argument('--config', type=str, default='configs/segmentation_parameters.yaml', help='配置文件路径')
    parser.add_argument('--max-videos', type=int, default=10, help='最大处理视频数量')
    parser.add_argument('--output-format', choices=['echomimicv2', 'segments', 'both'], default='echomimicv2', help='输出格式')
    
    # 爬虫参数
    parser.add_argument('--crawl', action='store_true', help='启用爬虫功能，自动爬取TED视频链接')
    parser.add_argument('--crawl-max', type=int, default=10, help='爬取的最大视频数量')
    parser.add_argument('--crawl-output', type=str, default='scraped_ted_videos.txt', help='爬虫输出文件名')
    
    # 安全参数
    parser.add_argument('--disable-safety', action='store_true', help='禁用安全模式（不备份，不检测重复，直接处理所有视频）')
    parser.add_argument('--force-overwrite', action='store_true', help='强制覆盖现有输出文件（谨慎使用）')
    
    # 本地视频处理
    parser.add_argument('--local-videos', action='store_true', help='处理本地视频文件（downloads文件夹）')
    
    # 工作流程模式
    parser.add_argument('--workflow', action='store_true', help='启用完整工作流程模式（爬虫+处理）')
    
    # 交互模式
    parser.add_argument('--interactive', action='store_true', help='启用交互模式，询问用户参数')
    
    args = parser.parse_args()
    
    print("🚀 TED视频分割处理程序 - 完整端到端解决方案")
    print("="*80)
    print("功能：爬虫 + 处理 + 安全扩展")
    print("="*80)
    
    # 交互模式
    if args.interactive or (not args.input and not args.crawl and not args.workflow and not args.local_videos):
        print("\n🎯 交互模式")
        print("请选择运行模式：")
        print("1. 完整工作流程（爬虫+处理）")
        print("2. 仅爬虫模式")
        print("3. 处理现有文件")
        print("4. 处理本地视频")
        
        choice = get_user_input("请选择 (1-4)", "1")
        
        if choice == "1":
            args.workflow = True
            args.crawl_max = get_user_int_input("请输入要爬取的最大视频数量", 10)
            args.max_videos = get_user_int_input("请输入要处理的最大视频数量", 10)
            args.output = get_user_input("请输入输出文件名", "filtered_ted_segments.txt")
        elif choice == "2":
            args.crawl = True
            args.crawl_max = get_user_int_input("请输入要爬取的最大视频数量", 10)
            args.crawl_output = get_user_input("请输入爬虫输出文件名", "scraped_ted_videos.txt")
        elif choice == "3":
            args.input = get_user_input("请输入输入文件路径")
            args.max_videos = get_user_int_input("请输入要处理的最大视频数量", 10)
            args.output = get_user_input("请输入输出文件名", "filtered_ted_segments.txt")
        elif choice == "4":
            args.local_videos = True
            args.max_videos = get_user_int_input("请输入要处理的最大视频数量", 10)
            args.output = get_user_input("请输入输出文件名", "filtered_ted_segments.txt")
        else:
            print("❌ 无效选择")
            return False
    
    # 检查爬虫可用性
    if args.crawl or args.workflow:
        if not CRAWLER_AVAILABLE:
            print("❌ 爬虫功能不可用，请确保links.py文件存在")
            return False
    
    try:
        # 初始化安全处理器
        safe_processor = None
        if not args.disable_safety:
            safe_processor = SafeVideoProcessor()
            print("🛡️  安全模式已启用")
        else:
            print("⚠️  安全模式已禁用")
        
        # 确定输入文件
        actual_input_file = None
        
        if args.workflow:
            # 完整工作流程模式：爬虫 + 处理
            print("\n🎯 完整工作流程模式")
            
            # 步骤1: 爬取视频链接
            print(f"\n📡 步骤1: 爬取TED视频链接")
            crawler = VideoCrawler()
            crawled_file = crawler.crawl_ted_videos(args.crawl_max, args.crawl_output)
            
            if not crawled_file:
                print("❌ 爬取失败，尝试创建示例文件")
                if not crawler.create_manual_links_file(args.crawl_output):
                    return False
                crawled_file = args.crawl_output
            
            actual_input_file = crawled_file
            
        elif args.crawl:
            # 仅爬虫模式
            print(f"\n📡 爬虫模式")
            crawler = VideoCrawler()
            crawled_file = crawler.crawl_ted_videos(args.crawl_max, args.crawl_output)
            
            if not crawled_file:
                print("❌ 爬取失败")
                return False
            
            print(f"✅ 爬取完成，文件保存为: {crawled_file}")
            return True
            
        elif args.input:
            # 处理现有文件
            actual_input_file = args.input
        else:
            print("❌ 请指定输入文件或启用爬虫功能")
            print("   使用 --input 指定输入文件")
            print("   使用 --crawl 启用爬虫")
            print("   使用 --workflow 启用完整工作流程")
            print("   使用 --interactive 启用交互模式")
            return False
        
        # 检查输入文件
        if actual_input_file and not check_file_exists(actual_input_file, "输入文件"):
            return False
        
        # 安全处理逻辑
        if safe_processor and actual_input_file:
            print(f"\n🛡️  安全处理检查")
            
            # 步骤1: 创建备份
            files_to_backup = []
            if Path(args.output).exists():
                files_to_backup.append(args.output)
            if Path("filtered_ted_segments.txt").exists():
                files_to_backup.append("filtered_ted_segments.txt")
            
            if files_to_backup:
                backup_path = safe_processor.create_backup(files_to_backup)
                print(f"✅ 备份完成: {backup_path}")
            
            # 步骤2: 检查重复
            new_videos, duplicate_videos = safe_processor.check_duplicates(actual_input_file)
            print(f"📊 重复检查结果:")
            print(f"   新视频: {len(new_videos)} 个")
            print(f"   重复视频: {len(duplicate_videos)} 个")
            
            if not new_videos:
                print("✅ 所有视频都已处理过，无需重复处理")
                return True
            
            # 步骤3: 创建过滤后的输入文件
            filtered_input = safe_processor.create_filtered_input(actual_input_file, new_videos)
            actual_input_file = filtered_input
            print(f"📝 创建过滤后的输入文件: {filtered_input}")
        
        # 处理视频
        print(f"\n🎬 开始处理视频...")
        pipeline = VideoSegmentationPipeline(args.config)
        
        # 设置输出文件
        echomimicv2_output_file = args.output if args.output_format in ['echomimicv2', 'both'] else None
        print(f"📄 输出文件设置: {echomimicv2_output_file}")
        print(f"📄 输出格式: {args.output_format}")
        
        if args.local_videos:
            all_results = pipeline.process_local_videos(
                max_videos=args.max_videos,
                echomimicv2_output_file=echomimicv2_output_file
            )
        else:
            all_results = pipeline.process_input_file(
                actual_input_file,
                max_videos=args.max_videos,
                echomimicv2_output_file=echomimicv2_output_file
            )
        
        # 处理结果
        if all_results:
            print(f"\n✅ 处理完成！")
            print(f"📊 处理统计:")
            print(f"   处理视频数: {len(all_results)}")
            
            # 更新已处理视频记录
            if safe_processor and not args.disable_safety:
                processed_urls = []
                for result_key, result in all_results.items():
                    if 'entry_info' in result and 'url' in result['entry_info']:
                        processed_urls.append(result['entry_info']['url'])
                safe_processor.update_processed_videos(processed_urls)
                print(f"✅ 已更新处理记录")
            
            # 生成echomimicv2输出（如果还没有增量写入）
            if args.output_format in ['echomimicv2', 'both'] and not pipeline.current_echomimicv2_file:
                output_file = args.output
                total_duration = 0  # 统计总时长
                
                with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['URL', 'Start Timecode', 'End Timecode'])
                    
                    for result_key, result in all_results.items():
                        if 'methods_results' in result:
                            for method_name, method_result in result['methods_results'].items():
                                if method_result and 'segments' in method_result:
                                    for segment in method_result['segments']:
                                        start_time = segment['start_time']
                                        end_time = segment['end_time']
                                        duration = end_time - start_time
                                        total_duration += duration
                                        
                                        # 转换为时间码格式
                                        start_timecode = pipeline._seconds_to_timecode(start_time)
                                        end_timecode = pipeline._seconds_to_timecode(end_time)
                                        
                                        writer.writerow([
                                            result['entry_info']['url'],
                                            start_timecode,
                                            end_timecode
                                        ])
                
                print(f"✅ 已生成echomimicv2格式输出: {output_file}")
                print(f"📊 总片段时长: {total_duration:.2f}秒 ({total_duration/60:.1f}分钟)")
                
                # 更新报告文件中的总时长统计
                pipeline._update_report_total_duration(total_duration)
            
            # 清理临时文件
            if safe_processor and not args.disable_safety:
                temp_files = [f for f in os.listdir('.') if f.startswith('temp_filtered_')]
                for temp_file in temp_files:
                    try:
                        os.remove(temp_file)
                        print(f"🧹 清理临时文件: {temp_file}")
                    except:
                        pass
        else:
            print(f"\n❌ 没有视频被处理")
    
    except KeyboardInterrupt:
        print(f"\n⚠️  用户中断操作")
    except Exception as e:
        print(f"\n❌ 处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 