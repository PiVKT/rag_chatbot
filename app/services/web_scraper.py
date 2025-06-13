import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Set
import time
import logging
from dataclasses import dataclass
import posixpath

logger = logging.getLogger(__name__)

@dataclass
class ScrapedContent:
    url: str
    title: str
    content: str
    metadata: Dict[str, str]

class WebScraper:
    def __init__(self, delay: float = 1.0):
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Danh sách các extension cần loại bỏ
        self.ignored_extensions = {
            # documents
            'pdf', 'doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx', 'pps', 'odt', 'ods', 'odg', 'odp',
            # archives
            '7z', '7zip', 'bz2', 'rar', 'tar', 'tar.gz', 'xz', 'zip',
            # images
            'jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'svg', 'ico', 'webp',
            # audio/video
            'mp3', 'mp4', 'avi', 'mov', 'wmv', 'flv', 'webm', 'wav', 'ogg',
            # other
            'css', 'js', 'exe', 'bin', 'dmg', 'iso', 'apk'
        }
    
    def _is_valid_url(self, url: str) -> bool:
        """
        Kiểm tra xem URL có hợp lệ để scrape hay không
        
        Args:
            url: URL cần kiểm tra
            
        Returns:
            bool: True nếu URL hợp lệ, False nếu cần loại bỏ
        """
        try:
            parsed_url = urlparse(url)
            # Lấy extension từ path
            extension = posixpath.splitext(parsed_url.path)[1].lower().lstrip('.')
            
            # Nếu có extension và nằm trong danh sách loại bỏ
            if extension and extension in self.ignored_extensions:
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error checking URL validity {url}: {str(e)}")
            return True  # Nếu có lỗi, vẫn cho phép scrape
    
    def scrape_url(self, url: str) -> ScrapedContent:
        """
        Scrape nội dung từ một URL
        
        Args:
            url: URL cần scrape
            
        Returns:
            ScrapedContent: Nội dung đã scrape
        """
        # Kiểm tra URL trước khi scrape
        if not self._is_valid_url(url):
            raise ValueError(f"URL not valid for scraping: {url}")
            
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Lấy title
            title = soup.find('title')
            title = title.get_text().strip() if title else "Untitled"
            
            # Loại bỏ script và style
            for script in soup(["script", "style", "nav", "footer", "aside"]):
                script.decompose()
            
            # Lấy text chính
            content = soup.get_text()
            content = self._clean_text(content)
            
            # Metadata
            metadata = {
                "scraped_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "domain": urlparse(url).netloc,
                "content_length": str(len(content))
            }
            
            return ScrapedContent(url, title, content, metadata)
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            raise
    
    def scrape_website(self, start_url: str, max_depth: int = 2, max_pages: int = 10) -> List[ScrapedContent]:
        """
        Scrape toàn bộ website theo độ sâu
        
        Args:
            start_url: URL bắt đầu
            max_depth: Độ sâu tối đa
            max_pages: Số trang tối đa
            
        Returns:
            List[ScrapedContent]: Danh sách nội dung đã scrape
        """
        visited: Set[str] = set()
        to_visit: List[tuple] = [(start_url, 0)]  # (url, depth)
        results: List[ScrapedContent] = []
        
        base_domain = urlparse(start_url).netloc
        count = 0
        while to_visit and len(results) < max_pages:
            url, depth = to_visit.pop(0)
            
            if url in visited or depth > max_depth:
                continue
            
            # Kiểm tra URL có hợp lệ không
            if not self._is_valid_url(url):
                continue
                
            count += 1
            visited.add(url)
            
            try:
                # Scrape trang hiện tại
                content = self.scrape_url(url)
                results.append(content)
                logger.info(f"Scraped: {count} - {url}")
                
                # Tìm links mới nếu chưa đạt max depth
                if depth < max_depth:
                    new_links = self._extract_links(url, base_domain)
                    for link in new_links:
                        if link not in visited and self._is_valid_url(link):
                            to_visit.append((link, depth + 1))
                
                time.sleep(self.delay)
                
            except Exception as e:
                logger.error(f"Failed to scrape {url}: {str(e)}")
                continue
        
        return results
    
    def _extract_links(self, url: str, base_domain: str) -> List[str]:
        """Trích xuất links từ trang"""
        try:
            response = self.session.get(url, timeout=30)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                full_url = urljoin(url, href)
                
                # Chỉ lấy links cùng domain và hợp lệ
                if (urlparse(full_url).netloc == base_domain and 
                    self._is_valid_url(full_url)):
                    links.append(full_url)
            
            return list(set(links))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Error extracting links from {url}: {str(e)}")
            return []
    
    def _clean_text(self, text: str) -> str:
        """Làm sạch text"""
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line and len(line) > 10:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
