import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Set
import time
import logging
from dataclasses import dataclass

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
    
    def scrape_url(self, url: str) -> ScrapedContent:
        """
        Scrape nội dung từ một URL
        
        Args:
            url: URL cần scrape
            
        Returns:
            ScrapedContent: Nội dung đã scrape
        """
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
        
        while to_visit and len(results) < max_pages:
            url, depth = to_visit.pop(0)
            
            if url in visited or depth > max_depth:
                continue
                
            visited.add(url)
            
            try:
                # Scrape trang hiện tại
                content = self.scrape_url(url)
                results.append(content)
                logger.info(f"Scraped: {url}")
                
                # Tìm links mới nếu chưa đạt max depth
                if depth < max_depth:
                    new_links = self._extract_links(url, base_domain)
                    for link in new_links:
                        if link not in visited:
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
                
                # Chỉ lấy links cùng domain
                if urlparse(full_url).netloc == base_domain:
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
            if line and len(line) > 10:  # Loại bỏ dòng quá ngắn
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)