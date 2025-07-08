from bs4 import BeautifulSoup, Tag, NavigableString
from typing import List, Optional, Any, Union
from pydantic import BaseModel
import re

class Chunk(BaseModel):
    type: str
    content: str
    heading_path: Optional[str] = ""
    children: Optional[List[Any]] = []

class HierarchicalWebChunker:
    def __init__(self, max_words: int = 1000, overlap_words: int = 200):
        self.max_words = max_words
        self.overlap_words = overlap_words
        
        # Mapping heading levels
        self.heading_map = {
            'h1': 1, 'h2': 2, 'h3': 3, 
            'h4': 4, 'h5': 5, 'h6': 6
        }
        
        # Content tags that should be treated as paragraphs
        self.content_tags = {
            'p', 'div', 'span', 'section', 'article', 
            'ul', 'ol', 'li', 'blockquote', 'pre', 'code'
        }
    
    def chunk_html_content(self, html_content: str, base_url: str = "") -> Chunk:
        """
        Chia nhỏ HTML content thành cấu trúc phân cấp theo headings.
        
        Args:
            html_content: Raw HTML content
            base_url: URL gốc để tạo context
            
        Returns:
            Chunk: Cây phân cấp các phần của trang web
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove unwanted elements
        self._clean_soup(soup)
        
        # Extract page title as root heading
        title = self._extract_title(soup)
        root = Chunk(type="webpage", content=title or "Untitled Page", children=[])
        
        # Current heading hierarchy
        current_headings = [None] * 7  # h0 (title) to h6
        current_headings[0] = root
        
        # Buffer for collecting content
        content_buffer = []
        buffer_owner = root
        heading_path = [root.content] if root.content != "Untitled Page" else []
        
        # Process all elements in order
        body = soup.find('body') or soup
        self._process_elements(body, current_headings, content_buffer, buffer_owner, heading_path)
        
        # Flush remaining content
        self._flush_content_buffer(content_buffer, buffer_owner, heading_path)
        
        return root
    
    def _clean_soup(self, soup: BeautifulSoup):
        """Remove unwanted HTML elements"""
        unwanted_tags = [
            'script', 'style', 'nav', 'footer', 'aside', 
            'header', 'menu', 'form', 'input', 'button',
            'iframe', 'embed', 'object'
        ]
        
        for tag_name in unwanted_tags:
            for tag in soup.find_all(tag_name):
                tag.decompose()
        
        # Remove comments and ads
        for element in soup.find_all(string=lambda text: isinstance(text, str) and 
                                   any(keyword in text.lower() for keyword in ['advertisement', 'ads', 'cookie'])):
            if element.parent:
                element.parent.decompose()
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title"""
        title_tag = soup.find('title')
        if title_tag:
            return self._clean_text(title_tag.get_text())
        
        # Fallback to h1
        h1_tag = soup.find('h1')
        if h1_tag:
            return self._clean_text(h1_tag.get_text())
        
        return "Untitled Page"
    
    def _process_elements(self, container, current_headings, content_buffer, buffer_owner, heading_path):
        """Process all elements recursively"""
        for element in container.children:
            if isinstance(element, NavigableString):
                text = self._clean_text(str(element))
                if text:
                    content_buffer.append(text)
                continue
            
            if not isinstance(element, Tag):
                continue
                
            tag_name = element.name.lower()
            
            # Handle headings
            if tag_name in self.heading_map:
                level = self.heading_map[tag_name]
                heading_text = self._clean_text(element.get_text())
                
                if heading_text:
                    # Flush current content buffer
                    self._flush_content_buffer(content_buffer, buffer_owner, heading_path)
                    
                    # Create new heading chunk
                    new_heading = Chunk(
                        type=f"heading{level}", 
                        content=heading_text, 
                        children=[]
                    )
                    
                    # Find parent heading
                    parent_level = level - 1
                    while parent_level >= 0 and current_headings[parent_level] is None:
                        parent_level -= 1
                    
                    if parent_level >= 0:
                        parent = current_headings[parent_level]
                        parent.children.append(new_heading)
                    
                    # Clear lower level headings
                    for i in range(level, 7):
                        current_headings[i] = None
                    
                    # Set current heading
                    current_headings[level] = new_heading
                    buffer_owner = new_heading
                    
                    # Update heading path
                    heading_path = []
                    for i in range(0, level + 1):
                        if current_headings[i] and current_headings[i].content:
                            heading_path.append(current_headings[i].content)
                    
                    content_buffer = []
            
            # Handle content elements
            elif tag_name in self.content_tags or tag_name in ['text', 'br']:
                text = self._extract_content_text(element)
                if text:
                    content_buffer.append(text)
            
            # Handle nested elements
            else:
                self._process_elements(element, current_headings, content_buffer, buffer_owner, heading_path)
    
    def _extract_content_text(self, element: Tag) -> str:
        """Extract and format text from content elements"""
        if element.name == 'br':
            return '\n'
        
        text = element.get_text(separator=' ', strip=True)
        
        # Special formatting for lists
        if element.name in ['ul', 'ol']:
            items = []
            for li in element.find_all('li', recursive=True):
                item_text = li.get_text(strip=True)
                if item_text:
                    items.append(f"• {item_text}")
            return '\n'.join(items) if items else ''
        
        # Special formatting for code blocks
        if element.name in ['pre', 'code']:
            return f"```\n{text}\n```"
        
        # Special formatting for blockquotes
        if element.name == 'blockquote':
            return f"> {text}"
        
        return self._clean_text(text)
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Remove very short meaningless text
        if len(text) < 3:
            return ""
        
        return text
    
    def _flush_content_buffer(self, content_buffer: List[str], buffer_owner: Chunk, heading_path: List[str]):
        """Flush accumulated content to chunks"""
        if not content_buffer or not buffer_owner:
            return
        
        # Create heading prefix
        heading_prefix = '. '.join(f"{heading} - content" for heading in heading_path) + '. ' if heading_path else ''
        
        # Join all content
        full_content = '\n'.join(content_buffer)
        content_words = full_content.split()
        
        # Calculate available words for content
        prefix_words = heading_prefix.split() if heading_prefix else []
        available_words = self.max_words - len(prefix_words)
        
        if len(content_words) <= available_words:
            # Single chunk
            final_content = heading_prefix + full_content
            buffer_owner.children.append(Chunk(
                type="content_chunk", 
                content=final_content,
                heading_path=' > '.join(heading_path)
            ))
        else:
            # Multiple chunks with overlap
            start = 0
            while start < len(content_words):
                end = min(start + available_words, len(content_words))
                chunk_content = ' '.join(content_words[start:end])
                final_content = heading_prefix + chunk_content
                
                buffer_owner.children.append(Chunk(
                    type="content_chunk", 
                    content=final_content,
                    heading_path=' > '.join(heading_path)
                ))
                
                if end == len(content_words):
                    break
                start = end - self.overlap_words
        
        content_buffer.clear()
    
    def flatten_chunks(self, chunk: Chunk) -> List[Chunk]:
        """
        Flatten hierarchical chunks into list for vector storage
        
        Args:
            chunk: Root chunk
            
        Returns:
            List[Chunk]: Flattened content chunks
        """
        flat_chunks = []
        
        def traverse(node: Chunk):
            if node.type == "content_chunk":
                flat_chunks.append(Chunk(
                    type="content_chunk",
                    content=node.content,
                    heading_path=node.heading_path
                ))
            
            for child in node.children:
                traverse(child)
        
        traverse(chunk)
        return flat_chunks

    def chunk_scraped_content(self, scraped_content) -> List[str]:
        """
        Main method to process ScrapedContent and return chunk texts
        
        Args:
            scraped_content: ScrapedContent object from web scraper
            
        Returns:
            List[str]: List of chunk texts ready for vector storage
        """
        # Build hierarchical structure
        hierarchical_chunk = self.chunk_html_content(scraped_content.content, scraped_content.url)
        
        # Flatten to list of chunks
        flat_chunks = self.flatten_chunks(hierarchical_chunk)
        
        # Extract text content
        chunk_texts = [chunk.content for chunk in flat_chunks]
        
        return chunk_texts