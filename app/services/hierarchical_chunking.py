import logging
from typing import Any, List, Dict, Optional
import pandas as pd
from app.models.schemas import Chunk

logger = logging.getLogger(__name__)

class HierarchicalTextProcessor:
    """
    Xử lý phân mảnh tài liệu theo cấu trúc phân cấp (heading, paragraph, excel row, ...)
    """
    def __init__(self):
        pass

    def chunk_docx(self, doc: Any) -> Chunk:
        """
        Chia nhỏ tài liệu docx thành cấu trúc phân cấp các heading và đoạn văn.
        Mỗi chunk sẽ có format: "Heading1 - content. Heading2 - content. ... paragraph"
        Nếu chunk lớn > 1000 từ, sẽ tách thành nhiều chunk nhỏ hơn, overlap 200 từ.
        Returns:
            Chunk: Cây phân cấp các phần của tài liệu
        """
        from itertools import chain
        root = Chunk(type="docx", content="", children=[])
        current_h1 = None
        current_h2 = None
        current_h3 = None
        current_h4 = None
        # Lưu tạm các paragraph để gom
        para_buffer = None
        buffer_owner = None
        buffer_level = None
        heading_path = []
        
        def flush_buffer():
            nonlocal para_buffer, buffer_owner, buffer_level, heading_path
            if para_buffer and buffer_owner:
                # Tạo prefix từ heading path với format "Heading1 - content. Heading2 - content."
                heading_prefix = '. '.join(f"{heading} " for heading in heading_path) + '. ' if heading_path else ''
                
                # Gộp các paragraph thành 1 đoạn lớn
                full_paragraph = '\n'.join(para_buffer)
                
                # Tách paragraph thành words để kiểm tra độ dài
                paragraph_words = full_paragraph.split()
                max_words = 1000
                overlap = 200
                
                # Tính số từ của heading prefix
                heading_words = heading_prefix.split() if heading_prefix else []
                available_words_for_paragraph = max_words - len(heading_words)
                
                if len(paragraph_words) <= available_words_for_paragraph:
                    # Chunk nhỏ, không cần tách
                    full_text = heading_prefix + full_paragraph
                    buffer_owner.children.append(Chunk(type="paragraph_chunk", content=full_text))
                else:
                    # Chunk lớn, cần tách thành nhiều chunk nhỏ với overlap
                    start = 0
                    while start < len(paragraph_words):
                        end = min(start + available_words_for_paragraph, len(paragraph_words))
                        chunk_paragraph = ' '.join(paragraph_words[start:end])
                        chunk_text = heading_prefix + chunk_paragraph
                        buffer_owner.children.append(Chunk(type="paragraph_chunk", content=chunk_text))
                        
                        if end == len(paragraph_words):
                            break
                        start = end - overlap
                        
            para_buffer = []
            buffer_owner = None
            buffer_level = None
            heading_path = []
        
        for para in doc.paragraphs:
            style = getattr(para.style, 'name', '')
            text = para.text.strip()
            if not text:
                continue
                
            if style.startswith("Heading 1"):
                flush_buffer()
                current_h1 = Chunk(type="heading1", content=text, children=[])
                root.children.append(current_h1)
                current_h2 = current_h3 = current_h4 = None
                buffer_owner = current_h1
                buffer_level = 1
                para_buffer = []
                heading_path = [text]
                
            elif style.startswith("Heading 2"):
                flush_buffer()
                if current_h1 is None:
                    current_h1 = Chunk(type="heading1", content="(no heading 1)", children=[])
                    root.children.append(current_h1)
                current_h2 = Chunk(type="heading2", content=text, children=[])
                current_h1.children.append(current_h2)
                current_h3 = current_h4 = None
                buffer_owner = current_h2
                buffer_level = 2
                para_buffer = []
                heading_path = [current_h1.content, text]
                
            elif style.startswith("Heading 3"):
                flush_buffer()
                if current_h2 is None:
                    current_h2 = Chunk(type="heading2", content="(no heading 2)", children=[])
                    if current_h1:
                        current_h1.children.append(current_h2)
                    else:
                        current_h1 = Chunk(type="heading1", content="(no heading 1)", children=[current_h2])
                        root.children.append(current_h1)
                current_h3 = Chunk(type="heading3", content=text, children=[])
                current_h2.children.append(current_h3)
                current_h4 = None
                buffer_owner = current_h3
                buffer_level = 3
                para_buffer = []
                heading_path = [current_h1.content, current_h2.content, text]
                
            elif style.startswith("Heading 4"):
                flush_buffer()
                if current_h3 is None:
                    current_h3 = Chunk(type="heading3", content="(no heading 3)", children=[])
                    if current_h2:
                        current_h2.children.append(current_h3)
                    elif current_h1:
                        current_h2 = Chunk(type="heading2", content="(no heading 2)", children=[current_h3])
                        current_h1.children.append(current_h2)
                    else:
                        current_h1 = Chunk(type="heading1", content="(no heading 1)", children=[])
                        root.children.append(current_h1)
                        current_h2 = Chunk(type="heading2", content="(no heading 2)", children=[current_h3])
                        current_h1.children.append(current_h2)
                current_h4 = Chunk(type="heading4", content=text, children=[])
                current_h3.children.append(current_h4)
                buffer_owner = current_h4
                buffer_level = 4
                para_buffer = []
                heading_path = [current_h1.content, current_h2.content, current_h3.content, text]
                
            else:
                # Gom paragraph vào heading cấp thấp nhất hiện tại
                if buffer_owner is not None:
                    if para_buffer is None:
                        para_buffer = []
                    para_buffer.append(text)
                else:
                    # Không có heading nào, tạo chunk trực tiếp
                    flush_buffer()
                    root.children.append(Chunk(type="paragraph_chunk", content=text))
                    
        flush_buffer()
        return root

    def flatten_docx_chunk(self, chunk: Chunk) -> Chunk:
        """
        Làm phẳng cây chunk thành danh sách các chunk paragraph.
        Content đã được format sẵn với heading prefix nên không cần thêm gì.
        """
        flat = Chunk(type="docx_flat", content="", children=[])
        
        def traverse(node):
            if node.type == "paragraph_chunk":
                # Content đã có format đầy đủ, chỉ cần thêm vào flat
                flat.children.append(Chunk(type="paragraph_chunk", content=node.content, children=[]))
            for child in getattr(node, "children", []):
                traverse(child)
                
        traverse(chunk)
        return flat

    def chunk_excel(self, df: pd.DataFrame) -> Chunk:
        """
        Chia nhỏ DataFrame excel thành cây chunk các dòng, loại bỏ ký tự noise, format tự nhiên.
        Args:
            df: DataFrame pandas
        Returns:
            Chunk: Cây chunk với mỗi dòng là một node (text sạch, dễ search)
        """
        import re
        root = Chunk(type="excel", content="", children=[])
        for idx, row in df.iterrows():
            # Loại bỏ trường null/empty
            clean_items = [(str(col).strip(), str(val).strip()) for col, val in row.items() if pd.notnull(val) and str(val).strip() != '']
            # Format: col1: val1. col2: val2. ...
            clean_text = ". ".join(f"{col}: {val}" for col, val in clean_items)
            # Loại bỏ ký tự noise dư thừa, xuống dòng, tab, nhiều dấu cách
            clean_text = re.sub(r'[\n\r\t]+', ' ', clean_text)
            clean_text = re.sub(r'\s{2,}', ' ', clean_text).strip()
            # Có thể thêm bước unicode normalize nếu muốn
            if clean_text:
                row_chunk = Chunk(type="row", content=clean_text)
                root.children.append(row_chunk)
        logger.info("Excel chunking completed")
        return root
