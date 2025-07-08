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
        Args:
            doc: Đối tượng docx đã load (ví dụ: từ python-docx)
        Returns:
            Chunk: Cây phân cấp các phần của tài liệu
        """
        root = Chunk(type="docx", content="", children=[])
        current_h1 = None
        current_h2 = None
        current_h3 = None
        current_h4 = None
        for para in doc.paragraphs:
            style = getattr(para.style, 'name', '')
            text = para.text.strip()
            if not text:
                continue
            if style.startswith("Heading 1"):
                current_h1 = Chunk(type="heading1", content=text, children=[])
                root.children.append(current_h1)
                current_h2 = current_h3 = current_h4 = None
            elif style.startswith("Heading 2"):
                if current_h1 is None:
                    current_h1 = Chunk(type="heading1", content="(no heading 1)", children=[])
                    root.children.append(current_h1)
                current_h2 = Chunk(type="heading2", content=text, children=[])
                current_h1.children.append(current_h2)
                current_h3 = current_h4 = None
            elif style.startswith("Heading 3"):
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
            elif style.startswith("Heading 4"):
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
            else:
                paragraph_chunk = Chunk(type="paragraph", content=text)
                if current_h4:
                    current_h4.children.append(paragraph_chunk)
                elif current_h3:
                    current_h3.children.append(paragraph_chunk)
                elif current_h2:
                    current_h2.children.append(paragraph_chunk)
                elif current_h1:
                    current_h1.children.append(paragraph_chunk)
                else:
                    root.children.append(paragraph_chunk)
        logger.info("Docx chunking completed")
        return root

    def flatten_docx_chunk(self, chunk: Chunk) -> Chunk:
        """
        Chuyển cây phân cấp chunk docx thành danh sách các section phẳng, mỗi section gồm đầy đủ context heading.
        Args:
            chunk: Cây chunk phân cấp
        Returns:
            Chunk: Cây chunk phẳng (children là các section)
        """
        flattened = Chunk(type="docx_flat", content="", children=[])
        current = {"heading1": "", "heading2": "", "heading3": "", "heading4": ""}
        def traverse(node: Chunk):
            nonlocal current
            if node.type.startswith("heading"):
                current[node.type] = node.content
                # Reset lower levels
                if node.type == "heading1":
                    current["heading2"] = current["heading3"] = current["heading4"] = ""
                elif node.type == "heading2":
                    current["heading3"] = current["heading4"] = ""
                elif node.type == "heading3":
                    current["heading4"] = ""
            elif node.type == "paragraph":
                heading_path = " ".join([
                    current["heading1"], current["heading2"], current["heading3"], current["heading4"]
                ]).strip()
                full_content = f"{heading_path} {node.content}" if heading_path else node.content
                section = Chunk(type="section", content=full_content)
                flattened.children.append(section)
            # Traverse children if any
            if getattr(node, 'children', None):
                for child in node.children:
                    traverse(child)
        traverse(chunk)
        logger.info("Docx flattening completed")
        return flattened

    def chunk_excel(self, df: pd.DataFrame) -> Chunk:
        """
        Chia nhỏ DataFrame excel thành cây chunk các dòng.
        Args:
            df: DataFrame pandas
        Returns:
            Chunk: Cây chunk với mỗi dòng là một node
        """
        root = Chunk(type="excel", content="", children=[])
        for idx, row in df.iterrows():
            row_chunk = Chunk(type="row", content=str(row.to_dict()))
            root.children.append(row_chunk)
        logger.info("Excel chunking completed")
        return root
