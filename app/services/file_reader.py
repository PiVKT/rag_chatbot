import logging
from typing import Any, Optional
from docx import Document
import pandas as pd
from io import BytesIO

logger = logging.getLogger(__name__)

class FileReader:
    """
    Đọc các loại file phổ biến: docx, csv, excel (xls, xlsx).
    Các phương thức trả về đối tượng phù hợp cho xử lý tiếp theo.
    """
    def __init__(self):
        pass

    def read_docx(self, file_bytes: bytes) -> Optional[Document]:
        """
        Đọc file docx từ bytes.
        Args:
            file_bytes: Nội dung file dạng bytes
        Returns:
            Document: Đối tượng docx đã load, hoặc None nếu lỗi
        """
        try:
            doc = Document(BytesIO(file_bytes))
            logger.info("Đọc file docx thành công")
            return doc
        except Exception as e:
            logger.error(f"Lỗi khi đọc file docx: {str(e)}")
            return None

    def read_file_by_extension(self, filename: str, file_bytes: bytes) -> Any:
        """
        Đọc file theo đuôi: .csv dùng pandas.read_csv, .xls/.xlsx dùng read_excel
        Args:
            filename: Tên file (có đuôi)
            file_bytes: Nội dung file dạng bytes
        Returns:
            pandas.DataFrame nếu là bảng, hoặc raise Exception nếu lỗi/không hỗ trợ
        """
        ext = filename.lower().split('.')[-1]
        if ext == 'csv':
            try:
                df = pd.read_csv(BytesIO(file_bytes))
                logger.info(f"Đọc file CSV thành công: {filename}")
                return df
            except Exception as e:
                logger.error(f"Lỗi khi đọc file CSV: {str(e)}")
                raise Exception(f"Lỗi khi đọc file CSV: {str(e)}")
        elif ext in ('xls', 'xlsx'):
            try:
                df = pd.read_excel(BytesIO(file_bytes))
                logger.info(f"Đọc file Excel thành công: {filename}")
                return df
            except Exception as e:
                logger.error(f"Lỗi khi đọc file Excel: {str(e)}")
                raise Exception(f"Lỗi khi đọc file Excel: {str(e)}")
        else:
            logger.error(f"Định dạng file '{ext}' không hỗ trợ. Hỗ trợ: .csv, .xls, .xlsx")
            raise Exception(f"Định dạng file '{ext}' không hỗ trợ. Hỗ trợ: .csv, .xls, .xlsx")
