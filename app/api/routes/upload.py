import logging
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.file_reader import FileReader
from app.services.hierarchical_chunking import HierarchicalTextProcessor
from app.models.schemas import Chunk
from typing import Any

router = APIRouter(prefix="/upload", tags=["upload"])
logger = logging.getLogger(__name__)

file_reader = FileReader()
hierarchical_processor = HierarchicalTextProcessor()

@router.post("/file", response_model=Chunk)
async def upload_file(file: UploadFile = File(...)):
    """
    Upload file (docx, csv, xlsx, xls) và trả về cấu trúc chunk đã xử lý.
    Returns:
        Chunk: Cấu trúc phân cấp nội dung file
    """
    try:
        content = await file.read()
        filename = file.filename.lower()
        if filename.endswith(".docx"):
            doc = file_reader.read_docx(content)
            if doc is None:
                raise HTTPException(status_code=400, detail="Không đọc được file docx")
            chunk = hierarchical_processor.chunk_docx(doc)
            flat_chunk = hierarchical_processor.flatten_docx_chunk(chunk)
            return flat_chunk
        elif filename.endswith((".xlsx", ".xls", ".csv")):
            try:
                df = file_reader.read_file_by_extension(filename, content)
                chunk = hierarchical_processor.chunk_excel(df)
                return chunk
            except Exception as e:
                logger.error(f"Lỗi khi đọc file bảng: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Lỗi khi đọc file bảng: {str(e)}")
        else:
            logger.error(f"Định dạng file không hỗ trợ: {filename}")
            raise HTTPException(status_code=400, detail="Chỉ hỗ trợ file .docx, .csv, .xls, .xlsx")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Lỗi khi upload file: {str(e)}")
        raise HTTPException(status_code=500, detail="Lỗi khi xử lý file upload")