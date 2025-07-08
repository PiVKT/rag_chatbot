import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Depends
from sqlalchemy.orm import Session
from app.services.file_reader import FileReader
from app.services.hierarchical_chunking import HierarchicalTextProcessor
from app.services.vector_store import PgVectorStore
from app.models.database import get_db
from typing import Any

router = APIRouter(prefix="/upload", tags=["upload"])
logger = logging.getLogger(__name__)

file_reader = FileReader()
hierarchical_processor = HierarchicalTextProcessor()

def upload_file_task(filename: str, content: bytes, db: Session, metadata: dict = None):
    """
    Background task: đọc file, chunking và lưu vào vector store
    """
    try:
        logger.info(f"[UPLOAD] Bắt đầu xử lý file {filename}")
        vector_store = PgVectorStore(db)
        url = filename 
        title = filename
        meta = metadata or {"filename": filename}
        # Xác định loại file
        if filename.endswith(".docx"):
            doc = file_reader.read_docx(content)
            if doc is None:
                logger.error(f"Không đọc được file docx: {filename}")
                return
            chunk_tree = hierarchical_processor.chunk_docx(doc)
            flat_chunk = hierarchical_processor.flatten_docx_chunk(chunk_tree)
            # Lấy nội dung gốc
            raw_content = "\n".join([p.content for p in flat_chunk.children])
            chunk_texts = [c.content for c in flat_chunk.children]
        elif filename.endswith((".xlsx", ".xls", ".csv")):
            df = file_reader.read_file_by_extension(filename, content)
            chunk_tree = hierarchical_processor.chunk_excel(df)
            # Lấy nội dung gốc (toàn bộ bảng)
            raw_content = df.to_csv(index=False) if hasattr(df, 'to_csv') else str(df)
            chunk_texts = [c.content for c in chunk_tree.children]
        else:
            logger.error(f"Định dạng file không hỗ trợ: {filename}")
            return
        # Lưu vào vector store
        vector_store.add_document(
            url=url,
            title=title,
            content=raw_content,
            chunks=chunk_texts,
            metadata=meta
        )
        logger.info(f"[UPLOAD] Đã lưu file {filename} vào vector store với {len(chunk_texts)} chunks")
    except Exception as e:
        logger.error(f"[UPLOAD] Lỗi khi xử lý file {filename}: {str(e)}")

@router.post("/file", response_model=dict)
async def upload_file(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db)
):
    """
    Upload file (docx, csv, xlsx, xls) và lưu vào vector store bằng background task.
    Returns:
        dict: message, status
    """
    try:
        content = await file.read()
        filename = file.filename
        # Có thể bổ sung kiểm tra file đã tồn tại nếu muốn
        background_tasks.add_task(upload_file_task, filename, content, db)
        return {
            "message": f"Đã nhận file {filename}, đang xử lý nền.",
            "status": "processing"
        }
    except Exception as e:
        logger.error(f"[UPLOAD] Lỗi khi nhận file: {str(e)}")
        raise HTTPException(status_code=500, detail="Lỗi khi upload file")