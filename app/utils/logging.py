import logging
import sys
from pathlib import Path

def setup_logging():
    """Setup logging configuration"""
    
    # Tạo logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Logging format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Root logger
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(log_dir / "app.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Specific loggers
    loggers = [
        "app.services.web_scraper",
        "app.services.text_processor", 
        "app.services.embeddings",
        "app.services.vector_store",
        "app.services.chatbot"
    ]
    
    for logger_name in loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)