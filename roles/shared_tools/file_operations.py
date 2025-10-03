"""
File Operations Shared Tools

Common file handling functionality that can be used across multiple roles.
"""

from strands import tool
from typing import Dict, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@tool
def read_file_content(file_path: str, encoding: str = "utf-8") -> Dict:
    """
    Read content from a file.
    
    Args:
        file_path: Path to the file to read
        encoding: File encoding (default: utf-8)
        
    Returns:
        Dict containing file content and metadata
    """
    try:
        path = Path(file_path)
        
        if not path.exists():
            return {
                "success": False,
                "error": f"File not found: {file_path}",
                "content": None
            }
        
        if not path.is_file():
            return {
                "success": False,
                "error": f"Path is not a file: {file_path}",
                "content": None
            }
        
        with open(path, 'r', encoding=encoding) as f:
            content = f.read()
        
        return {
            "success": True,
            "content": content,
            "file_path": str(path),
            "size_bytes": len(content.encode(encoding)),
            "line_count": len(content.splitlines()),
            "encoding": encoding
        }
        
    except Exception as e:
        logger.error(f"Failed to read file {file_path}: {e}")
        return {
            "success": False,
            "error": str(e),
            "content": None
        }


@tool
def write_file_content(file_path: str, content: str, encoding: str = "utf-8", 
                      create_dirs: bool = True) -> Dict:
    """
    Write content to a file.
    
    Args:
        file_path: Path to the file to write
        content: Content to write to the file
        encoding: File encoding (default: utf-8)
        create_dirs: Create parent directories if they don't exist
        
    Returns:
        Dict containing operation result and metadata
    """
    try:
        path = Path(file_path)
        
        # Create parent directories if requested
        if create_dirs and not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding=encoding) as f:
            f.write(content)
        
        return {
            "success": True,
            "file_path": str(path),
            "bytes_written": len(content.encode(encoding)),
            "line_count": len(content.splitlines()),
            "encoding": encoding
        }
        
    except Exception as e:
        logger.error(f"Failed to write file {file_path}: {e}")
        return {
            "success": False,
            "error": str(e),
            "file_path": file_path
        }


@tool
def list_directory_contents(directory_path: str, include_hidden: bool = False,
                          file_types: Optional[list] = None) -> Dict:
    """
    List contents of a directory.
    
    Args:
        directory_path: Path to the directory to list
        include_hidden: Include hidden files/directories
        file_types: List of file extensions to filter by (e.g., ['.py', '.yaml'])
        
    Returns:
        Dict containing directory contents and metadata
    """
    try:
        path = Path(directory_path)
        
        if not path.exists():
            return {
                "success": False,
                "error": f"Directory not found: {directory_path}",
                "contents": []
            }
        
        if not path.is_dir():
            return {
                "success": False,
                "error": f"Path is not a directory: {directory_path}",
                "contents": []
            }
        
        contents = []
        for item in path.iterdir():
            # Skip hidden files if not requested
            if not include_hidden and item.name.startswith('.'):
                continue
            
            # Filter by file types if specified
            if file_types and item.is_file():
                if not any(item.name.endswith(ext) for ext in file_types):
                    continue
            
            contents.append({
                "name": item.name,
                "path": str(item),
                "type": "directory" if item.is_dir() else "file",
                "size": item.stat().st_size if item.is_file() else None
            })
        
        return {
            "success": True,
            "directory_path": str(path),
            "contents": sorted(contents, key=lambda x: (x['type'], x['name'])),
            "total_items": len(contents)
        }
        
    except Exception as e:
        logger.error(f"Failed to list directory {directory_path}: {e}")
        return {
            "success": False,
            "error": str(e),
            "contents": []
        }