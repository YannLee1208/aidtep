from pathlib import Path as pth


def check_file_exist(file_path: str) -> bool:
    file_path = pth(file_path)
    if not file_path.exists():
        return False
    return True