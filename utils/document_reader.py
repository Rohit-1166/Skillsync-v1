from pathlib import Path
import fitz # type: ignore

class DocumentReader:
    
    @staticmethod
    def read(file_path: str | Path) -> str:

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"{file_path} does not exist."
            )

        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            return DocumentReader._read_pdf(file_path)

        elif suffix == ".txt":
            return file_path.read_text(
                encoding="utf-8"
            )

        elif suffix == ".md":
            return file_path.read_text(
                encoding="utf-8"
            )

        else:
            raise ValueError(
                f"Unsupported file type: {suffix}"
            )
    
    @staticmethod
    def _read_pdf(file_path: Path) -> str:

        text = []

        pdf = fitz.open(file_path)

        for page in pdf:
            text.append(page.get_text())

        pdf.close()

        return "\n".join(text)