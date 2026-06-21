from pathlib import Path
import fitz # type: ignore


# Unified file reading interface that abstracts format-specific
# extraction logic behind a single static method.
# Keeps all I/O concerns isolated from the parsers and API layer,
# which consume plain text strings regardless of source format.
class DocumentReader:

    @staticmethod
    def read(file_path: str | Path) -> str:

        file_path = Path(file_path)

        # Fail early with a clear message rather than letting the
        # format-specific reader raise a cryptic low-level I/O error.
        if not file_path.exists():
            raise FileNotFoundError(
                f"{file_path} does not exist."
            )

        suffix = file_path.suffix.lower()

        # Route to the appropriate reader based on file extension.
        # PDF requires binary extraction via PyMuPDF; text formats
        # are read directly without an intermediate parsing step.
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
            # Raise explicitly rather than returning an empty string
            # so unsupported uploads are surfaced immediately at the API layer.
            raise ValueError(
                f"Unsupported file type: {suffix}"
            )

    @staticmethod
    def _read_pdf(file_path: Path) -> str:

        # Accumulate text page by page rather than concatenating strings
        # in a loop — list append + single join is more memory efficient
        # for large multi-page JD documents.
        text = []

        pdf = fitz.open(file_path)

        for page in pdf:
            text.append(page.get_text())

        # Explicitly close the PDF handle to release the file lock
        # immediately after extraction, especially important when
        # the temp file is deleted right after reading in the API layer.
        pdf.close()

        # Join pages with newlines to preserve section boundaries
        # that the JD parser relies on for regex-based field extraction.
        return "\n".join(text)