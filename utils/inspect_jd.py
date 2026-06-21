from config.settings import JD_FILE
from utils.document_reader import DocumentReader


# Standalone diagnostic script for verifying that the configured
# JD file is readable and its extracted text looks correct before
# feeding it into the parsing and ranking pipeline.
# Run directly during development to catch encoding or layout issues
# in the source document without invoking the full API stack.
def main():

    # Read the default JD file using the same DocumentReader
    # that the API and embedding pipeline use at runtime —
    # ensures the diagnostic reflects real extraction behavior.
    text = DocumentReader.read(JD_FILE)

    # Visual delimiters make the raw extracted text easy to inspect
    # in the terminal and clearly bound the document content.
    print("=" * 80)

    print(text)

    print("=" * 80)


if __name__ == "__main__":
    main()