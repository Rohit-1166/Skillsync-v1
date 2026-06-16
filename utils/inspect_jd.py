from config.settings import JD_FILE
from utils.document_reader import DocumentReader


def main():

    text = DocumentReader.read(JD_FILE)

    print("=" * 80)

    print(text)

    print("=" * 80)


if __name__ == "__main__":
    main()