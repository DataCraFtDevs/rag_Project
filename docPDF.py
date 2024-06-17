import os
from langchain_community.document_loaders import PDFPlumberLoader
import pandas as pd


# Extract document embeddings
def PdfFiles(_path):
    files = os.listdir(_path)
    df = pd.DataFrame(files, columns=["file_name"])

    df["desktop_path"] = _path + "/" + df["file_name"]
    # For testing just 3 files were iterated. 
    file_paths = df["desktop_path"].values

    all_docs_content = []  # List to store all extracted content

    for j in file_paths:
        loader = PDFPlumberLoader(j)
        docs = loader.load_and_split()

        all_docs_content.append(docs)

    return all_docs_content


# Extract document paths
def FilePaths(_path):
    files = os.listdir(_path)
    df = pd.DataFrame(files, columns=["file_name"])

    df["desktop_path"] = _path + "/" + df["file_name"]
    return df


# _path = "/Users/admin/Documents/pdf_file_blobs"
# extracted_contents = FilePaths(_path)
# print(extracted_contents)
# print(extracted_contents[0])

