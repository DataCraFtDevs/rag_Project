from docPDF import PdfFiles
from docPDF import FilePaths
import pandas as pd
from production_test import Language_ModelFunction
import os

_path = "/Users/admin/Documents/pdf_file_blobs"
extracted_contents = PdfFiles(_path)
filePaths_doc = FilePaths(_path)



user_prompt = input("Enter Prompt: ")


res_arr = []

for n in range(0, len(extracted_contents)):
    response = Language_ModelFunction(docs=extracted_contents[n], model_type="llama3", user_prompt=user_prompt)
    res_arr.append(response)

res_df = pd.DataFrame({"response": res_arr, "prompt":user_prompt})

results_df = pd.concat([res_df, filePaths_doc], axis=1)
print(results_df)

# results_df.to_csv("My_test_results.csv", index=False)
