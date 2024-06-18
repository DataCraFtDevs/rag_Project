import streamlit as st
import pandas as pd
import os
from docPDF import PdfFiles, FilePaths
from production_test import Language_ModelFunction


# Function to process PDF files and generate results
def process_pdfs(folder_path, user_prompt, model_type):
    # Process PDF files
    extracted_contents = PdfFiles(folder_path)
    filePaths_doc = FilePaths(folder_path)

    # Initialize an empty list to store responses
    res_arr = []

    # Iterate over extracted contents and apply language model
    for n in range(len(extracted_contents)):
        response = Language_ModelFunction(docs=extracted_contents[n], model_type=model_type, user_prompt=user_prompt)
        res_arr.append(response)

    # Create DataFrame with results
    res_df = pd.DataFrame({"response": res_arr, "prompt": user_prompt})

    # Combine results with file pathsapp_stre
    results_df = pd.concat([res_df, filePaths_doc], axis=1)

    return results_df

# Main Streamlit app code
def main():
    col1, col2, col3 = st.columns([1, 2, 5])

    with col1:
        st.image("VIRIDIEN_Logo.png", width=700)

    with col2:
        st.write("")  # Empty space to align with the title

    with col3:
        st.markdown("")

    st.title("PDF Processing tool with Language Model")



    # Sidebar input for folder path and user prompt
    folder_path = st.text_input("Enter the folder path containing PDF files:")
    user_prompt = st.text_input("Enter prompt for language model:")
    # save_folder = st.text_input("Enter name of document with the extension(csv/xlsx) to be saved as:")

    # Dropdown for selecting model type
    model_type = st.selectbox("Select Model Type", ["llama3", "mistral"])

    file_name = st.text_input("Enter file name for saving results (without extension):", "results")

    # Process button
    if st.button("Process PDFs"):
        if folder_path:
            if os.path.isdir(folder_path):
                # Process PDFs and get results based on selected model type
                results_df = process_pdfs(folder_path, user_prompt, model_type)

                # Create a dedicated folder for saving results
                # save_folder = st.text_input("Enter name of document to be saved as:")
                save_folder = "results"
                os.makedirs(save_folder, exist_ok=True)

                # Save results as CSV in the dedicated folder
                # save_path = os.path.join(save_folder, save_folder)
                save_path = os.path.join(save_folder, f"{file_name}.csv")
                # save_path = os.path.join(save_folder, "results.csv")
                results_df.to_csv(save_path, index=False)

                # Display results
                st.success(f"Results saved successfully at {save_path}")
                st.dataframe(results_df)
            else:
                st.error(f"'{folder_path}' is not a valid directory.")
        else:
            st.warning("Please enter a folder path.")

# Run the Streamlit app
if __name__ == "__main__":
    main()
