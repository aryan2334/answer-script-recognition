# Importing the os module for operating system related functionality
import os 

# import io
# import pickle
# Importing the Document AI library from the Google Cloud documentai_v1 package

from google.cloud import documentai_v1 as documentai
# Importing the List and Sequence types from the typing module for type hints
from typing import List, Sequence

import pandas as pd # Importing the pandas library for data manipulation and analysis
import numpy as np # Importing the numpy library for numerical operations
import yaml # Importing the yaml library for YAML parsing
import re # Importing the re module for regular expressions
from csv import writer # Importing the writer class from the csv module for writing CSV files

# Setting path to google service account credentials 


os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'google_vision_APIs.json'   #Akaash
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'API\answer-script-recognition\google_vision_APIs.json'   #Anirudh
# Setting path to processor and file credentials 


config_file = "Form_parser/form_parser_config.yaml"  #Akaash
# config_file = r"API\answer-script-recognition\Form_parser\form_parser_config.yaml"  #Anirudh

# Loading form parser configuration file
def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Fuction to establish a connection with the form parser processor from the google cloud service 
def online_process(
    project_id: str,
    location: str,
    processor_id: str,
    file_path: str,
    mime_type: str,
) -> documentai.Document:
    """
    Processes a document using the Document AI Online Processing API.
    """

    opts = {"api_endpoint": f"{location}-documentai.googleapis.com"}

    # Instantiates a client
    documentai_client = documentai.DocumentProcessorServiceClient(client_options=opts)

    # Details of the processor created 
    resource_name = documentai_client.processor_path(project_id, location, processor_id)

    # Read the file into memory
    with open(file_path, "rb") as image:
        image_content = image.read()

        # Load Binary Data into Document AI RawDocument Object
        raw_document = documentai.RawDocument(
            content=image_content, mime_type=mime_type
        )

        # Configure the process request
        request = documentai.ProcessRequest(
            name=resource_name, raw_document=raw_document
        )

        # Use the Document AI client to process the sample form
        # <class 'google.cloud.documentai_v1.types.document_processor_service.ProcessResponse'>

        result = documentai_client.process_document(request=request)

        
        # Write the document object to a file using pickle
        # output_file_path = "output_document.pkl"
        # with io.open(output_file_path, "wb") as output_file:
        #     pickle.dump(result.document, output_file)

        # print(f"Document dumped to file: {output_file_path}")

        # result is the API response
        # result.document stores the document information
        
        # print(type(result))
        # print(type(result.document))

        # <class 'google.cloud.documentai_v1.types.document.Document'>
        
        return result.document
    
#  Function to retrive the data from particular table in a image
def get_table_data(
    rows: Sequence[documentai.Document.Page.Table.TableRow], text: str
) -> List[List[str]]:
    """
    Get Text data from table rows
    """
    all_values: List[List[str]] = []
    for row in rows:
        current_row_values: List[str] = []
        for cell in row.cells:
            current_row_values.append(
                text_anchor_to_text(cell.layout.text_anchor, text)
            )
        all_values.append(current_row_values)

        # Return a list of sublists which contains the data of the table
    return all_values
    
# Function to retrieve all the key values pairs present in the image
def retrieve_key_value_pairs(document):
    keys = []
    values = []
    # name_confidence = []
    # value_confidence = []

    enrolment_no = None  # Variable to store enrolment number if found
    course_code = None # Variable to store course code if found
    total_marks_scored = None # Variable to store total marks if found

    for page in document.pages:
        for field in page.form_fields:
            # Get the extracted field names
            field_key = trim_text(field.field_name.text_anchor.content)
            field_value = trim_text(field.field_value.text_anchor.content)
            
            # Check if the field name starts with "Enrol"
            if field_key.startswith("Enrol"):
                # If yes, extract the value and store it in enrolment_no
                enrolment_no = field_value

            # Check if the field name starts with "Course Code"
            if field_key.startswith("Course Code"):
                # If yes, extract the value and store it in course_code
                course_code = field_value

            # Check if the field name starts with "Total Marks"
            if field_key.startswith("Total Marks"):
                # If yes, extract the value and store it in total_marks_scored
                match = re.search(r'\d+(\.\d+)?', field_value)
                if match:
                    # If a number is found, store it in total_marks_scored
                    total_marks_scored = float(match.group())
                
            keys.append(field_key)
            values.append(field_value)

            # We can also get the confidence: how sure the model is that the text is correct
            # name_confidence.append(field.field_name.confidence)
            # value_confidence.append(field.field_value.confidence)


    # Create a Pandas Dataframe to print the values in tabular format.
    df_KeyValue = pd.DataFrame(
        {
            "Field Name": keys,
            "Field Value": values
        }
    )

    return df_KeyValue, enrolment_no, course_code, total_marks_scored

# Function used to trim the text found in the key value pairs 
def trim_text(text: str):
    """
    Remove extra space characters from text (blank, newline, tab, etc.)
    """
    return text.strip().replace("\n", " ")

# Function used to retrieve the table data, return a data frame 
def retrieve_table_data(document):
    header_row_values: List[List[str]] = []
    body_row_values: List[List[str]] = []

    for page in document.pages:
        for index, table in enumerate(page.tables):
            header_row_values = get_table_data(table.header_rows, document.text)
            body_row_values = get_table_data(table.body_rows, document.text)

            # Create a Pandas Dataframe to print the values in tabular format.
            df_table = pd.DataFrame(
                data=body_row_values,
                columns=pd.MultiIndex.from_arrays(header_row_values),
            )

    return df_table

# Function used to process the data found in the table
def text_anchor_to_text(text_anchor: documentai.Document.TextAnchor, text: str) -> str:
    """
    Document AI identifies table data by their offsets in the entirety of the
    document's text. This function converts offsets to a string.
    """
    response = ""
    # If a text segment spans several lines, it will
    # be stored in different text segments.
    for segment in text_anchor.text_segments:
        start_index = int(segment.start_index)
        end_index = int(segment.end_index)
        response += text[start_index:end_index]
    return response.strip().replace("\n", " ")

# Returns a list of marks obtained from the table dataframe 
def retrieve_marks_obtained(df_table):
    # Get the "Marks Obtained" column from the dataframe
    marks_obtained_column = df_table["Marks Obtained"]

    # Convert the column to a list of values
    marks_obtained_values = marks_obtained_column.values.tolist()

    # Initialize an empty list to store the concatenated values
    marks_obtained = []

    # Loop through each index of the sublists
    for i in range(len(marks_obtained_values[0])):
        # Loop through each sublist
        for sublist in marks_obtained_values:
            # Append the value at index i of the current sublist
            marks_obtained.append(sublist[i])


    # Initialize an empty list to store the processed values
    processed_marks_obtained = []

    # Function used to change marks recognised as i to 1 and s to 5
    def letter_to_number(mark):
        if mark == 'I':
            return 1
        elif mark == 'S' or mark == 's':
            return 5
        else:
            return mark

    # Iterate over each value in all_values
    for mark in marks_obtained:
        # Check if the value is a string
        if isinstance(mark, str):
            # Convert 'I' to 1 and 'S' to 5
            mark = letter_to_number(mark)
            match = re.search(r'\d+(\.\d+)?', str(mark))
            if match:
                processed_marks_obtained.append(float(match.group()))
            else:
                # If no number found, set it to 0
                processed_marks_obtained.append(0)
        # Check if the value is a number
        elif isinstance(mark, (int, float)):
            processed_marks_obtained.append(mark)
        # Handle None values
        elif pd.isnull(mark):
            processed_marks_obtained.append(0)


    processed_marks_obtained = [0 if val > 100 else val for val in processed_marks_obtained]
    return processed_marks_obtained

# Fucntion to update the excel sheet
def update_csv(row_details):

    # Create a file object for this file
    with open('Form_parser/Marks.csv', 'a') as f_object:  #Akaash
    # with open(r'API\answer-script-recognition\Form_parser\Marks.csv', 'a') as f_object:  #Anirudh

        # Pass this file object to csv.writer()
        # and get a writer object
        writer_object = writer(f_object)
 
        # Pass the list as an argument into
        # the writerow()
        writer_object.writerow(row_details)
 
        # Close the file object
        f_object.close()

    print('csv file updated successfully')

def main():
    
    config_file = "Form_parser/form_parser_config.yaml" #Akaash
    # config_file = r"API\answer-script-recognition\Form_parser\form_parser_config.yaml" #Anirudh
    config = load_config(config_file)

    PROJECT_ID = config.get("PROJECT_ID")
    LOCATION = config.get("LOCATION")
    PROCESSOR_ID = config.get("PROCESSOR_ID")
    FILE_PATH = config.get("FILE_PATH")
    MIME_TYPE = config.get("MIME_TYPE")

    document = online_process(
        project_id=PROJECT_ID,
        location=LOCATION,
        processor_id=PROCESSOR_ID,
        file_path=FILE_PATH,
        mime_type=MIME_TYPE,
    )

    # Retrieving Key-value pairs from the document
    df_KeyValue, enrolment_no, course_code, total_marks_scored = retrieve_key_value_pairs(document)

    # Retrieving Table data
    df_table = retrieve_table_data(document)

    # print(df_table)
    # print(df_KeyValue)

    # Retrieving Marks obtained from the table
    marks_obtained = retrieve_marks_obtained(df_table)

    new_row_input = [enrolment_no, course_code, total_marks_scored] + marks_obtained


    print("Enrolment Number: ", enrolment_no)
    print("Course Code: ", course_code)
    print("Total marks Scored: ", total_marks_scored)
    # print(marks_obtained)

    update_csv(new_row_input)
    

if __name__ == "__main__":
    main()

