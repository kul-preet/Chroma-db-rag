
import os
from src.fileReaders import read_txt
from src.fileReaders import read_pdf
from src.fileReaders import read_csv
from src.fileReaders import read_excel
from src.fileReaders import read_docx


#---------------------READ ROUTER --------------------------
def read_file(filepath):
    '''route each file to the correct reader'''
    extension = os.path.splitext(filepath)[1].lower()
    
    #map extenstion to render
    readers = {
        ".txt":read_txt,
        ".pdf":read_pdf,
        ".csv":read_csv,
        ".xlsx":read_excel,
        ".docx":read_docx
    }
    
    if extension in readers:
        return readers[extension](filepath)
    else:
        print(f" skipping unsupported file: {filepath}")
        return[]
