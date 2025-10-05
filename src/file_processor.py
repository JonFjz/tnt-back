#the class that from the id downloads the CSV TO ANALYZE FOR THE STAR ID GET THE LATEST DATA FROM THE API AND SAVE THE FILE IN THE SERVER 
import os
import shutil
import pathlib

class FileProcessor:
    def __init__(self, file):
        
        self.file = file
        self.tmp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tmp', 'Tess')
        os.makedirs(self.tmp_dir, exist_ok=True)
        
        self.save_file()
    
    def save_file(self):
       
        if isinstance(self.file, str):
            # If the input is a file path
            file_path = self.file
            file_name = os.path.basename(file_path)
            destination = os.path.join(self.tmp_dir, file_name)
            self.file_path = destination
            # Check if file already exists in destination
            if not os.path.exists(destination):
                shutil.copy2(file_path, destination)
                print(f"File saved to {destination}")
            else:
                print(f"File already exists at {destination}")
                
        elif hasattr(self.file, 'read'):
           
            try:
                file_name = os.path.basename(self.file.name)
            except AttributeError:
               
                import uuid
                file_name = f"file_{str(uuid.uuid4())[:8]}"
                
            destination = os.path.join(self.tmp_dir, file_name)
            self.file_path = destination
            if not os.path.exists(destination):
               
                with open(destination, 'wb') as f:
                    self.file.seek(0)
                    shutil.copyfileobj(self.file, f)
                print(f"File saved to {destination}")
            else:
                print(f"File already exists at {destination}")
        else:
            raise ValueError("Input must be a file path string or a file-like object")
