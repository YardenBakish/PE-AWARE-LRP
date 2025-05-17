import os
import subprocess
import tarfile



if __name__ == "__main__":

  #step 0 - create workenv
  create_train_dir = f"mkdir train"
  create_val_dir = f"mkdir val"

  try:
      subprocess.run(create_train_dir, check=True, shell=True)
      print(f"organized files successfully")
  except subprocess.CalledProcessError as e:
      print(f"Error: {e}")
      exit(1)
  try:
      subprocess.run(create_val_dir, check=True, shell=True)
      print(f"organized files successfully")
  except subprocess.CalledProcessError as e:
      print(f"Error: {e}")
      exit(1)

  tar_file_path = "ILSVRC2012_img_val.tar"

  # Step 1: Change working directory to where the Python file is located
  script_dir = os.path.dirname("val/")
  try:
    os.chdir(script_dir)
  except:
    exit(1)


  # Step 2: Download the file using wget (Linux command)
  url = "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar"
  wget_command = ["wget", url, "--no-check-certificate"]

  # Run the command using subprocess
  try:
      subprocess.run(wget_command, check=True)
      print(f"Downloaded {url} successfully.")
  except subprocess.CalledProcessError as e:
      print(f"Error downloading the file: {e}")
      exit(1)
  
  # Step 3: Extract the .tar file
  tar_file = "ILSVRC2012_img_val.tar"
  if os.path.exists(tar_file):
      try:
          with tarfile.open(tar_file, "r") as tar:
              tar.extractall()
              print("Extraction complete.")
      except tarfile.TarError as e:
          print(f"Error extracting the tar file: {e}")
  else:
      print(f"{tar_file} not found.")

  
  # Step 4: organize to correct folders
  url = "https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh"
  wget_command = f'wget -qO- {url} | bash' 


  try:
      subprocess.run(wget_command, check=True, shell=True)
      print(f"organized files successfully")
  except subprocess.CalledProcessError as e:
      print(f"Error: {e}")
      exit(1)
  
  #step 5 - remove tar file
  if os.path.exists(tar_file_path):
    os.remove(tar_file_path)


    '''
  #testing - see how many empty directories and remove
  for item in os.listdir(script_dir):
    item_path = f'{script_dir}/{item}'
    if os.path.isfile(item_path):
      continue
    if len(os.listdir(item_path)) !=0:
      print(f'{item_path}')
    else:
      os.rmdir(item_path)
 '''
