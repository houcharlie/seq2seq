from array import array 
from pathlib import Path 
import oci 
from multiprocessing import Process 
from multiprocessing import Semaphore 
# Number of max processes allowed at a time 
concurrency= 5 
sema = Semaphore(concurrency) 
# The root directory path, Replace with your path 
p = Path('/home/ubuntu/llama_files') 
# The Compartment OCID 
compartment_id = "ocid1.tenancy.oc1..aaaaaaaa4izub2e4gehm5kqyw4cmbpheugppzutvlqu7aqk6t66eq3guajaa" 
# The Bucket name where we will upload 
bucket_name = "houcharlie" 
bucket_folder = "llama/"
""" 
upload_to_object_storage will upload a file to an object storage bucket. This function is intended to be run as a separate process. The client is created with each invocation so that the separate processes do not have a reference to the same client. 
:param path: The path of the object to upload 
:param name: The name of the object to upload 
:param object_storage_client: The object storage sdk client 
:param namespace: The namespace of the bucket 

""" 
def upload_to_object_storage(path:str,name:str,object_storage_client,namespace): 
  with open(path, "rb") as in_file: 
    print("Starting upload {}", format(name)) 
    object_storage_client.put_object(namespace,bucket_name,bucket_folder + name,in_file) 
    print("Finished uploading {}".format(name)) 
    sema.release() 

""" 
createUploadProcess will create a concurrent upload process and put it in proc_list. 
:param object: The path of the object to upload 
:param object_storage_client: The object storage sdk client 
:param namespace: The namespace of the bucket 
:param proc_list:The client list, The client is created with each invocation so that the separate processes do not have a reference to the same client. 

""" 
def createUploadProcess(object:Path,object_storage_client,namespace,proc_list): 
  name = object.relative_to(p).as_posix() 
  sema.acquire() 
  process = Process(target=upload_to_object_storage, args=(object.as_posix(),name,object_storage_client,namespace)) 
  proc_list.append(process) 
  process.start() 

""" 
processDirectoryObjects will check if the current object path is a file or not, if yes it will create a upload process 
:param object: The path of the object to upload 
:param object_storage_client: The object storage sdk client 
:param namespace: The namespace of the bucket 

:param proc_list: The client list, The client is created with each invocation so that the separate processes do not have a reference to the same client. 

""" 
def processDirectoryObjects(object:Path,object_storage_client,namespace,proc_list): 
  if object.is_file(): 
    createUploadProcess(object,object_storage_client,namespace,proc_list) 

""" 
processDirectory will process the current directory 
:param path: the path of the current directory 
:param object_storage_client: The object storage sdk client 
:param namespace: The namespace of the bucket 
:proc_list: The client list, The client is created with each invocation so that the separate processes do not have a reference to the same client. 

""" 
def processDirectory(path:Path,object_storage_client,namespace,proc_list): 
  if path.exists(): 
    print("in directory ---- " + path.relative_to(p).as_posix()) 
    for objects in path.iterdir(): 
      if objects.is_dir(): 
        processDirectory(objects,object_storage_client,namespace,proc_list) 
      else: 
        processDirectoryObjects(objects,object_storage_client,namespace,proc_list) 

if __name__ == '__main__': 
    config = oci.config.from_file() 
    object_storage_client = oci.object_storage.ObjectStorageClient(config) 
    namespace = object_storage_client.get_namespace().data 
    proc_list: array = [] 
    sema = Semaphore(concurrency) 
    if p.exists() and p.is_dir(): 
        processDirectory(p,object_storage_client,namespace,proc_list) 
    for job in proc_list: 
        job.join()