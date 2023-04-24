def get_gzipped_model_size(file_path):
  # Returns size of gzipped model, in bytes.
  import os
  import zipfile
  import tempfile

  _, zipped_file = tempfile.mkstemp('.zip')
  with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
    f.write(file_path)

  return os.path.getsize(zipped_file)