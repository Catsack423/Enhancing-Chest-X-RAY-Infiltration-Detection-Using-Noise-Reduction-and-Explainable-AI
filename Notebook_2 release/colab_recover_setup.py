"""Paste this into Colab to recover setup using the already-uploaded portable ZIP."""
from pathlib import Path
import json
import sys
import zipfile
from google.colab import drive

drive.mount("/content/drive")
package = Path("/content/drive/MyDrive/Notebook_2_colab.zip")
assert package.is_file(), "Upload Notebook_2_colab.zip to My Drive in the mounted Google account"
for name in tuple(sys.modules):
    if name == "shared" or name.startswith("shared."):
        del sys.modules[name]
with zipfile.ZipFile(package) as archive:
    notebook = json.loads(archive.read("Notebook_2/CNN/cnn_comparison.ipynb"))
setup = next(cell for cell in notebook["cells"] if cell["cell_type"] == "code")
exec("".join(setup["source"]), globals())
