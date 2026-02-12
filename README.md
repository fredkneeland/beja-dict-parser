# Parser for the Beja dictionary to convert it into json
The goal of this project is to convert two different pdf dictionaries of the Beja language into machine readable json.  The original dictionaries can be seen under the folder 'data/input' After the script build_all.py is run the results can be seen in 'data/output'


## Getting started

Install tesseract for ocr in both english and arabic
```bash
# Ubuntu / Debian
sudo apt install tesseract-ocr
sudo apt install tesseract-ocr-ara

# macOS
brew install tesseract
brew install tesseract-lang
```



## Python virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

Requirements
```bash
pip install -r requirements.txt
```

Run all the scripts:

```bash
python build_all.py
```

rm -rf .venv
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt