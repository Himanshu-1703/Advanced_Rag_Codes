# Additional Dependencies Installation Steps

This project requires **poppler** and **tesseract** to be installed at the OS level for `hi_res` PDF extraction via `unstructured`.

---

## Windows (Chocolatey)

Open PowerShell as Administrator, then run:

```powershell
choco install poppler -y
choco install tesseract -y
```

Restart your terminal after installation so PATH is updated.

> **Note:** If `pytesseract` raises `TesseractNotFoundError`, set the path explicitly in your notebook:
> ```python
> import pytesseract
> pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
> ```

---

## macOS (Homebrew)

```bash
brew install poppler
brew install tesseract
```

---

## Verify Everything Works

After installation, confirm `unstructured` can load a PDF with `hi_res` strategy:

```python
from langchain_unstructured import UnstructuredLoader

loader = UnstructuredLoader(
    "path/to/your.pdf",
    mode="elements",
    strategy="hi_res",
    extract_images_in_pdf=True,
)
elements = loader.load()
print(f"Loaded {len(elements)} elements")
```
