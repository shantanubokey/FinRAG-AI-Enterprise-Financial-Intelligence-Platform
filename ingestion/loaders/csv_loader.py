"""CSV and Excel loader. Converts structured data to both text and table format."""

import io
from typing import Any

from config.logging_config import get_logger
from ingestion.loaders.base_loader import BaseDocumentLoader, RawDocument

logger = get_logger(__name__)


class CSVLoader(BaseDocumentLoader):

    @property
    def supported_extensions(self) -> list[str]:
        return [".csv", ".xlsx", ".xls"]

    async def load(self, content: bytes, filename: str) -> RawDocument:
        import pandas as pd

        ext = "." + filename.rsplit(".", 1)[-1].lower()

        if ext == ".csv":
            df = pd.read_csv(io.BytesIO(content))
            sheets = {"Sheet1": df}
        else:
            xl = pd.ExcelFile(io.BytesIO(content))
            sheets = {name: xl.parse(name) for name in xl.sheet_names}

        tables: list[dict[str, Any]] = []
        text_parts: list[str] = []

        for sheet_name, df in sheets.items():
            df = df.fillna("")
            tables.append({
                "sheet": sheet_name,
                "headers": df.columns.tolist(),
                "rows": df.values.tolist(),
                "data": [df.columns.tolist()] + df.values.tolist(),
            })
            # Convert to readable text for embedding
            text_parts.append(f"=== {sheet_name} ===\n{df.to_string(index=False)}")

        return RawDocument(
            content="\n\n".join(text_parts),
            tables=tables,
            metadata={"filename": filename, "sheets": list(sheets.keys())},
            page_contents=text_parts,
        )
