"""Excel export utilities."""

from __future__ import annotations

from io import BytesIO

import pandas as pd


def build_excel_report(sheets: dict[str, pd.DataFrame]) -> bytes:
    """Create an XLSX workbook in memory from named pandas DataFrames."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            safe_name = sheet_name[:31]
            df.to_excel(writer, sheet_name=safe_name, index=False)
    output.seek(0)
    return output.read()
