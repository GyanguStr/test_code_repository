import csv
from typing import Dict, List, Optional

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


class CSVLoader(BaseLoader):
    """Loads a CSV file into a list of documents.

    Each document represents one row of the CSV file. Every row is converted into a
    key/value pair and outputted to a new line in the document's page_content.

    The source for each document loaded from csv is set to the value of the
    `file_path` argument for all doucments by default.
    You can override this by setting the `source_column` argument to the
    name of a column in the CSV file.
    The source of each document will then be set to the value of the column
    with the name specified in `source_column`.

    Output Example:
        .. code-block:: txt

            column1: value1
            column2: value2
            column3: value3
    """

    def __init__(
        self,
        csv_object: object,
        source_column: Optional[str] = None,
        csv_args: Optional[Dict] = None,
        encoding: Optional[str] = None,
    ):
        self.csv_object = csv_object
        self.source_column = source_column
        self.encoding = encoding
        self.csv_args = csv_args or {}

    def load(self) -> List[Document]:
        """Load data into document objects."""

        docs = []
        csv_reader = csv.DictReader(self.csv_object, **self.csv_args)  # type: ignore
        for i, row in enumerate(csv_reader):
            content = "\n".join(f"{k.strip()}: {v.strip()}" for k, v in row.items())
            try:
                source = (
                    row[self.source_column]
                    if self.source_column is not None
                    else self.csv_object
                )
            except KeyError:
                raise ValueError(
                    f"Source column '{self.source_column}' not found in CSV file."
                )
            metadata = {"source": source, "row": i}
            doc = Document(page_content=content, metadata=metadata)
            docs.append(doc)

        return docs
