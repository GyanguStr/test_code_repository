from __future__ import annotations

import logging
from typing import Any, Iterable, List, Optional, Union
from uuid import uuid4

from langchain.docstore.document import Document

from modules.embedding import aget_embeddings
import asyncio

logger = logging.getLogger(__name__)

DEFAULT_MILVUS_CONNECTION = {
    "host": "localhost",
    "port": "19530",
    "user": "",
    "password": "",
    "secure": False,
}


class MassMilvus():
    """Wrapper around the Milvus vector database."""

    def __init__(
        self,
        collection_name: str = "LangChainCollection",
        connection_args: Optional[dict[str, Any]] = None,
        consistency_level: str = "Session",
        index_params: Optional[dict] = None,
        search_params: Optional[dict] = None,
        drop_old: Optional[bool] = False,
    ):
        try:
            from pymilvus import Collection, utility
        except ImportError:
            raise ValueError(
                "Could not import pymilvus python package. "
                "Please install it with `pip install pymilvus`."
            )

        # Default search params when one is not provided.
        self.default_search_params = {
            "IVF_FLAT": {"metric_type": "L2", "params": {"nprobe": 10}},
            "IVF_SQ8": {"metric_type": "L2", "params": {"nprobe": 10}},
            "IVF_PQ": {"metric_type": "L2", "params": {"nprobe": 10}},
            "HNSW": {"metric_type": "L2", "params": {"ef": 10}},
            "RHNSW_FLAT": {"metric_type": "L2", "params": {"ef": 10}},
            "RHNSW_SQ": {"metric_type": "L2", "params": {"ef": 10}},
            "RHNSW_PQ": {"metric_type": "L2", "params": {"ef": 10}},
            "IVF_HNSW": {"metric_type": "L2", "params": {"nprobe": 10, "ef": 10}},
            "ANNOY": {"metric_type": "L2", "params": {"search_k": 10}},
            "AUTOINDEX": {"metric_type": "L2", "params": {}},
        }

        self.collection_name = collection_name
        self.index_params = index_params
        self.search_params = search_params
        self.consistency_level = consistency_level

        # In order for a collection to be compatible, pk needs to be auto'id and int
        self._primary_field = "pk"
        # In order for compatiblility, the text field will need to be called "text"
        self._text_field = "text"
        # In order for compatbility, the vector field needs to be called "vector"
        self._vector_field = "vector"
        self.fields: list[str] = []
        # Create the connection to the server
        if connection_args is None:
            connection_args = DEFAULT_MILVUS_CONNECTION
        self.alias = self._create_connection_alias(connection_args)
        self.col: Optional[Collection] = None

        # Grab the existing colection if it exists
        if utility.has_collection(self.collection_name, using=self.alias):
            self.col = Collection(
                self.collection_name,
                using=self.alias,
            )
        # If need to drop old, drop it
        if drop_old and isinstance(self.col, Collection):
            self.col.drop()
            self.col = None

        # Initialize the vector store
        self._init()

    def _create_connection_alias(self, connection_args: dict) -> str:
        """Create the connection to the Milvus server."""
        from pymilvus import MilvusException, connections

        # Grab the connection arguments that are used for checking existing connection
        host: str = connection_args.get("host", None)
        port: Union[str, int] = connection_args.get("port", None)
        address: str = connection_args.get("address", None)
        uri: str = connection_args.get("uri", None)
        user = connection_args.get("user", None)

        # Order of use is host/port, uri, address
        if host is not None and port is not None:
            given_address = str(host) + ":" + str(port)
        elif uri is not None:
            given_address = uri.split("https://")[1]
        elif address is not None:
            given_address = address
        else:
            given_address = None
            logger.debug("Missing standard address type for reuse atttempt")

        # User defaults to empty string when getting connection info
        if user is not None:
            tmp_user = user
        else:
            tmp_user = ""

        # If a valid address was given, then check if a connection exists
        if given_address is not None:
            for con in connections.list_connections():
                addr = connections.get_connection_addr(con[0])
                if (
                    con[1]
                    and ("address" in addr)
                    and (addr["address"] == given_address)
                    and ("user" in addr)
                    and (addr["user"] == tmp_user)
                ):
                    logger.debug("Using previous connection: %s", con[0])
                    return con[0]

        # Generate a new connection if one doesnt exist
        alias = uuid4().hex
        try:
            connections.connect(alias=alias, **connection_args)
            logger.debug("Created new connection using: %s", alias)
            return alias
        except MilvusException as e:
            logger.error("Failed to create new connection using: %s", alias)
            raise e

    def _init(
        self, embeddings: Optional[list] = None, metadatas: Optional[list[dict]] = None
    ) -> None:
        if embeddings is not None:
            self._create_collection(embeddings, metadatas)
        self._extract_fields()
        self._create_index()
        self._create_search_params()
        self._load()

    def _create_collection(
        self, embeddings: list, metadatas: Optional[list[dict]] = None
    ) -> None:
        from pymilvus import (
            Collection,
            CollectionSchema,
            DataType,
            FieldSchema,
            MilvusException,
        )
        from pymilvus.orm.types import infer_dtype_bydata

        # Determine embedding dim
        dim = len(embeddings[0])
        fields = []
        # Determine metadata schema
        if metadatas:
            # Create FieldSchema for each entry in metadata.
            for key, value in metadatas[0].items():
                # Infer the corresponding datatype of the metadata
                dtype = infer_dtype_bydata(value)
                # Datatype isnt compatible
                if dtype == DataType.UNKNOWN or dtype == DataType.NONE:
                    logger.error(
                        "Failure to create collection, unrecognized dtype for key: %s",
                        key,
                    )
                    raise ValueError(f"Unrecognized datatype for {key}.")
                # Dataype is a string/varchar equivalent
                elif dtype == DataType.VARCHAR:
                    fields.append(FieldSchema(key, DataType.VARCHAR, max_length=65_535))
                else:
                    fields.append(FieldSchema(key, dtype))

        # Create the text field
        fields.append(
            FieldSchema(self._text_field, DataType.VARCHAR, max_length=65_535)
        )
        # Create the primary key field
        fields.append(
            FieldSchema(
                self._primary_field, DataType.INT64, is_primary=True, auto_id=True
            )
        )
        # Create the vector field, supports binary or float vectors
        fields.append(
            FieldSchema(self._vector_field, infer_dtype_bydata(embeddings[0]), dim=dim)
        )

        # Create the schema for the collection
        schema = CollectionSchema(fields)

        # Create the collection
        try:
            self.col = Collection(
                name=self.collection_name,
                schema=schema,
                consistency_level=self.consistency_level,
                using=self.alias,
            )
        except MilvusException as e:
            logger.error(
                "Failed to create collection: %s error: %s", self.collection_name, e
            )
            raise e

    def _extract_fields(self) -> None:
        """Grab the existing fields from the Collection"""
        from pymilvus import Collection

        if isinstance(self.col, Collection):
            schema = self.col.schema
            for x in schema.fields:
                self.fields.append(x.name)
            # Since primary field is auto-id, no need to track it
            self.fields.remove(self._primary_field)

    def _get_index(self) -> Optional[dict[str, Any]]:
        """Return the vector index information if it exists"""
        from pymilvus import Collection

        if isinstance(self.col, Collection):
            for x in self.col.indexes:
                if x.field_name == self._vector_field:
                    return x.to_dict()
        return None

    def _create_index(self) -> None:
        """Create a index on the collection"""
        from pymilvus import Collection, MilvusException

        if isinstance(self.col, Collection) and self._get_index() is None:
            try:
                # If no index params, use a default HNSW based one
                if self.index_params is None:
                    self.index_params = {
                        "metric_type": "L2",
                        "index_type": "HNSW",
                        "params": {"M": 8, "efConstruction": 64},
                    }

                try:
                    self.col.create_index(
                        self._vector_field,
                        index_params=self.index_params,
                        using=self.alias,
                    )

                # If default did not work, most likely on Zilliz Cloud
                except MilvusException:
                    # Use AUTOINDEX based index
                    self.index_params = {
                        "metric_type": "L2",
                        "index_type": "AUTOINDEX",
                        "params": {},
                    }
                    self.col.create_index(
                        self._vector_field,
                        index_params=self.index_params,
                        using=self.alias,
                    )
                logger.debug(
                    "Successfully created an index on collection: %s",
                    self.collection_name,
                )

            except MilvusException as e:
                logger.error(
                    "Failed to create an index on collection: %s", self.collection_name
                )
                raise e

    def _create_search_params(self) -> None:
        """Generate search params based on the current index type"""
        from pymilvus import Collection

        if isinstance(self.col, Collection) and self.search_params is None:
            index = self._get_index()
            if index is not None:
                index_type: str = index["index_param"]["index_type"]
                metric_type: str = index["index_param"]["metric_type"]
                self.search_params = self.default_search_params[index_type]
                self.search_params["metric_type"] = metric_type

    def _load(self) -> None:
        """Load the collection if available."""
        from pymilvus import Collection

        if isinstance(self.col, Collection) and self._get_index() is not None:
            self.col.load()

    def aadd_documents(
        self,
        documents: Iterable[Document],
        metadatas: Optional[List[dict]] = None,
        retry: Optional[int] = 50,
        timeout: Optional[int] = None,
        batch_size: int = 1000,
        **kwargs: Any,
    ) -> List[str]:
        from pymilvus import Collection, MilvusException

        # Seperate texts and metadatas
        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]

        # Get the embeddings using asynchronously embedding function
        embedding_res = asyncio.run(aget_embeddings(texts, retry))
        embeddings = [x['data'][0]['embedding'] for x in embedding_res]

        # If the collection hasnt been initialized yet, perform all steps to do so
        if not isinstance(self.col, Collection):
            self._init(embeddings, metadatas)

        # Dict to hold all insert columns
        insert_dict: dict[str, list] = {
            self._text_field: texts,
            self._vector_field: embeddings,
        }

        # Collect the metadata into the insert dict.
        if metadatas is not None:
            for d in metadatas:
                for key, value in d.items():
                    if key in self.fields:
                        insert_dict.setdefault(key, []).append(value)

        # Total insert count
        vectors: list = insert_dict[self._vector_field]
        total_count = len(vectors)

        pks: list[str] = []

        print("Inserting data into Milvus...", end='')
        assert isinstance(self.col, Collection)
        for i in range(0, total_count, batch_size):
            # Grab end index
            end = min(i + batch_size, total_count)
            # Convert dict to list of lists batch for insertion
            insert_list = [insert_dict[x][i:end] for x in self.fields]
            # Insert into the collection.
            try:
                res: Collection
                res = self.col.insert(insert_list, timeout=timeout, **kwargs)
                pks.extend(res.primary_keys)
            except MilvusException as e:
                logger.error(
                    "Failed to insert batch starting at entity: %s/%s", i, total_count
                )
                raise e
        print('Done.')
        return pks
        

    @classmethod
    def afrom_documents(
        cls,
        documents: List[Document],
        metadatas: Optional[List[dict]] = None,
        collection_name: str = "LangChainCollection",
        connection_args: dict[str, Any] = DEFAULT_MILVUS_CONNECTION,
        consistency_level: str = "Session",
        index_params: Optional[dict] = None,
        search_params: Optional[dict] = None,
        drop_old: bool = False,
        **kwargs: Any,
    ) -> MassMilvus:
        vector_db = cls(
            collection_name=collection_name,
            connection_args=connection_args,
            consistency_level=consistency_level,
            index_params=index_params,
            search_params=search_params,
            drop_old=drop_old,
            **kwargs,
        )
        pks = vector_db.aadd_documents(documents=documents, metadatas=metadatas)
        return pks