from src.components.retriever import BaseRetriever
from typing import List, Any, Union, Dict, Optional
from src.components.vector_retriever import VectorRetriever
from src.components.keyword_retriever import KeywordRetriever
from src.components.filter_extractor import FilterExtractor
import math
import pymongo
from pymongo import MongoClient
from config import MONGODB_CONFIG


metadata_label_map = {
    "ma_so": "Mã số",
    "chu_de": "Chủ đề",
    "loai_van_ban": "Loại văn bản",
    "co_quan_ban_hanh": "Cơ quan ban hành",
    "noi_ban_hanh": "Nơi ban hành",
    "ngay_ban_hanh": "Ngày ban hành",
    "tinh_trang": "Tình trạng",
    "chuong": "Chương",
    "ten_chuong": "Tên chương",
    "muc": "Mục",
    "ten_muc": "Tên mục",
    "ten_dieu": "Tên điều"
}

class HybridRetriever(BaseRetriever):
    def __init__(
        self,
        embedding_model,
        reranker,
        filter_extractor: FilterExtractor = None,
        pickle_path="./data/parent_documents_mongo.pkl",
        persist_directory="./vector_stores",
        collection_name="law_docs",
        corpus_path_512="./data/bm25_corpus_512.pkl",
        corpus_path_1024="./data/bm25_corpus_1024.pkl",
        chunk_size=1024,  
        chunk_overlap=100,
        vector_num_chunks=10,
        keyword_num_chunks=10,
        hybrid_num_chunks=5,
        num_final_docs = 5,
        vector_search_weight=0.6,
        auto_extract_filters=True,
    ):
        if chunk_size not in [512, 1024]:
            raise ValueError("chunk_size must be either 512 or 1024")
        
        self.chunk_size = chunk_size
        self.hybrid_num_chunks = hybrid_num_chunks
        self.num_final_docs = num_final_docs
        self.vector_search_weight = vector_search_weight
        self.reranker = reranker
        self.filter_extractor = filter_extractor
        self.auto_extract_filters = auto_extract_filters

        self.mongodb_config = MONGODB_CONFIG
        self.mongo_client = None
        self.mongo_collection = None
        self._connect_mongodb()

        self.vector_retriever = VectorRetriever(
            embedding_model=embedding_model,
            pickle_path=pickle_path,
            persist_directory=persist_directory,
            collection_name=collection_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            num_chunks=vector_num_chunks
        )
        
        self.keyword_retriever = KeywordRetriever(
            corpus_path_512=corpus_path_512,
            corpus_path_1024=corpus_path_1024,
            num_chunks=keyword_num_chunks
        )

    def _connect_mongodb(self):
        try:
            uri = self.mongodb_config.get("mongodb_uri")
            db_name = self.mongodb_config.get("mongodb_db_name")
            collection_name_512 = self.mongodb_config.get("mongodb_collection_name_512")
            collection_name_1024 = self.mongodb_config.get("mongodb_collection_name_1024")

            self.mongo_client = MongoClient(uri, serverSelectionTimeoutMS=3000)

            # Test connection
            self.mongo_client.admin.command('ping')
            print("MongoDB connection successful.")

            mongo_db = self.mongo_client[db_name]

            if self.chunk_size == 512:
                self.mongo_collection = mongo_db[collection_name_512]
                print(f"Using collection: {db_name}.{collection_name_512}")
            elif self.chunk_size == 1024:
                self.mongo_collection = mongo_db[collection_name_1024]
                print(f"Using collection: {db_name}.{collection_name_1024}")
            else:
                raise ValueError("Invalid chunk_size. Must be 512 or 1024.")

        except pymongo.errors.ServerSelectionTimeoutError as e:
            print(f"MongoDB ping failed: Cannot connect to server. Details: {e}")
            self.mongo_client = None
            self.mongo_collection = None
        except Exception as e:
            print(f"Failed to connect to MongoDB: {e}")
            self.mongo_client = None
            self.mongo_collection = None

    def _get_filtered_vector_ids(self, filters: Dict) -> List[str]:
        """
        Pre-filter documents using MongoDB and return vector IDs
        
        Args:
            filters: Dictionary containing MongoDB query filters
        
        Returns:
            List of vector IDs that match the filter criteria
        """
        try:
            # Query MongoDB with filters
            cursor = self.mongo_collection.find(filters, {"vector_id": 1})
            
            # Collect all vector IDs from matching documents
            vector_ids = []
            for doc in cursor:
                doc_vector_ids = doc.get("vector_id", [])
                if isinstance(doc_vector_ids, list):
                    vector_ids.extend(doc_vector_ids)
                else:
                    vector_ids.append(doc_vector_ids)
            print(f"MongoDB pre-filtering: Found {len(vector_ids)} vector chunks matching filters")
            return vector_ids
            
        except Exception as e:
            print(f"Error in MongoDB pre-filtering: {e}")
            return []

    def _create_chroma_where_clause(self, vector_ids: List[str]) -> Dict:
        """
        Create ChromaDB where clause to filter by vector IDs
        
        Args:
            vector_ids: List of vector IDs to filter by
            
        Returns:
            Dictionary containing ChromaDB where clause
        """
        if not vector_ids:
            return {}
        
        # ChromaDB where clause to filter by vector_id
        where_clause = {
            "vector_id": {
                "$in": vector_ids
            }
        }
        return where_clause

    def _weighted_merge_and_select_docs(
        self, 
        vector_docs_with_scores: List[tuple], 
        keyword_docs_with_scores: List[tuple]
    ) -> List[Any]:
        """
        Merge and select documents from vector and keyword retrievers
        using weighted approach
        """

        vector_count = math.ceil(self.hybrid_num_chunks * self.vector_search_weight)
        keyword_count = self.hybrid_num_chunks - vector_count

        vector_count = min(vector_count, len(vector_docs_with_scores))
        keyword_count = min(keyword_count, len(keyword_docs_with_scores))

        selected_vector_docs = [doc for doc, score in vector_docs_with_scores[:vector_count]]
        selected_keyword_docs = [doc for doc, score in keyword_docs_with_scores[:keyword_count]]

        unique_docs = {}

        # Add vector documents
        for doc in selected_vector_docs:
            parent_id = doc.metadata.get("parent_id")
            if parent_id and parent_id not in unique_docs:
                unique_docs[parent_id] = doc
 
        # Add keyword documents
        for doc in selected_keyword_docs:
            parent_id = doc.metadata.get("parent_id")
            if parent_id and parent_id not in unique_docs:
                unique_docs[parent_id] = doc
        
        final_docs = list(unique_docs.values())

        # Fill remaining slots if needed
        if len(final_docs) < self.hybrid_num_chunks:
            remaining_needed = self.hybrid_num_chunks - len(final_docs)

            all_remaining_docs = []
            # Add remaining vector docs
            for doc, score in vector_docs_with_scores[vector_count:]:
                parent_id = doc.metadata.get("parent_id")
                if parent_id not in unique_docs:
                    all_remaining_docs.append(doc)
                    
            # Add remaining keyword docs
            for doc, score in keyword_docs_with_scores[keyword_count:]:
                parent_id = doc.metadata.get("parent_id")
                if parent_id not in unique_docs:
                    all_remaining_docs.append(doc)
            
            final_docs.extend(all_remaining_docs[:remaining_needed])
        
        return final_docs[:self.hybrid_num_chunks]

    def _format_context(self, parent_docs: List[Any]) -> str:
        """Format parent documents into context string"""
        def format_metadata_vietnamese(metadata):
            parts = []
            for key, value in metadata.items():
                if key in ("sourcedoc", "parent_id", "vector_id", "mongo_id"):
                    continue
                label = metadata_label_map.get(key, key)
                parts.append(f"{label}: {value}")
            return "(Metadata: " + "; ".join(parts) + ")"
        
        contexts = []
        for i, doc in enumerate(parent_docs, 1):
            content = doc.page_content
            metadata_formatted = format_metadata_vietnamese(doc.metadata)
            contexts.append(f"Tài liệu {i}:\n{content}\n{metadata_formatted}")
        
        return "\n\n".join(contexts)


    def retrieve(self, query: Union[str, List[str]], filters: Optional[Dict] = None) -> str:
            """
            Enhanced retrieve method with per-query filter extraction and search
            
            Args:
                query: Search query (string or list of strings)
                filters: Optional MongoDB filters to pre-filter documents
                        If None and auto_extract_filters=True, will extract from each query
            
            Returns:
                Formatted context string from retrieved documents
            """
            if isinstance(query, str):
                queries = [query]
            elif isinstance(query, list):
                queries = query
            else:
                raise ValueError("Query must be a string or list of strings")
            
            all_vector_docs = []
            all_keyword_docs = []
            
            # Process each query individually
            for i, single_query in enumerate(queries):
                print(f"Processing query {i+1}/{len(queries)}: {single_query}")
                
                # Step 1: Extract filters for this specific query
                query_filters = {}
                if filters is None and self.auto_extract_filters and self.filter_extractor:
                    try:
                        query_filters = self.filter_extractor.get_query_filter(single_query)
                        if query_filters:
                            print(f"Auto-extracted filters for query '{single_query}': {query_filters}")
                    except Exception as e:
                        print(f"Error in automatic filter extraction for query '{single_query}': {e}")
                        query_filters = {}
                
                # Use provided filters or extracted filters for this query
                final_filters = filters if filters is not None else query_filters
                
                # Step 2: MongoDB Pre-filtering for this query (if filters available)
                chroma_where_clause = {}
                if final_filters and self.mongo_collection is not None:
                    print(f"Step 2: MongoDB pre-filtering for query '{single_query}'...")
                    filtered_vector_ids = self._get_filtered_vector_ids(final_filters)
                    
                    if not filtered_vector_ids:
                        print(f"No documents match the filters for query '{single_query}'")
                        continue  # Skip this query if no documents match
                    
                    # Create ChromaDB where clause for filtered vector IDs
                    chroma_where_clause = self._create_chroma_where_clause(filtered_vector_ids)
                    print(f"Will search within {len(filtered_vector_ids)} pre-filtered chunks for query '{single_query}'")
                
                # Step 3: Vector search for this query on filtered subset (or all if no filters)
                query_vector_docs = self.vector_retriever.get_unique_parent_docs_with_scores(
                    [single_query], where_clause=chroma_where_clause if chroma_where_clause else None
                )
                
                # Step 4: Keyword search for this query with filtering if available
                use_512 = (self.chunk_size == 512)
                query_keyword_docs = self.keyword_retriever.get_unique_parent_docs_with_scores(
                    [single_query], use_512=use_512
                )
                
                # Filter keyword results if filters are available for this query
                if final_filters:
                    query_keyword_docs = self._filter_keyword_results(query_keyword_docs, final_filters)
                
                # Add query-specific results to overall collections
                all_vector_docs.extend(query_vector_docs)
                all_keyword_docs.extend(query_keyword_docs)
                
                print(f"Query '{single_query}' returned {len(query_vector_docs)} vector docs and {len(query_keyword_docs)} keyword docs")
            
            # Step 5: Remove duplicates and merge documents using weighted approach
            vector_docs_dict = {}
            for doc, score in all_vector_docs:
                parent_id = doc.metadata.get("parent_id")
                if parent_id:
                    if parent_id not in vector_docs_dict or score > vector_docs_dict[parent_id][1]:
                        vector_docs_dict[parent_id] = (doc, score)
                else:
                    # Fallback to document content hash if no parent_id
                    doc_hash = hash(doc.page_content)
                    if doc_hash not in vector_docs_dict or score > vector_docs_dict[doc_hash][1]:
                        vector_docs_dict[doc_hash] = (doc, score)
            
            keyword_docs_dict = {}
            for doc, score in all_keyword_docs:
                parent_id = doc.metadata.get("parent_id")
                if parent_id:
                    if parent_id not in keyword_docs_dict or score > keyword_docs_dict[parent_id][1]:
                        keyword_docs_dict[parent_id] = (doc, score)
                else:
                    # Fallback to document content hash if no parent_id
                    doc_hash = hash(doc.page_content)
                    if doc_hash not in keyword_docs_dict or score > keyword_docs_dict[doc_hash][1]:
                        keyword_docs_dict[doc_hash] = (doc, score)
            
            # Convert back to lists
            deduped_vector_docs = list(vector_docs_dict.values())
            deduped_keyword_docs = list(keyword_docs_dict.values())
            
            print(f"After deduplication: {len(deduped_vector_docs)} vector docs, {len(deduped_keyword_docs)} keyword docs")
            
            # Step 6: Merge documents using weighted approach
            merged_docs = self._weighted_merge_and_select_docs(deduped_vector_docs, deduped_keyword_docs)
            
            # Step 7: Rerank documents using all queries
            reranked_docs_with_scores = self.reranker.rerank_documents(queries, merged_docs)
            
            # Step 8: Select final documents and format context
            reranked_docs = [doc for doc, *_ in reranked_docs_with_scores]
            final_docs = reranked_docs[:self.num_final_docs]
            
            if not final_docs:
                return "Không tìm thấy tài liệu nào phù hợp với các truy vấn đã cung cấp."
            
            return self._format_context(final_docs)


    def _filter_keyword_results(self, keyword_docs_with_scores: List[tuple], filters: Dict) -> List[tuple]:
        """
        Filter keyword search results based on document metadata
        Check if document metadata contains all the required filter values
        Supports both simple filters and $or operations
        
        Args:
            keyword_docs_with_scores: List of (document, score) tuples from keyword search
            filters: MongoDB filters to apply (will be converted to metadata field names)
                    Can be simple dict or contain "$or" with list of filter conditions
            
        Returns:
            Filtered list of (document, score) tuples
        """
        if not filters:
            return keyword_docs_with_scores
        
        # Mapping from MongoDB field names to metadata field names
        mongo_to_metadata_field_map = {
            "document_code": "ma_so",
            "document_title": "chu_de", 
            "document_type": "loai_van_ban",
            "issuing_authority": "co_quan_ban_hanh",
            "issuing_place": "noi_ban_hanh",
            "effective_date": "ngay_ban_hanh",
            "status": "tinh_trang",
            "hierarchy.chapter_number": "chuong",
            "hierarchy.chapter_title": "ten_chuong",
            "hierarchy.section_number": "muc",
            "hierarchy.section_title": "ten_muc",
            "hierarchy.article_number": "dieu",
            "hierarchy.article_title": "ten_dieu"
        }
        
        def convert_filter_to_metadata(single_filter: Dict) -> Dict:
            """Convert a single filter from MongoDB field names to metadata field names"""
            metadata_filter = {}
            for mongo_field, filter_value in single_filter.items():
                if mongo_field in mongo_to_metadata_field_map:
                    metadata_field = mongo_to_metadata_field_map[mongo_field]
                    metadata_filter[metadata_field] = filter_value
                else:
                    # If no mapping found, use the original field name
                    metadata_filter[mongo_field] = filter_value
            return metadata_filter
        
        def matches_single_filter(doc_metadata: Dict, metadata_filter: Dict) -> bool:
            """Check if document metadata matches a single filter (AND logic)"""
            for metadata_field, required_value in metadata_filter.items():
                doc_value = doc_metadata.get(metadata_field)
                
                # Handle different value types
                if doc_value is None:
                    return False
                
                # Convert both values to string for comparison (case-insensitive)
                doc_value_str = str(doc_value).lower().strip()
                required_value_str = str(required_value).lower().strip()
                
                # Check if the document value matches the required value
                if doc_value_str != required_value_str:
                    return False
            
            return True
        
        # Handle $or filters
        if "$or" in filters:
            or_conditions = filters["$or"]
            if not isinstance(or_conditions, list):
                print(f"Warning: $or value should be a list, got {type(or_conditions)}")
                return keyword_docs_with_scores
            
            # Convert each OR condition to metadata fields
            converted_or_conditions = []
            for condition in or_conditions:
                converted_condition = convert_filter_to_metadata(condition)
                converted_or_conditions.append(converted_condition)
            
            print(f"Converted $or filters: {len(or_conditions)} conditions")
            for i, (original, converted) in enumerate(zip(or_conditions, converted_or_conditions)):
                print(f"  Condition {i+1}: {original} -> {converted}")
            
            try:
                filtered_results = []
                
                for doc, score in keyword_docs_with_scores:
                    doc_metadata = doc.metadata
                    
                    # Check if document matches ANY of the OR conditions
                    matches_any_condition = False
                    
                    for i, metadata_condition in enumerate(converted_or_conditions):
                        if matches_single_filter(doc_metadata, metadata_condition):
                            matches_any_condition = True
                            print(f"  Document matches OR condition {i+1}: {metadata_condition}")
                            break
                    
                    if matches_any_condition:
                        filtered_results.append((doc, score))
                
                print(f"Keyword metadata filtering ($or): {len(filtered_results)}/{len(keyword_docs_with_scores)} documents match filters")
                return filtered_results
                
            except Exception as e:
                print(f"Error filtering keyword results by metadata ($or): {e}")
                return keyword_docs_with_scores
        
        else:
            # Handle simple filters (original logic)
            metadata_filters = convert_filter_to_metadata(filters)
            
            print(f"Converted filters: {filters} -> {metadata_filters}")
            
            
            try:
                filtered_results = []
                
                for doc, score in keyword_docs_with_scores:
                    doc_metadata = doc.metadata
                    
                    # Check if document metadata contains all required filter values
                    if matches_single_filter(doc_metadata, metadata_filters):
                        filtered_results.append((doc, score))
                
                print(f"Keyword metadata filtering: {len(filtered_results)}/{len(keyword_docs_with_scores)} documents match filters")
                return filtered_results
                
            except Exception as e:
                print(f"Error filtering keyword results by metadata: {e}")
                return keyword_docs_with_scores


    def _close_mongodb(self):
        """Close MongoDB connection"""
        if self.mongo_client:
            self.mongo_client.close()
            print("MongoDB connection closed.")

    def __del__(self):
        try:
            self._close_mongodb()
        except Exception as e:
            print(f"Warning in __del__: {e}")