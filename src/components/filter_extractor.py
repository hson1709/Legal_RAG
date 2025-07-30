import re
import json
from typing import Dict, Any
from datetime import datetime
from langchain.schema import HumanMessage
from utils.prompts import FILTER_EXTRACT_PROMPT


class FilterExtractor:
    """Extract and process MongoDB filters from user queries"""

    def __init__(self, llm):
        self.llm = llm
        self.filter_extract_prompt = FILTER_EXTRACT_PROMPT


        self.allowed_fields = {
            "document_code", "document_type", "issuing_authority",
            "chapter_number", "section_number", "article_number"
        }

    def _clean_json_string(self, raw_content: str) -> str:
        """Clean JSON string from LLM response"""
        return re.sub(r"^```json\s*|\s*```$", "", raw_content.strip())

    def _normalize_date_format(self, date_str: str) -> str:
        """Normalize date string to YYYY-MM-DD"""
        if not date_str or date_str.strip() == "":
            return ""

        date_str = date_str.strip()

        patterns = [
            (r'^(\d{1,2})[/-](\d{1,2})[/-](\d{4})$', lambda m: f"{m.group(3)}-{m.group(2).zfill(2)}-{m.group(1).zfill(2)}"),
            (r'^(\d{1,2})[/-](\d{4})$', lambda m: f"{m.group(2)}-{m.group(1).zfill(2)}-01"),
            (r'^(\d{4})[/-](\d{1,2})$', lambda m: f"{m.group(1)}-{m.group(2).zfill(2)}-01"),
            (r'^(\d{4})$', lambda m: f"{m.group(1)}-01-01"),
            (r'^(\d{4})-(\d{1,2})-(\d{1,2})$', lambda m: f"{m.group(1)}-{m.group(2).zfill(2)}-{m.group(3).zfill(2)}")
        ]

        for pattern, formatter in patterns:
            match = re.match(pattern, date_str)
            if match:
                try:
                    formatted_date = formatter(match)
                    datetime.strptime(formatted_date, "%Y-%m-%d")
                    return formatted_date
                except ValueError:
                    continue

        print(f"Warning: Could not normalize date format: {date_str}")
        return date_str

    def _process_date_filter(self, date_value: Any) -> Any:
        """Process effective_date field"""
        if isinstance(date_value, str):
            return self._normalize_date_format(date_value)
        elif isinstance(date_value, dict):
            return {k: self._normalize_date_format(v) for k, v in date_value.items() if v}
        return date_value

    def get_query_filter(self, user_query: str) -> Dict:
        """Extract and return MongoDB-compatible filter"""
        try:
            prompt = self.filter_extract_prompt.replace("{user_query}", user_query)
            response = self.llm.invoke([HumanMessage(content=prompt)])

            try:
                cleaned_content = self._clean_json_string(response.content)
                query_json = json.loads(cleaned_content)
            except json.JSONDecodeError:
                raise ValueError(f"Lỗi khi parse JSON từ output của LLM:\n{response.content}")

            # Case 1: LLM return 1 filter dict
            if isinstance(query_json, dict):
                filter_obj = self._process_single_filter(query_json)
                return filter_obj

            # Case 2: LLM return a list of filters
            elif isinstance(query_json, list):
                or_filters = []
                for sub_filter in query_json:
                    if isinstance(sub_filter, dict):
                        processed = self._process_single_filter(sub_filter)
                        if processed:
                            or_filters.append(processed)
                if or_filters:
                    final = {"$or": or_filters}
                    print(f"Extracted OR filter: {final}")
                    return final
                else:
                    print("No valid filters extracted from query (list)")
                    return {}

            else:
                raise ValueError("LLM returned unexpected format (not dict or list)")

        except Exception as e:
            print(f"Error extracting filter: {e}")
            return {}

    def _process_single_filter(self, query_json: Dict) -> Dict:
        """Process a single filter dict"""
        final_filter = {}

        for field in self.allowed_fields:
            if field in query_json:
                value = query_json[field]
                if isinstance(value, str) and value.strip() == "":
                    continue
                if value is None:
                    continue

                if field == "effective_date":
                    processed = self._process_date_filter(value)
                    if processed:
                        final_filter[field] = processed
                elif field in ["chapter_number", "section_number", "article_number"]:
                    final_filter[f"hierarchy.{field}"] = value
                else:
                    final_filter[field] = value

        return final_filter

    def create_date_range_filter(self, start_date: str = None, end_date: str = None) -> Dict:
        """Optional: Build date range filter"""
        date_filter = {}
        if start_date:
            norm_start = self._normalize_date_format(start_date)
            if norm_start:
                date_filter["$gte"] = norm_start
        if end_date:
            norm_end = self._normalize_date_format(end_date)
            if norm_end:
                date_filter["$lte"] = norm_end
        return date_filter if date_filter else None
