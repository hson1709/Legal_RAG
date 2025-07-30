import json
from typing import List, Dict
from tqdm import tqdm
from langchain.schema.output_parser import StrOutputParser
from utils.prompts import JSON_FORMAT_DOC_PROMPT
from langchain_core.prompts import ChatPromptTemplate


class LLMParser:
    
    def __init__(self, llm_model, prompt_template=None):

        self.llm = llm_model
        self.prompt_template = prompt_template or self._set_json_doc_promt()
        self.parser = StrOutputParser()
        self.chain = self.prompt_template | self.llm | self.parser
    
    def _set_json_doc_promt():
        system_prompt = JSON_FORMAT_DOC_PROMPT
        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", """
            Dưới đây là văn bản pháp lý bạn cần phân tích:
            {doc}
            """)
        ])
    
    def parse_single_document(self, doc_text: str) -> List[Dict]:

        try:
            # Invoke LLM chain
            json_str = self.chain.invoke({"doc": doc_text})
            
            # Clean JSON string
            json_str = json_str.strip().removeprefix("```json").removesuffix("```").strip()
            
            # Parse JSON
            parsed = json.loads(json_str)
            
            # Validate là list
            if not isinstance(parsed, list):
                print("LLM không trả về list JSON hợp lệ.")
                return []
                
            return parsed
            
        except json.JSONDecodeError as e:
            print(f"JSON decode failed: {e}")
            print(f"LLM Output: {json_str}")
            return []
        except Exception as e:
            print(f"Error parsing document: {e}")
            return []
    
    def parse(self, docs_list: List[str], show_progress: bool = True) -> List[Dict]:

        all_results = []
        
        iterator = tqdm(docs_list, desc="Parsing documents") if show_progress else docs_list
        
        for i, doc_text in enumerate(iterator):
            try:
                results = self.parse_single_document(doc_text)
                
                if results:
                    all_results.extend(results)
                    if show_progress:
                        print(f"Doc {i}: Thành công - {len(results)} objects")
                else:
                    if show_progress:
                        print(f"Doc {i}: Không extract được objects")
                        
            except Exception as e:
                if show_progress:
                    print(f"Doc {i}: Lỗi không xác định - {e}")
                continue
        
        return all_results
    
    def parse_with_retry(self, docs_list: List[str], max_retries: int = 2, show_progress: bool = True) -> List[Dict]:

        all_results = []
        failed_docs = list(enumerate(docs_list))
        
        for attempt in range(max_retries + 1):
            if not failed_docs:
                break
                
            if show_progress:
                print(f"\n--- Attempt {attempt + 1}/{max_retries + 1} ---")
                print(f"Processing {len(failed_docs)} documents")
            
            current_failed = []
            
            for original_idx, doc_text in failed_docs:
                try:
                    results = self.parse_single_document(doc_text)
                    
                    if results:
                        all_results.extend(results)
                        if show_progress:
                            print(f"Doc {original_idx}: Thành công - {len(results)} objects")
                    else:
                        current_failed.append((original_idx, doc_text))
                        if show_progress:
                            print(f"Doc {original_idx}: Failed, sẽ retry")
                            
                except Exception as e:
                    current_failed.append((original_idx, doc_text))
                    if show_progress:
                        print(f"Doc {original_idx}: Error - {e}")
            
            failed_docs = current_failed
        
        if show_progress:
            print(f"\n=== PARSING SUMMARY ===")
            print(f"Total documents: {len(docs_list)}")
            print(f"Successfully parsed: {len(docs_list) - len(failed_docs)}")
            print(f"Failed documents: {len(failed_docs)}")
            print(f"Total extracted objects: {len(all_results)}")
        
        return all_results
    
    def validate_json_structure(self, json_objects: List[Dict], required_fields: List[str] = None) -> List[Dict]:

        if not required_fields:
            required_fields = ["_id", "content_text"]  
        
        valid_objects = []
        
        for i, obj in enumerate(json_objects):
            is_valid = True
            missing_fields = []
            
            for field in required_fields:
                if field not in obj or not obj[field]:
                    missing_fields.append(field)
                    is_valid = False
            
            if is_valid:
                valid_objects.append(obj)
            else:
                print(f"Object {i}: Missing required fields: {missing_fields}")
        
        print(f"Validation: {len(valid_objects)}/{len(json_objects)} objects are valid")
        return valid_objects
    