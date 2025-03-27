import os
import pdfplumber
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
import hashlib
import re

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# 載入環境變數
load_dotenv()

# 檢查API金鑰
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("請設定 OPENAI_API_KEY 環境變數")

# 定義文檔與對應數據庫的映射關係
DOC_DB_MAPPING = {
    "1.pdf": "stock_db",
    "2.pdf": "history_db",
    "galirage.md": "program_db"
}

# 數據庫路徑格式化為一致的方式
DB_DIRECTORIES = {
    "stock_db": "chroma_db_stock_db",
    "history_db": "chroma_db_history_db",
    "program_db": "chroma_db_program_db"
}

class DocumentProcessor:
    """處理不同類型文件的基礎類別"""
    
    def __init__(self, file_path: str):
        """
        初始化文件處理器
        
        Args:
            file_path: 檔案的路徑
        """
        self.file_path = file_path
        
    def extract_text(self) -> str:
        """
        從文件提取文字（由子類別實現）
        
        Returns:
            從文件提取的所有文字內容
        """
        raise NotImplementedError("This method should be implemented by subclasses")
    
    def extract_metadata(self) -> Dict[str, Any]:
        """
        提取文件的基本元數據
        
        Returns:
            包含文件元數據的字典
        """
        return {
            "filename": os.path.basename(self.file_path),
            "path": self.file_path,
            "file_type": os.path.splitext(self.file_path)[1]
        }

class PDFProcessor(DocumentProcessor):
    """使用 pdfplumber 處理 PDF 文件的類別"""
    
    def extract_text(self) -> str:
        """
        使用 pdfplumber 從 PDF 提取文字
        
        Returns:
            從 PDF 提取的所有文字內容
        """
        all_text = []
        
        try:
            with pdfplumber.open(self.file_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        all_text.append(text)
                        
            return "\n".join(all_text)
        except Exception as e:
            print(f"處理 PDF 時發生錯誤 {self.file_path}: {str(e)}")
            return ""
    
    def extract_tables(self) -> List[pd.DataFrame]:
        """
        從 PDF 提取表格
        
        Returns:
            PDF 中所有表格的列表，每個表格都是 pandas DataFrame
        """
        all_tables = []
        
        try:
            with pdfplumber.open(self.file_path) as pdf:
                for page in pdf.pages:
                    tables = page.extract_tables()
                    for table in tables:
                        if table:
                            # 將表格轉換為 DataFrame
                            df = pd.DataFrame(table[1:], columns=table[0])
                            all_tables.append(df)
                            
            return all_tables
        except Exception as e:
            print(f"從 PDF 提取表格時發生錯誤 {self.file_path}: {str(e)}")
            return []
    
    def extract_metadata(self) -> Dict[str, Any]:
        """
        提取 PDF 的元數據
        
        Returns:
            包含 PDF 元數據的字典
        """
        metadata = super().extract_metadata()
        try:
            with pdfplumber.open(self.file_path) as pdf:
                pdf_metadata = pdf.metadata
                # 添加基本文件信息
                metadata.update({
                    "pages": len(pdf.pages),
                    **pdf_metadata
                })
                return metadata
        except Exception as e:
            print(f"提取 PDF 元數據時發生錯誤 {self.file_path}: {str(e)}")
            return metadata

class MarkdownProcessor(DocumentProcessor):
    """處理 Markdown 文件的類別"""
    
    def extract_text(self) -> str:
        """
        從 Markdown 提取文字
        
        Returns:
            從 Markdown 提取的所有文字內容
        """
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"處理 Markdown 時發生錯誤 {self.file_path}: {str(e)}")
            return ""
            
    def extract_sections(self) -> List[Dict[str, Any]]:
        """
        從 Markdown 中提取按 '## File:' 分隔的段落
        
        Returns:
            包含每個段落及其元數據的字典列表
        """
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 使用正則表達式匹配 "## File:" 開頭的標題
            sections = re.split(r'(## File: .*?\n)', content)
            
            # 處理分割結果，將標題與內容配對
            result = []
            current_title = ""
            current_content = ""
            
            for i, section in enumerate(sections):
                if section.startswith("## File:"):
                    # 如果有之前的內容，先保存
                    if current_title and current_content:
                        file_path = current_title.replace("## File:", "").strip()
                        result.append({
                            "title": current_title.strip(),
                            "file_path": file_path,
                            "content": current_content.strip(),
                            "section_index": len(result)
                        })
                    
                    # 新的標題
                    current_title = section
                    current_content = ""
                else:
                    # 內容部分
                    current_content += section
            
            # 添加最後一個部分
            if current_title and current_content:
                file_path = current_title.replace("## File:", "").strip()
                result.append({
                    "title": current_title.strip(),
                    "file_path": file_path,
                    "content": current_content.strip(),
                    "section_index": len(result)
                })
            
            return result
        except Exception as e:
            print(f"從 Markdown 提取段落時發生錯誤 {self.file_path}: {str(e)}")
            return []
    
    def extract_metadata(self) -> Dict[str, Any]:
        """
        提取 Markdown 的元數據
        
        Returns:
            包含 Markdown 元數據的字典
        """
        metadata = super().extract_metadata()
        # 可以在這裡添加更多 Markdown 特定的元數據
        try:
            # 計算行數和字數等統計數據
            with open(self.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                metadata.update({
                    "line_count": len(content.splitlines()),
                    "character_count": len(content)
                })
        except Exception as e:
            print(f"計算 Markdown 統計數據時發生錯誤: {str(e)}")
        
        return metadata

# 獲取文件處理器的工廠函數
def get_document_processor(file_path: str) -> DocumentProcessor:
    """
    根據文件類型返回適當的處理器
    
    Args:
        file_path: 文件路徑
        
    Returns:
        對應的文件處理器實例
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        return PDFProcessor(file_path)
    elif ext in ['.md', '.markdown']:
        return MarkdownProcessor(file_path)
    else:
        raise ValueError(f"不支持的文件類型: {ext}")

# 計算文檔的哈希值，用於檢查文檔是否已更新
def get_file_hash(file_path: str) -> str:
    with open(file_path, 'rb') as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    return file_hash

class RAGSystem:
    """RAG 系統主類別"""
    
    def __init__(self, 
                 chunk_size: int = 500, 
                 chunk_overlap: int = 100,
                 embedding_model: str = "text-embedding-3-large",
                 chat_model: str = "gpt-4o-mini"
                 ):
        """
        初始化 RAG 系統
        
        Args:
            chunk_size: 文件分塊的大小
            chunk_overlap: 文件分塊的重疊大小
            embedding_model: 使用的 OpenAI 嵌入模型名稱
            chat_model: 使用的 OpenAI 聊天模型名稱
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.chat_model = ChatOpenAI(model=chat_model, temperature=0.7)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        self.vector_dbs = {}  # 存儲多個向量資料庫的字典
        self.retrievers = {}  # 存儲多個檢索器的字典
        self.loaded_dbs = set()  # 追蹤已載入的資料庫
        
    def process_document(self, file_path: str) -> List[Document]:
        """
        處理單個文件 (PDF 或 Markdown)
        
        Args:
            file_path: 檔案的路徑
            
        Returns:
            包含分塊文檔的列表
        """
        processor = get_document_processor(file_path)
        docs = []
        
        # 檢查是否為 Markdown 文件且是否為 galirage.md （需要特殊處理）
        if isinstance(processor, MarkdownProcessor) and os.path.basename(file_path) == "galirage.md":
            # 提取按 "## File:" 分割的各個段落
            sections = processor.extract_sections()
            base_metadata = processor.extract_metadata()
            
            if not sections:
                print(f"警告: 從 {file_path} 無法提取段落")
                return []
            
            # 為每個段落創建文檔
            for section in sections:
                section_metadata = {
                    **base_metadata,
                    "title": section["title"],
                    "file_path": section["file_path"],
                    "section_index": section["section_index"],
                    "content_type": "markdown_section"
                }
                
                # 分割每個段落內容
                section_docs = self.text_splitter.create_documents(
                    texts=[section["content"]],
                    metadatas=[section_metadata]
                )
                
                docs.extend(section_docs)
        else:
            # 常規處理方式 (PDF或非特殊Markdown)
            text = processor.extract_text()
            metadata = processor.extract_metadata()
            
            if not text:
                print(f"警告: 從 {file_path} 無法提取文字")
                return []
                
            # 將文本分割成多個區塊
            docs = self.text_splitter.create_documents(
                texts=[text],
                metadatas=[metadata]
            )
            
            # 處理表格 (如果是 PDF 且有表格)
            if isinstance(processor, PDFProcessor):
                tables = processor.extract_tables()
                if tables:
                    for i, table in enumerate(tables):
                        table_text = table.to_string(index=False)
                        table_doc = Document(
                            page_content=table_text,
                            metadata={
                                **metadata,
                                "content_type": "table",
                                "table_index": i
                            }
                        )
                        docs.append(table_doc)
                
        print(f"從 {file_path} 提取了 {len(docs)} 個文檔區塊")
        return docs
    
    def process_file_by_type(self, file_path: str) -> Optional[str]:
        """
        處理特定文件並將其存儲到對應的數據庫中
        
        Args:
            file_path: 檔案的路徑
            
        Returns:
            數據庫路徑 (如果成功處理)，否則 None
        """
        file_name = os.path.basename(file_path)
        
        # 檢查文件是否在映射關係中
        if file_name not in DOC_DB_MAPPING:
            print(f"警告: 檔案 {file_name} 沒有對應的數據庫映射，將被忽略")
            return None
        
        db_name = DOC_DB_MAPPING[file_name]
        db_path = DB_DIRECTORIES[db_name]
        
        # 檢查數據庫是否已存在
        if os.path.exists(db_path):
            # 檢查文件是否已更新
            hash_file_path = os.path.join(db_path, f"{file_name}.hash")
            current_hash = get_file_hash(file_path)
            
            if os.path.exists(hash_file_path):
                try:
                    with open(hash_file_path, 'r') as f:
                        stored_hash = f.read().strip()
                    
                    if stored_hash == current_hash:
                        print(f"文件 {file_name} 未變更，使用現有的 {db_name} 數據庫")
                        # 載入現有數據庫
                        self.load_vector_db(db_name, db_path)
                        return db_path
                    else:
                        print(f"文件 {file_name} 已更新，重新建立 {db_name} 數據庫")
                except Exception as e:
                    print(f"讀取哈希值時發生錯誤: {str(e)}")
            else:
                print(f"找不到哈希值文件，將重新處理 {file_name}")
        else:
            print(f"數據庫 {db_name} 不存在，將處理文件 {file_name}")
        
        # 處理文件
        documents = self.process_document(file_path)
        
        if not documents:
            print(f"無法從 {file_name} 提取有效內容")
            return None
        
        # 創建向量存儲
        self.create_vector_store(documents, db_name=db_name, persist_directory=db_path)
        
        # 儲存文件哈希值
        os.makedirs(db_path, exist_ok=True)
        current_hash = get_file_hash(file_path)
        with open(os.path.join(db_path, f"{file_name}.hash"), 'w') as f:
            f.write(current_hash)
        
        return db_path
    
    def load_vector_db(self, db_name: str, db_path: str) -> None:
        """
        載入現有向量數據庫
        
        Args:
            db_name: 數據庫名稱
            db_path: 數據庫路徑
        """
        try:
            if not os.path.exists(db_path):
                print(f"警告: 數據庫路徑 {db_path} 不存在")
                return
            
            vector_db = Chroma(
                embedding_function=self.embeddings,
                persist_directory=db_path
            )
            self.vector_dbs[db_name] = vector_db
            
            # 建立檢索器
            self.retrievers[db_name] = vector_db.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            )
            
            self.loaded_dbs.add(db_name)
            print(f"成功載入數據庫 {db_name}")
        except Exception as e:
            print(f"載入數據庫 {db_name} 時發生錯誤: {str(e)}")
    
    def create_vector_store(self, documents: List[Document], db_name: str, persist_directory: str) -> Chroma:
        """
        從文檔區塊創建向量存儲
        
        Args:
            documents: 要嵌入的文檔列表
            db_name: 數據庫名稱
            persist_directory: 持久化向量存儲的目錄
            
        Returns:
            Chroma 向量存儲實例
        """
        if not documents:
            raise ValueError("沒有文檔可以嵌入，無法創建向量資料庫")
            
        print(f"正在為 {len(documents)} 個文檔創建向量存儲 {db_name}...")
        
        # 確保目錄存在
        os.makedirs(persist_directory, exist_ok=True)
        
        # 創建 Chroma 向量存儲
        vector_db = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=persist_directory
        )
        
        self.vector_dbs[db_name] = vector_db
        
        # 建立檢索器
        self.retrievers[db_name] = vector_db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10}
        )
        
        self.loaded_dbs.add(db_name)
        print(f"向量存儲 {db_name} 已持久化到 {persist_directory}")
        return vector_db
    
    def similarity_search(self, query: str, db_name: Optional[str] = None, k: int = 3) -> List[Document]:
        """
        在向量存儲中進行相似度搜索
        
        Args:
            query: 搜索查詢
            db_name: 要搜索的數據庫名稱 (如果為 None，則搜索所有數據庫)
            k: 要檢索的最相似文檔數量
            
        Returns:
            最相似文檔的列表
        """
        if not self.vector_dbs:
            raise ValueError("尚未創建任何向量存儲，請先處理文檔")
        
        if db_name and db_name in self.vector_dbs:
            results = self.vector_dbs[db_name].similarity_search(query, k=4)
            # 標記來源數據庫
            for doc in results:
                doc.metadata["db_source"] = db_name
            return results
        
        # 如果沒有指定數據庫或數據庫不存在，則搜索所有數據庫
        all_results = []
        for db_key, vector_db in self.vector_dbs.items():
            try:
                results = vector_db.similarity_search(query, k=4)
                # 標記來源數據庫
                for doc in results:
                    doc.metadata["db_source"] = db_key
                all_results.extend(results)
            except Exception as e:
                print(f"在數據庫 {db_key} 中搜索時發生錯誤: {str(e)}")
        
        # 根據相關性排序所有結果 (這裡可以實現自定義的排序邏輯)
        # 暫時返回前 k 個結果
        return all_results[:k] if all_results else []
    
    def setup_qa_chain(self, db_name: str):
        """
        使用 LCEL 設置問答鏈，用於與檢索到的文檔一起生成回答
        
        Args:
            db_name: 要使用的數據庫名稱

        Returns:
            設置好的問答鏈
        """
        if db_name not in self.vector_dbs:
            raise ValueError(f"數據庫 {db_name} 未載入，無法設置問答鏈")

        # 設定檢索器
        retriever = self.vector_dbs[db_name].as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10}
        )
        
        # 儲存檢索器
        self.retrievers[db_name] = retriever

        # 建立提示模板
        system_template = """你是一個專業的助手，負責回答與公司文件相關的問題。
        僅使用以下檢索到的上下文信息來回答問題。如果上下文中沒有足夠的信息來回答問題，
        請說明"我找不到足夠的信息來回答這個問題"。請不要編造信息。

        上下文信息：
        {context}
        """

        messages = [
            SystemMessage(content=system_template),
            HumanMessage(content="{question}")
        ]

        prompt = ChatPromptTemplate.from_messages(messages)

        # 使用 LCEL 建立查詢鏈
        qa_chain = (
            {
                "context": retriever | RunnableLambda(self.format_docs),
                "question": RunnablePassthrough()
            }
            | prompt
            | self.chat_model
            | StrOutputParser()
        )

        # 建立帶源文檔的查詢鏈
        qa_chain_with_sources = {
            "context": retriever | RunnableLambda(self.format_docs),
            "question": RunnablePassthrough(),
            "docs": retriever
        } | RunnablePassthrough().assign(
            answer=(
                prompt
                | self.chat_model
                | StrOutputParser()
            )
        )

        return qa_chain, qa_chain_with_sources
    
    def format_docs(self, docs):
        """
        格式化文檔以供提示使用

        Args:
            docs: 檢索到的文檔列表

        Returns:
            格式化後的文檔文本
        """
        if not docs:
            return "沒有找到相關文檔。"
            
        formatted_docs = []
        
        for doc in docs:
            db_source = doc.metadata.get('db_source', 'Unknown')
            
            # 針對 program_db 中的 Markdown 段落添加更多詳細信息
            if db_source == 'program_db' and doc.metadata.get('content_type') == 'markdown_section':
                file_path = doc.metadata.get('file_path', '')
                section_header = f"[From: {db_source}, File: {file_path}]"
                formatted_docs.append(f"{section_header}\n{doc.page_content}")
            else:
                # 一般文檔的格式化
                formatted_docs.append(f"[From: {db_source}]\n{doc.page_content}")
        
        return "\n\n" + "\n\n".join(formatted_docs)
    
    def determine_database_type(self, query: str) -> str:
        """
        使用 ChatGPT 根據用戶輸入的問題內容，決定使用哪個資料庫
        
        Args:
            query: 用戶輸入的問題
            
        Returns:
            資料庫類型名稱: "program_db", "history_db", 或 "stock_db"
        """
        # 構建提示訊息，要求 ChatGPT 分類問題
        system_message = """あなたは専門的な質問分類器です。あなたの任務は、ユーザーの質問を次の3つのタイプのいずれかに分類することです：
        1. program - プログラミングやコードに関連する質問
        2. history - 2025年以前の会社に関連する質問
        3. stock - 上記のカテゴリに属さないその他の質問
        
        あなたは「program」、「history」または「stock」のみで回答してください。他の説明は一切提供しないでください。質問内容に基づいて最も適切なカテゴリを選んでください。
        """
        
        # 構建 ChatGPT 請求
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=query)
        ]
        
        # 發送請求給 ChatGPT
        try:
            response = self.chat_model.invoke(messages)
            classification = response.content.strip().lower()
            
            # 標準化回應
            if "program" in classification:
                db_type = "program_db"
            elif "history" in classification:
                db_type = "history_db"
            else:
                db_type = "stock_db"
                
            print(f"ChatGPT 分類結果: {classification} -> 使用 {db_type}")
            return db_type
        
        except Exception as e:
            print(f"使用 ChatGPT 分類問題時發生錯誤: {str(e)}")
            print("預設使用 stock_db")
            return "stock_db"
    
    def chat(self, query: str, db_name: Optional[str] = None, with_sources: bool = False) -> Dict[str, Any]:
        """
        與 ChatGPT 對話，利用檢索到的相關文檔生成回答
        
        Args:
            query: 使用者的問題
            db_name: 要查詢的數據庫名稱 (如果為 None，則由系統決定)
            with_sources: 是否返回來源文檔
            
        Returns:
            包含回答（和可選的來源文檔）的字典
        """
        # 如果沒有指定數據庫，使用 ChatGPT 決定
        if db_name is None:
            db_name = self.determine_database_type(query)
        
        # 檢查是否有任何數據庫已載入
        if not self.loaded_dbs:
            return {"answer": "系統尚未載入任何數據庫，請先處理文檔後再嘗試查詢。", "db_used": None}
        
        # 確保數據庫已載入
        if db_name not in self.loaded_dbs:
            db_path = DB_DIRECTORIES.get(db_name)
            if db_path and os.path.exists(db_path):
                self.load_vector_db(db_name, db_path)
            else:
                print(f"警告: 數據庫 {db_name} 不存在或未載入")
                # 如果指定的數據庫不存在，使用第一個已載入的數據庫
                db_name = next(iter(self.loaded_dbs))
                print(f"將使用可用的數據庫: {db_name}")
        
        try:
            # 首先使用 similarity_search 獲取相關文檔
            relevant_docs = self.similarity_search(query, db_name=db_name, k=4)
            
            if not relevant_docs:
                return {
                    "answer": "找不到與查詢相關的文檔。",
                    "db_used": db_name
                }
            
            # 格式化檢索到的文檔
            formatted_docs = self.format_docs(relevant_docs)
            
            # 創建系統提示
            system_template = f"""你是一個專業的助手，負責回答與公司文件相關的問題。
            僅使用以下檢索到的上下文信息來回答問題。如果上下文中沒有足夠的信息來回答問題，
            請說明"我找不到足夠的信息來回答這個問題"。請不要編造信息。

            上下文信息：
            {formatted_docs}
            """

            messages = [
                SystemMessage(content=system_template),
                HumanMessage(content=query)
            ]

            # 直接使用 chat_model 而不是 qa_chain
            if with_sources:
                # 生成回答
                response = self.chat_model.invoke(messages)
                answer = response.content
                
                # 增強源文檔的顯示
                enhanced_docs = []
                for doc in relevant_docs:
                    # 添加數據庫來源標記
                    doc.metadata['db_source'] = db_name
                    
                    if db_name == 'program_db' and doc.metadata.get('content_type') == 'markdown_section':
                        # 為程式碼段落添加文件路徑信息
                        doc.metadata['display_info'] = f"File: {doc.metadata.get('file_path', 'Unknown')}"
                    enhanced_docs.append(doc)
                
                return {
                    "answer": answer,
                    "docs": enhanced_docs,
                    "db_used": db_name
                }
            else:
                # 生成回答
                response = self.chat_model.invoke(messages)
                answer = response.content
                
                return {
                    "answer": answer,
                    "db_used": db_name
                }
                
        except Exception as e:
            print(f"生成回答時發生錯誤: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # 返回錯誤信息
            return {
                "answer": f"生成回答時發生錯誤: {str(e)}",
                "db_used": db_name
            }

def main():
    """主函數示例用法"""
    
    # 創建 RAG 系統
    rag = RAGSystem(chunk_size=500, chunk_overlap=200, chat_model="gpt-4o-mini")
    
    # 文件路徑 - 確保這些目錄和檔案存在
    dataset_dir = "dataset"
    os.makedirs(dataset_dir, exist_ok=True)
    
    files = []
    for file_name in DOC_DB_MAPPING.keys():
        file_path = os.path.join(dataset_dir, file_name)
        files.append(file_path)
        if not os.path.exists(file_path):
            print(f"警告: 檔案 {file_path} 不存在，請確保所有必要檔案都已放置在正確位置")
    
    # 處理每個文件並存儲到對應的數據庫
    processed_dbs = []
    for file_path in files:
        if os.path.exists(file_path):
            try:
                db_path = rag.process_file_by_type(file_path)
                if db_path:
                    processed_dbs.append(db_path)
            except Exception as e:
                print(f"處理檔案 {file_path} 時發生錯誤: {str(e)}")
                import traceback
                traceback.print_exc()
        else:
            print(f"檔案 {file_path} 不存在，跳過處理")
    
    if not processed_dbs and not rag.loaded_dbs:
        print("警告: 沒有成功處理任何文件且沒有載入任何數據庫")
        print("系統將創建空的示例數據庫以便示範")
        
        # 創建空的示例數據庫
        for db_name, db_path in DB_DIRECTORIES.items():
            sample_doc = [Document(
                page_content=f"這是 {db_name} 的示例文檔",
                metadata={"source": "sample", "filename": f"sample_{db_name}.txt"}
            )]
            try:
                rag.create_vector_store(sample_doc, db_name=db_name, persist_directory=db_path)
            except Exception as e:
                print(f"創建示例數據庫 {db_name} 時發生錯誤: {str(e)}")
    
    # 開始對話循環
    print("\n歡迎使用多資料庫 RAG 聊天系統！")
    print("輸入 'exit' 或 'quit' 結束對話。")
    print("輸入 'sources' 切換是否顯示來源。")
    print("輸入 'db:名稱' 選擇特定的數據庫 (stock_db, history_db, program_db), 輸入 'db:auto' 使用自動選擇")
    print("已載入的數據庫: " + (", ".join(rag.loaded_dbs) if rag.loaded_dbs else "無"))
    
    with_sources = False
    current_db = None
    
    while True:
        user_input = input("\n請輸入您的問題: ")
        
        if user_input.lower() in ['exit', 'quit', '退出']:
            print("感謝使用，再見！")
            break
        
        if user_input.lower() == 'sources':
            with_sources = not with_sources
            print(f"{'開啟' if with_sources else '關閉'}顯示文檔來源")
            continue
        
        if user_input.lower().startswith('db:'):
            db_choice = user_input[3:].strip()
            if db_choice == 'auto':
                current_db = None
                print("使用自動選擇資料庫模式")
            elif db_choice in ["program_db", "history_db", "stock_db"]:
                current_db = db_choice
                print(f"使用 {current_db} 資料庫進行查詢")
            else:
                print(f"未知的資料庫: {db_choice}")
                print(f"可用的資料庫: program_db, history_db, stock_db, auto")
            continue
        
        try:
            # 獲取回答
            result = rag.chat(user_input, db_name=current_db, with_sources=with_sources)
            
            db_used = result.get("db_used", "自動選擇")
            if db_used:
                print(f"\n使用資料庫: {db_used}")
            
            print(f"\nAI回答: {result['answer']}")
            
            if with_sources and 'docs' in result:
                print("\n相關文檔來源:")
                for i, doc in enumerate(result['docs']):
                    print(f"\n文檔 {i+1}:")
                    print(f"內容: {doc.page_content[:150]}...")
                    
                    # 根據不同類型的文檔顯示不同的元數據
                    if doc.metadata.get('db_source') == 'program_db' and doc.metadata.get('content_type') == 'markdown_section':
                        print(f"來源: {doc.metadata.get('file_path', 'Unknown')} (DB: {doc.metadata.get('db_source', 'Unknown')})")
                    else:
                        print(f"來源: {doc.metadata.get('filename', 'Unknown')} (DB: {doc.metadata.get('db_source', 'Unknown')})")
                        
        except Exception as e:
            print(f"發生錯誤: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()