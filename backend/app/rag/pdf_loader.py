from langchain_community.document_loaders import PyMuPDFLoader,DirectoryLoader,PyPDFLoader
from dotenv import load_dotenv
import os
from schemas.metadata import ResearchMetadata
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
llm=ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)
structured_llm=llm.with_structured_output(ResearchMetadata)

def load_documents(file_path:str):
    if not os.path.exists(file_path):
        raise FileNotFoundError
    loader=DirectoryLoader(
        path=file_path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True
    )
    document=loader.load()
    return document

def extract_first_page(pdf_path:str)->str:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError
    loader=PyMuPDFLoader(pdf_path)
    document=loader.load()

    if len(document)==0:
        return ""
    return document[0].page_content
    
def extract_metadata(page_content:str)->ResearchMetadata|None:
    if len(page_content)==0:
        return None
    prompt=f"""
    You are an expert Meta-Data extractor who extracts information metadata from Research Papers/Conference Papers/Journals etc
    of specific given format without providing hallucinated
    information and false information metadata.Your Task is to extract and provide metadata of specific given format.
    Rules:
    - Do not invent missing information.
    - Use None for unavailable scalar fields.
    - Return only information supported by the text.
    - Distinguish author names from affiliations.

    PAPER_TEXT:{page_content}
"""
    metadata=structured_llm.invoke(prompt)
    return metadata

if __name__=="__main__":
    # first_page=extract_first_page("C:/Users/vaish/Documents/ResearchAgent/backend/data/paper/NeurIPS-2024-crag-comprehensive-rag-benchmark-Paper-Datasets_and_Benchmarks_Track.pdf")
    # metadata=extract_metadata(first_page)
    # print(metadata.authors)
    doc=load_documents("C:/Users/vaish/Documents/ResearchAgent/backend/data/paper")
    print(doc[1].page_content)