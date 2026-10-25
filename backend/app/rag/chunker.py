from langchain_text_splitters import RecursiveCharacterTextSplitter

def generate_chunk(document):
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n","\n"," ",""]
    )

    chunks=text_splitter.split_documents(document)

    for idx, chunk in enumerate(chunks):
        page_num = chunk.metadata.get("page", 0) + 1 
        chunk.metadata["chunk_id"] = f"p{page_num}_c{idx}"
        chunk.metadata["page_number"] = page_num
    return chunks

if __name__=="__main__":
    from app.rag.pdf_loader import load_documents
    doc_path="C:/Users/vaish/Documents/ResearchAgent/backend/data/paper"
    docs=load_documents(doc_path)
    chunks=generate_chunk(docs)
    for i,chunk in enumerate(chunks):
        print(f"Chunk {i+1}:\n")
        print(f"Content: {chunk.page_content[:100]}")
        print(f"Page Number: {chunk.metadata['page_number']}")

