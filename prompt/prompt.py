from langchain.prompts import PromptTemplate

qa_doc = """
Answer the user question. If necessary, use the below context to provide a more accurate answer.

**Context:**

{context}

**Question:**

{question}

**Answer:**
"""

qa_template = PromptTemplate(template=qa_doc,
                             input_variables=["context", "question"])