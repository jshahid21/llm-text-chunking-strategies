!pip install langchain-text-splitters langchain_experimental spacy tiktoken sentence-transformers
!python -m spacy download en_core_web_sm

TEXT = '''Artificial intellegence is transforming healthcare rapidly.Machine learning models can now detect desease from medical images with remarkable accuray.This technology promises to make diagnosis faster and more accesible worldwide.'''

from langchain_text_splitters import CharacterTextSplitter,RecursiveCharacterTextSplitter, TokenTextSplitter,SpacyTextSplitter

#Paragraph Splitter
paragraph_splitter = CharacterTextSplitter()
para = paragraph_splitter.split_text(TEXT)
para

#sententce splitter
sentence_splitter =SpacyTextSplitter(pipeline="en_core_web_sm")
sent = sentence_splitter.split_text(TEXT)
sent

#fixed size chunking0
fixed_splitter = RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=20)
fixed = fixed_splitter.split_text(TEXT)
fixed

#slidign window
sliding_window = TokenTextSplitter(
    chunk_size=20,
    chunk_overlap=10,
    encoding_name="cl100k_base"
)
sliding_tokens = sliding_window.split_text(TEXT)
sliding_tokens

#semantic chunking
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
hf = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
semantic_splitter = SemanticChunker(hf)
semantic_chunks = semantic_splitter.split_text(TEXT)
semantic_chunks

#Entity Based Chunking
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp(TEXT)
sent_texts = [s.text.strip() for s in doc.sents if s.text.strip()]
print(sent_texts)
entity_map = {}
for ent in doc.ents:
  entity_map.setdefault(ent.text,set())
print(entity_map)
for s in sent_texts:
  s_doc = nlp(s)
  for ent in s_doc.ents:
    entity_map.setdefault(ent.text,set()).add(s)
entity_chunks=[]
for ent,ss in entity_map.items():
  entity_chunks.append(f"ENTITY :{ent}\n"+"\n".join(f"-{X}" for X in sorted(ss)))

entity_chunks=sent_texts
entity_chunks

from langchain_text_splitters import CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size=10, chunk_overlap=0,separator='',strip_whitespace=False)
text_splitter.create_documents([TEXT])
