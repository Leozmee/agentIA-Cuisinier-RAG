import os
import chromadb
from chromadb.utils import embedding_functions
import pdfplumber
import time

os.environ["CHROMA_ENABLE_TELEMETRY"] = "false"

# Configuration
PDF_FILES = [
    "/home/utilisateur/proget-agentIA-RAG/RAG/Allergenes-FPO-Enseigne.pdf",
    "/home/utilisateur/proget-agentIA-RAG/RAG/Allergenes-Pizza-Rhuys.pdf", 
    "/home/utilisateur/proget-agentIA-RAG/RAG/marco-fuso-recipe-booklet---final.pdf",
    "/home/utilisateur/proget-agentIA-RAG/RAG/Pizza-booklet-French.pdf",
    "/home/utilisateur/proget-agentIA-RAG/RAG/Pizza-maison.pdf",
    "/home/utilisateur/proget-agentIA-RAG/RAG/Recette-pizza-au-fromage.pdf",
    "/home/utilisateur/proget-agentIA-RAG/RAG/Tableau-des-allergenes.pdf"
]

COLLECTION_NAME = "pizzeria_collection"
EMBEDDING_MODEL = "mxbai-embed-large"
BATCH_SIZE = 20  # Traiter 20 chunks à la fois

def load_and_chunk_pdf_improved(file_path, chunk_size=800, chunk_overlap=100):
    """Version optimisée avec chunks plus petits"""
    print(f"Chargement du fichier : {os.path.basename(file_path)}")
    
    try:
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n--- Page {page_num + 1} ---\n"
                    text += page_text + "\n"
                
                tables = page.extract_tables()
                if tables:
                    for table_num, table in enumerate(tables):
                        text += f"\n--- Tableau {table_num + 1} Page {page_num + 1} ---\n"
                        for row in table:
                            if row and any(cell for cell in row if cell):
                                row_text = " | ".join([str(cell).strip() if cell else "" for cell in row])
                                text += row_text + "\n"
        
        print(f"  ✅ {os.path.basename(file_path)} : {len(text)} caractères extraits")
        
        if "curry" in text.lower():
            print(f"  🎯 'curry' détecté dans {os.path.basename(file_path)} !")
        
        # Chunks plus petits pour éviter timeout
        chunks = []
        file_name = os.path.basename(file_path)
        
        for i in range(0, len(text), chunk_size - chunk_overlap):
            chunk_text = text[i : i + chunk_size]
            chunk_with_source = f"[Source: {file_name}]\n{chunk_text}"
            chunks.append(chunk_with_source)
        
        print(f"  📦 {len(chunks)} chunks créés pour {file_name}")
        return chunks
        
    except Exception as e:
        print(f"❌ Erreur lors du traitement de {file_path}: {e}")
        return []

def add_chunks_in_batches(collection, all_chunks, batch_size=20):
    """Ajouter les chunks par lots pour éviter timeout"""
    total_chunks = len(all_chunks)
    print(f"📦 Ajout de {total_chunks} chunks par lots de {batch_size}...")
    
    for i in range(0, total_chunks, batch_size):
        batch_end = min(i + batch_size, total_chunks)
        batch_chunks = all_chunks[i:batch_end]
        batch_ids = [f"chunk_{j}" for j in range(i, batch_end)]
        
        print(f"  📤 Lot {i//batch_size + 1} : chunks {i+1} à {batch_end}")
        
        try:
            collection.add(documents=batch_chunks, ids=batch_ids)
            print(f"  ✅ Lot ajouté avec succès")
            time.sleep(1)  # Petite pause entre les lots
        except Exception as e:
            print(f"  ❌ Erreur lot {i//batch_size + 1}: {e}")
            # Essayer chunk par chunk pour ce lot
            for j, (doc, doc_id) in enumerate(zip(batch_chunks, batch_ids)):
                try:
                    collection.add(documents=[doc], ids=[doc_id])
                    print(f"    ✅ Chunk {doc_id} ajouté individuellement")
                except Exception as chunk_error:
                    print(f"    ❌ Échec chunk {doc_id}: {chunk_error}")

# Initialisation
print("Initialisation de ChromaDB...")
client = chromadb.PersistentClient(path="./chroma_db")

print("Initialisation de la fonction d'embedding via Ollama...")
ollama_ef = embedding_functions.OllamaEmbeddingFunction(
    url="http://localhost:11434/api/embeddings",
    model_name=EMBEDDING_MODEL,
)

# Suppression et recréation de la collection
print(f"Suppression de l'ancienne collection {COLLECTION_NAME} si elle existe...")
try:
    client.delete_collection(name=COLLECTION_NAME)
    print("✅ Ancienne collection supprimée")
except:
    print("ℹ️ Aucune collection à supprimer")

print(f"Création de la nouvelle collection : {COLLECTION_NAME}")
collection = client.get_or_create_collection(
    name=COLLECTION_NAME, embedding_function=ollama_ef
)

# Traitement des PDFs
all_chunks = []
files_with_curry = []

print(f"Traitement de {len(PDF_FILES)} fichiers PDF...")

for pdf_file in PDF_FILES:
    if os.path.exists(pdf_file):
        pdf_chunks = load_and_chunk_pdf_improved(pdf_file)
        
        if pdf_chunks:
            all_chunks.extend(pdf_chunks)
            print(f"✅ {os.path.basename(pdf_file)} traité avec succès")
            
            file_content = " ".join(pdf_chunks)
            if "curry" in file_content.lower():
                files_with_curry.append(os.path.basename(pdf_file))
        else:
            print(f"❌ Échec du traitement de {os.path.basename(pdf_file)}")
    else:
        print(f"❌ Fichier non trouvé : {pdf_file}")

print(f"\n--- Résumé ---")
print(f"Total de chunks créés : {len(all_chunks)}")
print(f"Fichiers contenant 'curry' : {files_with_curry}")

if all_chunks:
    # Ajout par lots
    add_chunks_in_batches(collection, all_chunks, BATCH_SIZE)
    
    print(f"\n--- Base de données vectorielle créée avec succès ! ---")
    print(f"Nombre de documents stockés : {collection.count()}")
    
    # Test immédiat
    print("\n🧪 TEST RAPIDE :")
    test_results = collection.query(query_texts=["curry allergène"], n_results=3)
    curry_found = any("curry" in doc.lower() for doc in test_results['documents'][0])
    print(f"Recherche 'curry allergène' : {'✅ TROUVÉ' if curry_found else '❌ NON TROUVÉ'}")
    
    if curry_found:
        print("🎉 SUCCÈS ! Votre RAG devrait maintenant fonctionner !")
    
else:
    print("❌ Aucun chunk créé. Vérifiez vos fichiers PDF.")