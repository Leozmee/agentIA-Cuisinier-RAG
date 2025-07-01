import os
import chromadb
from chromadb.utils import embedding_functions
import pdfplumber
import time
import json

os.environ["CHROMA_ENABLE_TELEMETRY"] = "false"

def lister_fichiers_pdf_recursivement(repertoire):
    """Lister tous les fichiers PDF dans un répertoire et ses sous-répertoires"""
    pdf_files = []
    print(f"🔍 Recherche de fichiers PDF dans : {repertoire}")
    
    for racine, repertoires, fichiers in os.walk(repertoire):
        for fichier in fichiers:
            if fichier.lower().endswith('.pdf'):
                chemin_complet = os.path.join(racine, fichier)
                pdf_files.append(chemin_complet)
                print(f"  📄 Trouvé : {fichier}")
    
    print(f"✅ {len(pdf_files)} fichier(s) PDF trouvé(s)")
    return pdf_files

def load_json_allergens():
    """Charger le fichier JSON des allergènes et le convertir en chunks"""
    json_path = "/home/utilisateur/proget-agentIA-RAG/RAG/Allergenes-Pizza-Rhuys.json"
    
    if not os.path.exists(json_path):
        print(f"⚠️ Fichier JSON non trouvé : {json_path}")
        return []
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"📋 Chargement du fichier JSON des allergènes...")
        chunks = []
        
        # Chunk 1 : Liste générale des allergènes
        allergen_chunk = "[Source: allergenes_pizzas.json]\n"
        allergen_chunk += "=== LISTE DES ALLERGÈNES MAJEURS ===\n"
        for allergen, details in data.get('allergens_reference', {}).items():
            allergen_chunk += f"- {allergen}: {details}\n"
        chunks.append(allergen_chunk)
        
        # Chunk 2 : Notes importantes
        notes_chunk = "[Source: allergenes_pizzas.json]\n"
        notes_chunk += "=== NOTES IMPORTANTES SUR LES ALLERGÈNES ===\n"
        for key, note in data.get('notes_importantes', {}).items():
            notes_chunk += f"{key.replace('_', ' ').upper()}: {note}\n"
        chunks.append(notes_chunk)
        
        # Chunks individuels pour chaque pizza
        pizzas_data = data.get('pizzas', {})
        for pizza_name, pizza_info in pizzas_data.items():
            pizza_chunk = "[Source: allergenes_pizzas.json]\n"
            pizza_chunk += f"=== PIZZA: {pizza_name.upper()} ===\n"
            pizza_chunk += f"Nom de la pizza: {pizza_name}\n"
            pizza_chunk += f"Description: {pizza_info.get('description', '')}\n"
            
            allergenes = pizza_info.get('allergenes', [])
            if allergenes:
                pizza_chunk += f"Allergènes présents: {', '.join(allergenes)}\n"
                pizza_chunk += f"Nombre d'allergènes: {len(allergenes)}\n"
            else:
                pizza_chunk += "Allergènes présents: Aucun allergène majeur déclaré\n"
            
            # Ajouter des variations pour améliorer la recherche
            pizza_chunk += f"\nRecherche alternative: Pizza {pizza_name} allergènes\n"
            pizza_chunk += f"Question type: Quels allergènes contient la pizza {pizza_name}?\n"
            pizza_chunk += f"Question type: La pizza {pizza_name} contient-elle du gluten/lait/œufs?\n"
            
            if allergenes:
                pizza_chunk += f"Réponse: La pizza {pizza_name} contient les allergènes suivants: {', '.join(allergenes)}\n"
            else:
                pizza_chunk += f"Réponse: La pizza {pizza_name} ne contient aucun allergène majeur déclaré\n"
            
            chunks.append(pizza_chunk)
        
        # Chunk pour recherches rapides
        recherche_chunk = "[Source: allergenes_pizzas.json]\n"
        recherche_chunk += "=== RECHERCHES RAPIDES ===\n"
        recherches = data.get('recherche_facile', {})
        for category, pizzas in recherches.items():
            recherche_chunk += f"{category.replace('_', ' ').upper()}: {', '.join(pizzas)}\n"
        chunks.append(recherche_chunk)
        
        print(f"  ✅ JSON chargé : {len(chunks)} chunks créés")
        print(f"  📊 {len(pizzas_data)} pizzas dans la base JSON")
        
        return chunks
        
    except Exception as e:
        print(f"❌ Erreur lors du chargement du JSON : {e}")
        return []

# Configuration
PDF_FILES = lister_fichiers_pdf_recursivement('/home/utilisateur/proget-agentIA-RAG/RAG')
COLLECTION_NAME = "pizzeria_collection"
EMBEDDING_MODEL = "mxbai-embed-large"
BATCH_SIZE = 20

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
        
        # Détection d'éléments intéressants
        detected_elements = []
        text_lower = text.lower()
        
        keywords_to_check = [
            'curry', 'mozzarella', 'allergène', 'gluten', 'lait', 
            'sauce', 'fromage', 'jambon', 'chorizo', 'saumon'
        ]
        
        for keyword in keywords_to_check:
            if keyword in text_lower:
                detected_elements.append(keyword)
        
        if detected_elements:
            print(f"  🎯 Éléments détectés : {', '.join(detected_elements[:5])}")
            if len(detected_elements) > 5:
                print(f"      ... et {len(detected_elements) - 5} autres")
        
        # Chunks
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
            time.sleep(1)
        except Exception as e:
            print(f"  ❌ Erreur lot {i//batch_size + 1}: {e}")
            for j, (doc, doc_id) in enumerate(zip(batch_chunks, batch_ids)):
                try:
                    collection.add(documents=[doc], ids=[doc_id])
                    print(f"    ✅ Chunk {doc_id} ajouté individuellement")
                except Exception as chunk_error:
                    print(f"    ❌ Échec chunk {doc_id}: {chunk_error}")

# Vérification des fichiers trouvés
if not PDF_FILES:
    print("⚠️ Aucun fichier PDF trouvé dans le répertoire spécifié")
    print("➡️ Continuons avec le JSON uniquement...")

# Initialisation
print("\n🚀 CRÉATION BASE DE DONNÉES PIZZERIA (PDF + JSON)")
print("=" * 55)

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

# ✅ AJOUT DU JSON D'ABORD
print("\n📋 TRAITEMENT DU FICHIER JSON...")
json_chunks = load_json_allergens()
all_chunks = json_chunks.copy()

# Traitement des PDFs
files_processed = 0
files_failed = 0

if PDF_FILES:
    print(f"\n📄 TRAITEMENT DE {len(PDF_FILES)} FICHIERS PDF...")
    
    for pdf_file in PDF_FILES:
        if os.path.exists(pdf_file):
            pdf_chunks = load_and_chunk_pdf_improved(pdf_file)
            
            if pdf_chunks:
                all_chunks.extend(pdf_chunks)
                files_processed += 1
                print(f"✅ {os.path.basename(pdf_file)} traité avec succès")
            else:
                files_failed += 1
                print(f"❌ Échec du traitement de {os.path.basename(pdf_file)}")
        else:
            files_failed += 1
            print(f"❌ Fichier non trouvé : {pdf_file}")

print(f"\n--- Résumé du traitement ---")
print(f"📋 Chunks JSON : {len(json_chunks)}")
print(f"📄 Fichiers PDF traités : {files_processed}")
print(f"❌ Fichiers PDF en échec : {files_failed}")
print(f"📦 Total de chunks créés : {len(all_chunks)}")

if all_chunks:
    # Ajout par lots
    add_chunks_in_batches(collection, all_chunks, BATCH_SIZE)
    
    print(f"\n--- Base de données vectorielle créée avec succès ! ---")
    print(f"Nombre de documents stockés : {collection.count()}")
    
    # ✅ Tests spécialisés pour pizzas
    print("\n🧪 TESTS DE VALIDATION SPÉCIALISÉS :")
    test_queries = [
        ("pizza margherita allergène", "margherita"),
        ("guiguitte jambon allergène", "jambon"),
        ("pizza végane gluten", "végane"),
        ("biodélice œufs", "biodélice"),
        ("italienne poisson", "italienne"),
        ("découverte crustacés", "découverte"),
        ("curry allergène", "curry"),
        ("mozzarella lait", "mozzarella")
    ]
    
    for query, keyword in test_queries:
        try:
            test_results = collection.query(query_texts=[query], n_results=3)
            found = any(keyword in doc.lower() for doc in test_results['documents'][0])
            status = "✅ TROUVÉ" if found else "❌ NON TROUVÉ"
            print(f"  {query:25} : {status}")
        except Exception as e:
            print(f"  {query:25} : ❌ ERREUR - {e}")
    
    print("\n🎉 SUCCÈS ! Base de données JSON + PDF créée !")
    print("🍕 Votre assistant peut maintenant répondre aux questions sur les pizzas")
    print("🔍 Testez avec des questions comme : 'Quels allergènes contient la pizza Margherita ?'")
    
else:
    print("❌ Aucun chunk créé.")
    print("💡 Vérifiez que le fichier allergenes_pizzas.json existe dans le dossier RAG")