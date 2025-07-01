import chromadb
import os
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import gradio as gr

os.environ["CHROMA_ENABLE_TELEMETRY"] = "false"

# Configuration
COLLECTION_NAME = "pizzeria_collection"
EMBEDDING_MODEL = "mxbai-embed-large"
LLM_MODEL = "llama3.2:latest"

print("Initialisation des composants LangChain...")

# Initialisation
ollama_embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
client = chromadb.PersistentClient(path="./chroma_db")

vectorstore = Chroma(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding_function=ollama_embeddings
)

print(f"Collection chargée : {COLLECTION_NAME}")
print(f"Nombre de documents : {vectorstore._collection.count()}")

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 10}  # Plus de résultats
)

llm = ChatOllama(model=LLM_MODEL)

# Prompt amélioré avec plus de contexte
template = """Tu es l'assistant IA de la pizzeria "Bella Napoli". Tu es spécialisé dans les informations sur les allergènes et ingrédients de nos pizzas et produits.

CONTEXTE DOCUMENTAIRE :
{context}

INSTRUCTIONS DÉTAILLÉES :

🍕 POUR LES PIZZAS :
- Nous avons 30+ pizzas : Margherita, Biodélice, Végane, Guiguitte (Jambon/Chorizo/Chèvre/Merguez/Bœuf), Cazimir, Super Regina, Italienne, Cow Boy, Croq Terroir, Découverte, etc.
- Chaque pizza a des allergènes spécifiques listés dans le contexte JSON
- Réponds avec les allergènes exacts de la pizza demandée

🥘 POUR LES INGRÉDIENTS :
- Analyse les tableaux d'allergènes où "P" = présence d'allergène
- Ingrédients disponibles : Curry, Crème fraîche, Mozzarella, Saumon fumé, Chorizo, Jambon, etc.
- Tous les fromages contiennent du "Lait"
- La pâte à pizza contient toujours "Céréales contenant du gluten"

📋 ALLERGÈNES OFFICIELS (14 majeurs UE) :
1. Céréales contenant du gluten (blé, seigle, orge, avoine, épeautre, kamut)
2. Crustacés 3. Œufs 4. Poissons 5. Arachides 6. Soja
7. Lait (y compris lactose) 8. Fruits à coques 9. Céleri 10. Moutarde
11. Graines de Sésame 12. Anhydrides sulfureux et Sulfites 13. Lupin 14. Mollusques

🎯 RÈGLES DE RÉPONSE :
- Sois TRÈS précis sur les allergènes présents
- Si c'est une pizza, donne la liste complète des allergènes
- Si c'est un ingrédient, cherche dans le tableau avec les "P"
- Mentionne toujours la source (JSON pour pizzas, tableau pour ingrédients)
- Ajoute les notes importantes si pertinent (traces possibles, environnement gluten)

⚠️ SÉCURITÉ ALIMENTAIRE :
- Ces informations concernent les allergènes volontairement incorporés
- Traces possibles d'autres allergènes lors de la fabrication
- Pizza "sans gluten" préparée dans environnement contenant du gluten
- En cas de doute grave, recommande de contacter directement la pizzeria

QUESTION : {question}

RÉPONSE (précise, sécurisée et professionnelle) :"""

prompt = ChatPromptTemplate.from_template(template)

def search_ingredient_comprehensive(ingredient_name):
    """Recherche exhaustive d'un ingrédient"""
    try:
        # 1. Recherche vectorielle
        vector_results = vectorstore.similarity_search(ingredient_name, k=10)
        
        # 2. Recherche directe dans tous les documents
        all_docs = vectorstore._collection.get()
        direct_results = []
        
        for doc in all_docs['documents']:
            if ingredient_name.lower() in doc.lower():
                # Extraire les lignes pertinentes
                lines = doc.split('\n')
                for i, line in enumerate(lines):
                    if ingredient_name.lower() in line.lower():
                        # Prendre aussi les lignes autour pour le contexte
                        context_lines = []
                        for j in range(max(0, i-2), min(len(lines), i+3)):
                            context_lines.append(lines[j])
                        direct_results.append('\n'.join(context_lines))
        
        return vector_results, direct_results
        
    except Exception as e:
        print(f"Erreur recherche : {e}")
        return [], []

def answer_question_improved(question):
    """Réponse améliorée avec recherche exhaustive"""
    try:
        # Identifier l'ingrédient dans la question
        keywords = {
            "crème fraîche": ["crème fraîche", "creme fraiche", "crème"],
            "mozzarella": ["mozzarella", "mozza"],
            "curry": ["curry"],
            "sauce tomate": ["sauce tomate", "tomate"],
            "pâte": ["pâte", "pate"],
            "fromage": ["fromage"],
            "bacon": ["bacon"],
            "jambon": ["jambon"],
            "emmental": ["emmental"],
            "gorgonzola": ["gorgonzola"],
            "parmigiano": ["parmigiano", "parmesan"],
            "reblochon": ["reblochon"],
            "raclette": ["raclette"]
        }
        
        ingredient_found = None
        for ingredient, variations in keywords.items():
            if any(var in question.lower() for var in variations):
                ingredient_found = ingredient
                break
        
        if ingredient_found:
            print(f"🔍 Recherche pour : {ingredient_found}")
            
            # Recherche exhaustive
            vector_results, direct_results = search_ingredient_comprehensive(ingredient_found)
            
            print(f"📊 Résultats vectoriels : {len(vector_results)}")
            print(f"📊 Résultats directs : {len(direct_results)}")
            
            # Construire le contexte enrichi
            all_context = []
            
            # Ajouter résultats vectoriels
            for doc in vector_results:
                all_context.append(doc.page_content)
            
            # Ajouter résultats directs
            all_context.extend(direct_results[:5])  # Limiter pour éviter trop de texte
            
            if all_context:
                # Créer prompt enrichi
                enriched_context = "\n\n--- DOCUMENT ---\n".join(all_context)
                
                enriched_prompt = f"""Tu es l'assistant IA de la pizzeria "Bella Napoli".

CONTEXTE COMPLET SUR {ingredient_found.upper()} :
{enriched_context}

QUESTION : {question}

INSTRUCTIONS :
- Cherche spécifiquement "{ingredient_found}" dans le contexte
- Si tu vois un tableau, trouve la ligne avec "{ingredient_found}"
- Note tous les "P" dans les colonnes d'allergènes pour cette ligne
- Liste clairement tous les allergènes présents (marqués "P")

RÉPONSE :"""
                
                try:
                    answer = llm.invoke(enriched_prompt).content
                    return answer
                except Exception as e:
                    print(f"Erreur LLM : {e}")
            
            # Si aucun contexte trouvé, utiliser les résultats bruts
            if direct_results:
                return f"Informations trouvées sur {ingredient_found} :\n\n" + "\n\n".join(direct_results[:3])
        
        # Fallback sur RAG standard
        return rag_chain.invoke(question)
        
    except Exception as e:
        return f"Erreur : {e}"

# Construction de la chaîne RAG standard
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


def chat_with_bot(message, history):
    """Fonction pour l'interface Gradio avec historique - Format messages"""
    if not message.strip():
        return history, history
    
    response = answer_question_improved(message)
    
    # Format moderne pour Gradio : dictionnaires avec 'role' et 'content'
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})
    
    return history, history

def test_search_gradio():
    """Test de recherche pour l'interface Gradio"""
    try:
       
        all_docs = vectorstore._collection.get()
        creme_docs = []
        
        for i, doc in enumerate(all_docs['documents']):
            if "crème fraîche" in doc.lower() or "creme fraiche" in doc.lower():
                creme_docs.append((i, doc))
        
        result = f"Documents contenant 'crème fraîche' : {len(creme_docs)}\n\n"
        
        for i, (doc_idx, doc) in enumerate(creme_docs[:2]):
            result += f"📄 Document {doc_idx} :\n"
            lines = doc.split('\n')
            for line in lines:
                if "crème" in line.lower() and len(line.strip()) > 5:
                    result += f"   {line.strip()}\n"
            result += "\n"
                    
        return result
        
    except Exception as e:
        return f"Erreur test : {e}"

# ✅ INTERFACE GRADIO INTÉGRÉE

def create_gradio_interface():
    """Interface Gradio complète"""
    with gr.Blocks(
        title="🍕 Assistant Pizzeria Bella Napoli", 
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        """
    ) as demo:
        
        gr.Markdown("""
        # 🍕 Assistant Pizzeria Bella Napoli
        ### Votre assistant spécialisé en allergènes et ingrédients
        
        Posez vos questions sur les allergènes des pizzas, sauces et ingrédients !
        """)
        
        with gr.Tabs():
            with gr.TabItem("💬 Chat Assistant"):
                chatbot = gr.Chatbot(
                    label="Conversation avec l'assistant",
                    height=500,
                    placeholder="Posez votre question sur les allergènes...",
                    show_copy_button=True,
                    type="messages"  
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        label="Votre question",
                        placeholder="Ex: La crème fraîche contient quels allergènes ?",
                        scale=4,
                        lines=2
                    )
                    with gr.Column(scale=1):
                        send_btn = gr.Button("📤 Envoyer", variant="primary")
                        clear_btn = gr.Button("🗑️ Effacer", variant="secondary")
                
                gr.Examples(
                    examples=[
                        "La crème fraîche contient quels allergènes ?",
                        "Quels sont les allergènes de la mozzarella ?",
                        "La sauce curry a-t-elle du gluten ?",
                        "Y a-t-il du lait dans la pâte à pizza ?",
                        "Quels fromages sont sans lactose ?",
                        "Le jambon contient-il des sulfites ?",
                        "Y a-t-il des fruits à coque dans les sauces ?"
                    ],
                    inputs=msg,
                    label="Exemples de questions"
                )
                
                msg.submit(chat_with_bot, [msg, chatbot], [chatbot, chatbot])
                send_btn.click(chat_with_bot, [msg, chatbot], [chatbot, chatbot])
                clear_btn.click(lambda: [], outputs=[chatbot, chatbot])
            
                msg.submit(lambda: "", outputs=msg)
                send_btn.click(lambda: "", outputs=msg)
            
            with gr.TabItem("🔧 Test & Debug"):
                gr.Markdown("### Outils de test et diagnostic")
                
                with gr.Row():
                    test_btn = gr.Button("🧪 Tester recherche crème fraîche", variant="secondary")
                    info_btn = gr.Button("ℹ️ Infos système", variant="secondary")
                
                test_output = gr.Textbox(
                    label="Résultats des tests",
                    lines=15,
                    interactive=False
                )
                
                def get_system_info():
                    """Obtenir les informations système"""
                    try:
                        doc_count = vectorstore._collection.count()
                        info = f"""
📊 INFORMATIONS SYSTÈME
=======================

✅ Collection : {COLLECTION_NAME}
✅ Nombre de documents : {doc_count}
✅ Modèle d'embedding : {EMBEDDING_MODEL}
✅ Modèle LLM : {LLM_MODEL}
✅ Recherche : Top {retriever.search_kwargs.get('k', 10)} résultats

🔍 MOTS-CLÉS RECONNUS :
- crème fraîche, mozzarella, curry
- sauce tomate, pâte, fromage
- bacon, jambon, emmental
- gorgonzola, parmigiano, reblochon, raclette

💡 CONSEILS D'UTILISATION :
- Soyez spécifique dans vos questions
- Mentionnez l'ingrédient exact
- Utilisez les exemples pour vous guider
                        """
                        return info
                    except Exception as e:
                        return f"Erreur : {e}"
                
                test_btn.click(test_search_gradio, outputs=test_output)
                info_btn.click(get_system_info, outputs=test_output)
        
        # Instructions d'utilisation
        gr.Markdown("""
        ---
        **💡 Instructions d'utilisation :**
        
        1. **Chat Principal** : Posez vos questions sur les allergènes des ingrédients
        2. **Test & Debug** : Vérifiez le fonctionnement du système
        3. **Exemples** : Cliquez sur les exemples pour voir le format de questions
        
        **🎯 Types de questions supportées :**
        - "Quels allergènes contient [ingrédient] ?"
        - "[Ingrédient] a-t-il du [allergène] ?"
        - "Y a-t-il du [allergène] dans [ingrédient] ?"
        
        **⚠️ Allergènes surveillés :** Gluten, Lait, Œufs, Fruits à coque, Soja, etc.
        """)
    
    return demo

# Point d'entrée principal
if __name__ == "__main__":
    print("\n🚀 Assistant Pizzeria Bella Napoli")
    print("=" * 50)
    
    # Vérification rapide du système
    try:
        doc_count = vectorstore._collection.count()
        if doc_count > 0:
            print(f"✅ Base de données chargée : {doc_count} documents")
            print("🚀 Lancement de l'interface Gradio...")
            
            # Créer et lancer l'interface Gradio
            demo = create_gradio_interface()
            demo.launch(
                share=False,
                server_name="127.0.0.1",
                server_port=7860,
                show_error=True
                # show_tips supprimé car non supporté
            )
        else:
            print("❌ Aucun document dans la base de données")
            print("➡️ Créez d'abord la base avec le script de création")
            
    except Exception as e:
        print(f"❌ Erreur : {e}")
        print("➡️ Vérifiez que la base de données existe")