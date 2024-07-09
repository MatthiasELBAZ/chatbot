from flask import Flask, request, jsonify
# from flask_jwt_extended import JWTManager, create_access_token, jwt_required
import os
import logging
import json
import ast
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Charger les variables d'environnement depuis un fichier .env
# load_dotenv()

# Importez les classes et fonctions nécessaires de votre fichier
from Self_Coach_It_Classes_Main_flask_2 import (
    SelectPrompt, UserSessionStoreHistory, UserDirectoryLoader,
    ChromaFormulaireRtriever, PineconeTimeWeightedRetriever, UserProfile, UpdateStore,
    initialization, CallBacker, RetrievalDocumentChainMemory, run_chat, update_and_store_after_exit, NewJournal
)

app = Flask(__name__)
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'votre_cle_secrete')
# jwt = JWTManager(app)

# Global variables to hold initialized objects per user and session
sessions = {}

@app.route('/initialize', methods=['POST'])
def initialize():
    os.environ["OPENAI_API_KEY"] = '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'
    os.environ["PINECONE_API_KEY"] = '46654ee1-d1ab-4e39-91e0-20917a15cb0b'
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "conversation_1"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_API_KEY"] = "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"
    try:
        data = request.json
        user_id = data.get('user_id')
        session_id = data.get('session_id')
        user_directory = data.get('user_directory')
        user_starting_date = data.get('user_starting_date')
        namespace = data.get('namespace')
        index_name = data.get('index_name')
        coach_name = data.get('coach_name')
        llm_name_model = data.get('llm_name_model')
        temperature = data.get('temperature')
        decay_rate = data.get('decay_rate')
        k = data.get('k')
        buffer_num_ongo_messages = data.get('buffer_num_ongo_messages')
        system_prompts = data.get('system_prompts')
        user_prompts = data.get('user_prompts')
        summary_prompts = data.get('summary_prompts')
        journal_file = data.get('journal_file')
        formulaire_file = data.get('formulaire_file')
        history_file = data.get('history_file')
        profile_file = data.get('profile_file')

        if not all([
            user_id, session_id, user_directory, user_starting_date, namespace, index_name, 
            coach_name, llm_name_model, temperature, decay_rate, k, buffer_num_ongo_messages, 
            system_prompts, user_prompts, summary_prompts, journal_file, formulaire_file, history_file, profile_file]):
            return jsonify({"status": "Error", "message": "All parameters are required."}), 400

        # Call the initialization method with all required parameters
        session_objects = initialization(
            user_directory, user_id, session_id, user_starting_date, namespace, index_name,
            coach_name, llm_name_model, temperature, decay_rate, k, buffer_num_ongo_messages, 
            system_prompts, user_prompts, summary_prompts, journal_file, formulaire_file, history_file, profile_file,
        )

        # Store session data
        sessions[(user_id, session_id)] = session_objects
        sessions[(user_id, session_id)]["session_data"] = ""  # Initialize session data

        # Log session storage
        logging.info(f"Session initialized for user_id: {user_id}, session_id: {session_id}")

        return jsonify({"status": "Initialized", "message": "Initialization complete."})
    except Exception as e:
        logging.error(f"Error during initialization: {str(e)}")
        return jsonify({"status": "Error", "message": "An error occurred during initialization. Please check the server logs for more details."}), 500

@app.route('/initialize_chain', methods=['POST'])
def initialize_chain():
    try:
        data = request.json
        user_id = data.get('user_id')
        session_id = data.get('session_id')
        instructions = data.get('instructions')

        if not user_id or not session_id or not instructions:
            return jsonify({"status": "Error", "message": "user_id, session_id, and instructions are required."}), 400

        session = sessions.get((user_id, session_id))
        if not session:
            return jsonify({"status": "Error", "message": "Session not found."}), 404

        logging.debug(f"User Profile {session['User_Profile']}")

        user_profile = session['User_Profile']

        # Initialize chain with the given instructions
        llm = instructions['llm']

        logging.debug(f"%%%%%%%%%%%%%%%%%%%%%%%")
        logging.debug(f"prompts {json.loads(json.dumps(session['prompts']))}")
        logging.debug(f"%%%%%%%%%%%%%%%%%%%%%%%")
        # exit(0)

        # prompts = json.loads(json.dumps(session['prompts']))
        #
        # user_profile.make_summarizer(llm, prompts)

        return jsonify({"status": "Chain Initialized", "message": "Chain initialization complete."})
    except Exception as e:
        logging.error(f"Error during chain initialization: {str(e)}")
        return jsonify({"status": "Error", "message": "An error occurred during chain initialization. Please check the server logs for more details."}), 500

@app.route('/initialize_profile', methods=['POST'])
def initialize_profile():
    try:
        data = request.json
        user_id = data.get('user_id')
        session_id = data.get('session_id')


        if not all([user_id]):
            return jsonify({"status": "Error", "message": "user_id is required."}), 400

        session = sessions.get((user_id, session_id))
        if not session:
            return jsonify({"status": "Error", "message": "Session not found."}), 404
        
        user_directory = session['user_directory']
        formulaire_file = session['formulaire_file']
        prompts = session['prompts']
        llm = session['llm']
        user_profile_csv_path = user_directory + session['profile_file']

        # load formulaire docs
        User_Loader = UserDirectoryLoader(user_directory)
        formulaire_docs = User_Loader.load_csv_from_directory(formulaire_file, None)

        # make user summary
        User_Profile = UserProfile()
        User_Profile.make_summarizer(llm, prompts)
        User_Profile.get_user_summary(formulaire_docs, prompts)

        # store user summary - it must be the first session_id 
        Update_Store = UpdateStore()
        Update_Store.update_store_user_profile_csv(user_profile_csv_path, User_Profile.user_summary, session_id)

        # Update user profile
        session['User_Profile'] = User_Profile

        return jsonify({"status": "Profile Initialized from Formuliare", "message": "User profile initialized successfully.", "profile": User_Profile.user_summary})
    except Exception as e:
        logging.error(f"Error during profile update: {str(e)}")
        return jsonify({"status": "Error", "message": "An error occurred during profile update. Please check the server logs for more details."}), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "Error", "message": "Invalid JSON data."}), 400

        user_id = data.get('user_id')
        session_id = data.get('session_id')
        new_prompt = data.get('new_prompt')
        llm_name_model = data.get('llm_name_model')

        logging.debug(f" information ICI {new_prompt}")


        if not all([user_id, session_id, new_prompt]):
            return jsonify({"status": "Error", "message": "user_id, session_id, and new_prompt are required."}), 400

        session = sessions.get((user_id, session_id))
        if not session or not isinstance(session, dict):
            return jsonify({"status": "Error", "message": "Session not found."}), 404

        session_data = session.get("session_data", "")
        chat_dict = run_chat(new_prompt, session, modify_llm_name_model=llm_name_model)  # Assurez-vous que run_chat gère correctement les inputs.

        response = chat_dict['result']
        session["session_data"] += f"\n{new_prompt}\n{chat_dict}"

        logging.debug("%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        logging.debug(f"{response['answer'].content}")
        logging.debug("%%%%%%%%%%%%%%%%%%%%%%%%%%%")

        return jsonify({"response":response['answer'].content})

    except Exception as e:
        logging.error(f"Error during chat processing: {str(e)}")
        return jsonify({"status": "Error", "message": "An error occurred during chat processing. Please check the server logs for more details."}), 500

@app.route('/exit', methods=['POST'])
def exit():
    try:
        data = request.json
        user_id = data.get('user_id')
        session_id = data.get('session_id')

        if not all([user_id, session_id]):
            return jsonify({"status": "Error", "message": "user_id and session_id are required."}), 400

        session = sessions.get((user_id, session_id))
        if not session:
            return jsonify({"status": "Error", "message": "Session not found."}), 404

        user_directory = session['user_directory']
        journal_csv_path = user_directory + session['journal_file']
        chat_history_csv_path = user_directory + session['history_file']
        user_profile_csv_path = user_directory + session['profile_file']
        Pinecone_Time_Retiever_Journal = session['Pinecone_Time_Retiever_Journal']
        llm = session['llm']
        prompts = session['prompts']
        store_history = session['store_history']
        namespace = session['namespace']
        coach_name = session['coach_name']
        User_Profile = session['User_Profile']


        # get new journal for full messages
        New_Journal = NewJournal()
        New_Journal.get_new_journal(llm, prompts, user_id, session_id, store_history)
        new_journal = New_Journal.new_journal
        print("New Journal made.")
        print()
        # update store
        Update_Store = UpdateStore()
        Update_Store.update_store_journal_csv(journal_csv_path, new_journal, session_id)
        Update_Store.update_store_chat_history_csv(chat_history_csv_path, store_history, user_id, session_id)
        print("CSV Store updated.")
        print()
        #update pinecone index
        Update_Store.update_journal_pinecone_index(Pinecone_Time_Retiever_Journal, new_journal, namespace, coach_name)
        print("Pinecone index updated.")

        # update user profile
        if User_Profile.user_profile==None:
            profile = User_Profile.user_summary
        else:
            profile = User_Profile.user_profile
        User_Profile.get_user_profile(llm, profile, new_journal, prompts)
        # update user profile csv
        Update_Store.update_store_user_profile_csv(user_profile_csv_path, profile, session_id)
        print("User Profile updated.")

        session['User_Profile'] = User_Profile
        session['Update_Store'] = Update_Store
        session['New_Journal'] = New_Journal

        # empty the session from the global sessions dictionary
        sessions.pop((user_id, session_id), None)

        return jsonify({"status": "Session Exited", "message": "Session exited successfully.", "journal": new_journal, "profile": profile, "session":sessions})
    except Exception as e:
        logging.error(f"Error during exit: {str(e)}")
        return jsonify({"status": "Error", "message": "An error occurred during exit. Please check the server logs for more details."}), 500

@app.route('/save', methods=['POST'])
def save():
    try:
        data = request.json
        user_id = data.get('user_id')
        session_id = data.get('session_id')

        if not all([user_id, session_id, new_prompt]):
            return jsonify({"status": "Error", "message": "user_id, session_id, and new_prompt are required."}), 400

        session = sessions.get((user_id, session_id))
        if not session:
            return jsonify({"status": "Error", "message": "Session not found."}), 404

        user_directory = session['user_directory']
        journal_csv_path = user_directory + session['journal_file']
        chat_history_csv_path = user_directory + session['history_file']
        user_profile_csv_path = user_directory + session['profile_file']
        Pinecone_Time_Retiever_Journal = session['Pinecone_Time_Retiever_Journal']
        llm = session['llm']
        prompts = session['prompts']
        store_history = session['store_history']
        namespace = session['namespace']
        coach_name = session['coach_name']
        User_Profile = session['User_Profile']


        # get new journal for full messages
        New_Journal = NewJournal()
        New_Journal.get_new_journal(llm, prompts, user_id, session_id, store_history)
        new_journal = New_Journal.new_journal
        print("New Journal made.")
        print()
        # update store
        Update_Store = UpdateStore()
        Update_Store.update_store_journal_csv(journal_csv_path, new_journal, session_id)
        Update_Store.update_store_chat_history_csv(chat_history_csv_path, store_history, user_id, session_id)
        print("CSV Store updated.")
        print()
        #update pinecone index
        Update_Store.update_journal_pinecone_index(Pinecone_Time_Retiever_Journal, new_journal, namespace, coach_name)
        print("Pinecone index updated.")

        # update user profile
        if User_Profile.user_profile==None:
            profile = User_Profile.user_summary
        else:
            profile = User_Profile.user_profile
        User_Profile.get_user_profile(llm, profile, new_journal, prompts)
        # update user profile csv
        Update_Store.update_store_user_profile_csv(user_profile_csv_path, profile, session_id)
        print("User Profile updated.")

        session['User_Profile'] = User_Profile
        session['Update_Store'] = Update_Store
        session['New_Journal'] = New_Journal


        return jsonify({"status": "Session Saved", "message": "Session saved successfully.", "journal": new_journal, "profile": profile,})
    except Exception as e:
        logging.error(f"Error during save processing: {str(e)}")
        return jsonify({"status": "Error", "message": "An error occurred during save processing. Please check the server logs for more details."}), 500

@app.errorhandler(400)
def bad_request(error):
    return jsonify({"status": "Error", "message": "Bad request. Check your input parameters."}), 400

@app.errorhandler(404)
def not_found(error):
    return jsonify({"status": "Error", "message": "Resource not found."}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"status": "Error", "message": "Internal server error."}), 500

@app.route('/login', methods=['POST'])
def login():
    username = request.json.get('username')
    password = request.json.get('password')
    # Vérifiez les identifiants de l'utilisateur ici
    if username == 'admin' and password == 'password':  # Exemple simple
        access_token = create_access_token(identity=username)
        return jsonify(access_token=access_token)
    else:
        return jsonify({"status": "Error", "message": "Invalid credentials"}), 401


@app.route('/get_api', methods=['POST'])
def get_api():
    try:
        api = os.getenv("OPENAI_API_KEY")
        return jsonify({"status": "Success", "message": "API is working.", "api_key": api})
    except Exception as e:
        return jsonify({"status": "Error", "message": "An error occurred during API retrieval."}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)