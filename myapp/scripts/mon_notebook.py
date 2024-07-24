import shap
import joblib
import os
import pandas as pd
import re
import torch
import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel


#CHARGER MODELES 

#REMPLACE CHEMIN_MODELES et CHEMIN_BINARIZER par les chemins sur ton pc cest dans le dossier que je vais tenvoyer 

def load_models(prefix, num_models, path):
    models = []
    for i in range(num_models):
        model_path = os.path.join(path, f'{prefix}_model_{i}.pkl')
        model = joblib.load(model_path)
        models.append(model)
    return models

# Chemin où les modèles sont sauvegardés
chemin_modeles = r'C:\Users\ngakoutatsing.franck\Downloads\model_subclass'  # Utiliser un raw string pour éviter les problèmes d'escape

# Chemin où les objets MultiLabelBinarizer sont sauvegardés
chemin_binarizer = r'C:\Users\ngakoutatsing.franck\Downloads\model_subclass\Binarizer'  # Utiliser un raw string pour éviter les problèmes d'escape


# Nombre de modèles pour chaque niveau hiérarchique
num_subclass_models = 638 # Remplacer par le nombre réel de modèles sauvegardés pour les sous-classes
# Charger les modèles
models_subclass = load_models('subclass', num_subclass_models, chemin_modeles)
# Charger les objets MultiLabelBinarizer sauvegardés)
mlb_subclass_path = os.path.join(chemin_binarizer, 'mlb_subclass.pkl')
mlb_subclass = joblib.load(mlb_subclass_path) 

#FONCTIONS

# Function to pad or truncate sequences
def pad_or_truncate(sequence, target_length):
    if len(sequence) > target_length:
        # Truncate the sequence
        return sequence[:target_length]
    elif len(sequence) < target_length:
        # Pad the sequence with zeros
        return sequence + [0.0] * (target_length - len(sequence))
    else:
        return sequence
    
#Le shap.TreeExplainer est optimisé pour les modèles basés sur des arbres de décision

def calculate_shap_values_for_each_model(models, X_train, X_test):
    shap_values_list = []
    for model in models:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        shap_values_list.append(shap_values)
    return shap_values_list

def retourneNShapEleve(shap_values_observation, n, longueur_de_linput):
    # Obtenir les indices des valeurs SHAP triées par valeur absolue en ordre décroissant
    sorted_indices = np.argsort(np.abs(shap_values_observation))[::-1]
    
    # Filtrer les indices pour qu'ils soient inférieurs à longueur_de_linput
    valid_indices = [idx for idx in sorted_indices if idx < longueur_de_linput]
    
    # Prendre les n premiers indices valides
    top_n_indices = valid_indices[:n]

    # Retourner les indices comme une liste
    return top_n_indices

def compter_mots(texte):
    # Diviser le texte en mots en utilisant les espaces comme séparateurs
    mots = texte.split()
    # Retourner le nombre de mots dans le texte
    return len(mots)

def calculate_shap_values_for_each_model_one_newInput(models, X_newInput):
    shap_values_list = []
    for model in models:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_newInput)
        shap_values_list.append(shap_values)
    return shap_values_list

#X_newInput doit etre tensor une ligne de 512 elements
def calculer_NShapEleve_newInput_parModel(models_section, X_newInput, profondeur , text ) :
    list_profondeur_perModel =[]
    shap_values_section_new_input= calculate_shap_values_for_each_model_one_newInput(models_section, X_newInput)
    for shap_values_NewInput in shap_values_section_new_input:
        shap_values_oberservation_NewInput = shap_values_NewInput[0]
        list_for_that_model = retourneNShapEleve(shap_values_oberservation_NewInput , profondeur , compter_mots(text) )
        list_profondeur_perModel.append(list_for_that_model)
    return list_profondeur_perModel

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
pretrain_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
# Utiliser la GPU si disponible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pretrain_model.to(device)

def get_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Déplacer les inputs sur le même appareil que le modèle
    with torch.no_grad():
        outputs = pretrain_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().squeeze().numpy()

def ndarray_to_float_list(arr):
    # Vérifier que l'entrée est bien une ndarray
    if not isinstance(arr, np.ndarray):
        raise TypeError("L'entrée doit être une instance de numpy.ndarray.")

    # Vérifier que le dtype de la ndarray est float32
    if arr.dtype != np.float32:
        raise ValueError("La ndarray doit contenir des valeurs de type numpy.float32.")

    # Convertir la ndarray en une liste de float
    float_list = arr.astype(float).tolist()

    return float_list

def retrouver_n_indices_plus_importants( text , profondeur) :
    textEmb= get_embeddings(text)
    textEmbFloat= ndarray_to_float_list(textEmb)
    textEmbFloat_padded = [pad_or_truncate(textEmbFloat, 512)]
    textEmbFloat_tensor = torch.tensor(textEmbFloat_padded)
    list_n_indices_importants_par_models = calculer_NShapEleve_newInput_parModel(models_subclass, textEmbFloat_tensor, profondeur , text  )
    return list_n_indices_importants_par_models

def trouver_indices_de_1(tableau):
    indices = np.where(tableau[0] == 1)
    return indices

def get_embeddings(text):
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        inputs = {key: value.to(device) for key, value in inputs.items()}  # Déplacer les inputs sur le même appareil que le modèle
        with torch.no_grad():
            outputs = pretrain_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).cpu().squeeze().numpy()

def predict(models, X):
        y_preds = np.zeros((X.shape[0], len(models)))
        for i, model in enumerate(models):
            y_preds[:, i] = model.predict(X)
        return y_preds


def prediction_krakow (new_patents) :
    from transformers import DistilBertTokenizer, DistilBertModel
    #Charger le tokenizer et le modèle DistilBERT pré-entraîné
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    pretrain_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pretrain_model.to(device)
   # Obtenir les embeddings du nouveau brevet et les convertir en tensor
    embeddings = get_embeddings(new_patents)
    X_new_tensor = torch.tensor(embeddings).unsqueeze(0)
    # Tronquer à la taille [3, 512]
    X_new_tensor_truncated = X_new_tensor[:, :512]
    # Faire des prédictions
    #y_pred_section = predict(models_section, X_new_tensor_truncated)
    #y_pred_class = predict(models_class, X_new_tensor_truncated)
    y_pred_subclass = predict(models_subclass, X_new_tensor_truncated)
    return  y_pred_subclass 

def return_une_array_avec_les_modeles_subclass_1_pour_les_modeles_choisis (text) :
    prediction = prediction_krakow(text)
    return prediction

def pour_chaque_model_les_pronfondeur_premiers_mots_importants (text , prof) : 
  resultat=retrouver_n_indices_plus_importants(text , prof )
  return resultat

def trouver_indices_des_modeles_a_considerer (prediction) :
    indices_des_models_a_considerer_Inter = trouver_indices_de_1(prediction)
    # Accédez au tableau numpy à l'intérieur du tuple
    array_inside_tuple = indices_des_models_a_considerer_Inter[0]
    # Convertir le tableau numpy en une liste Python
    indices_des_models_a_considerer = array_inside_tuple.tolist()
    return indices_des_models_a_considerer 

def les_profondeur_plus_importants_index_des_mots_pour_les_modeles_qui_ont_fais_la_prediction (indices_des_models_a_considerer , resultat) : 
    resultat_final = []
    # Parcourir les indices et extraire les éléments correspondants de resultat
    for idx in indices_des_models_a_considerer:
        resultat_final.append(resultat[idx])
    return resultat_final 

def a_partir_du_text_et_prof_retourne_resultat_final(text , prof ) : 
    prediction = prediction_krakow(text)
    resultat=pour_chaque_model_les_pronfondeur_premiers_mots_importants(text , prof )
    indices_des_models_a_considerer = trouver_indices_des_modeles_a_considerer (prediction) 
    resultat_final = les_profondeur_plus_importants_index_des_mots_pour_les_modeles_qui_ont_fais_la_prediction (indices_des_models_a_considerer,resultat) 
    return resultat_final 

def retourne_le_cpc_subclass_apres_prediction (y_pred_subclass) :
   y_pred_subclass_inversed = mlb_subclass.inverse_transform(y_pred_subclass.astype(int))
   return y_pred_subclass_inversed

def le_cpc_predit_en_subclass(text) : 
    prediction = return_une_array_avec_les_modeles_subclass_1_pour_les_modeles_choisis (text)
    cpc_predit=retourne_le_cpc_subclass_apres_prediction(prediction)
    return cpc_predit 

#transformer le texte brut en liste de strings 
import re
from bs4 import BeautifulSoup


def remove_tags_delete_num(text):
       # Utiliser le parseur HTML de BeautifulSoup
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text()

    # Convertir en minuscules
    text = text.lower()

    # Supprimer les caractères spéciaux et les ponctuations inutiles
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

    # Normaliser les espaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Remplacer les nombres par un token spécial
    text = re.sub(r'\b\d+\b', '<NUM>', text)
    # Utilisation de regex pour trouver et remplacer toutes les occurrences de <NUM>
    text = re.sub(r'<NUM>', '', text)
    # Supprimer les espaces supplémentaires créés par la suppression de <NUM>
    text = ' '.join(text.split())
    return text

# Fonction pour mapper les tokens du texte aux embeddings pour les indexes spécifiques
def map_tokens_to_embeddings(text, embeddings, indexes):
    # Tokeniser le texte pour obtenir les tokens
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    input_ids = inputs['input_ids'].squeeze().tolist()  # Récupérer les IDs des tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids)  # Convertir les IDs en tokens
    
    # Créer un dictionnaire pour stocker les embeddings des tokens spécifiques
    token_embedding_map = {}
    
    # Récupérer les embeddings pour les indexes spécifiques
    for i in indexes:
        token = tokens[i]  # Récupérer le token correspondant à l'index
        embedding = embeddings[i]  # Récupérer l'embedding correspondant à l'index
        token_embedding_map[token] = embedding
    
    return token_embedding_map

def fonction_pour_les_mots_ayant_influer_le_plus_la_prediction(text, profondeur):
    text = remove_tags_delete_num(text)
    liste_de_profondeur_indices_justidfiant_chaque_partie_cpc = a_partir_du_text_et_prof_retourne_resultat_final(text, profondeur)
    cpc_predit = le_cpc_predit_en_subclass(text)
    embeddings = get_embeddings(text)
    
    results = []
    results.append({"cpc_predit": cpc_predit})
    
    for index in range(len(cpc_predit[0])):
        result = {
            "cpc": cpc_predit[0][index],
            "mots_influents": []
        }
        token_embedding_map = map_tokens_to_embeddings(text, embeddings, liste_de_profondeur_indices_justidfiant_chaque_partie_cpc[index])
        for token, embedding in token_embedding_map.items():
            result["mots_influents"].append(token)
        results.append(result)
    
    return results
