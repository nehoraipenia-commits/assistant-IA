import json
import random
import re
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. FONCTIONS DE TRAITEMENT DU LANGAGE (NLP)
# ==========================================

def tokenize(phrase):
    """
    Sépare une phrase en un tableau de mots (minuscules, sans ponctuation).
    """
    return re.findall(r'\b\w+\b', phrase.lower())

def bag_of_words(phrase_tokenisee, tous_les_mots):
    """
    Crée un 'sac de mots' (bag of words).
    """
    bag = np.zeros(len(tous_les_mots), dtype=np.float32)
    for idx, mot in enumerate(tous_les_mots):
        if mot in phrase_tokenisee:
            bag[idx] = 1.0
    return bag

# ==========================================
# 2. ARCHITECTURE DU RÉSEAU DE NEURONES
# ==========================================

class ReseauChatbot(nn.Module):
    def __init__(self, taille_entree, taille_cachee, num_classes):
        super(ReseauChatbot, self).__init__()
        self.l1 = nn.Linear(taille_entree, taille_cachee) 
        self.l2 = nn.Linear(taille_cachee, taille_cachee) 
        self.l3 = nn.Linear(taille_cachee, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out

# ==========================================
# 3. GESTION DU DATASET ET ENTRAINEMENT
# ==========================================

class ChatDataset(Dataset):
    def __init__(self, X_train, y_train):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

def initialiser_json_si_besoin(nom_fichier="intents.json"):
    if not os.path.exists(nom_fichier):
        # Exemple minimal avec gestion de contexte
        donnees_base = {
            "intents": [
                {
                    "tag": "salutation",
                    "patterns": ["bonjour", "salut"],
                    "responses": ["Bonjour ! Tu veux que je te raconte une blague ?"],
                    "context_set": "proposer_blague"
                },
                {
                    "tag": "oui",
                    "patterns": ["oui", "ouais", "d'accord", "ok"],
                    "responses": ["Super ! Pourquoi les plongeurs plongent toujours en arrière ? Parce que sinon ils tombent dans le bateau !"],
                    "context_filter": "proposer_blague"
                },
                {
                    "tag": "non",
                    "patterns": ["non", "pas envie", "nan"],
                    "responses": ["Pas de souci, on peut parler d'autre chose !"],
                    "context_filter": "proposer_blague"
                }
            ]
        }
        with open(nom_fichier, 'w', encoding='utf-8') as f:
            json.dump(donnees_base, f, indent=4, ensure_ascii=False)

def entrainer_modele(nom_fichier="intents.json"):
    print("\n[Entraînement en cours...]")
    with open(nom_fichier, 'r', encoding='utf-8') as f:
        intents = json.load(f)

    tous_les_mots = []
    tags = []
    xy = []

    for intent in intents['intents']:
        tag = intent['tag']
        tags.append(tag)
        for pattern in intent['patterns']:
            mots = tokenize(pattern)
            tous_les_mots.extend(mots)
            xy.append((mots, tag))

    tous_les_mots = sorted(list(set(tous_les_mots)))
    tags = sorted(set(tags))

    X_train = []
    y_train = []
    for (phrase_tokenisee, tag) in xy:
        bag = bag_of_words(phrase_tokenisee, tous_les_mots)
        X_train.append(bag)
        label = tags.index(tag)
        y_train.append(label)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    num_epochs = 500
    batch_size = 8
    learning_rate = 0.001
    taille_entree = len(X_train[0])
    taille_cachee = 16 # Augmenté pour gérer plus de phrases
    num_classes = len(tags)

    dataset = ChatDataset(X_train, y_train)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    modele = ReseauChatbot(taille_entree, taille_cachee, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(modele.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for (words, labels) in train_loader:
            words = words.to(torch.float32)
            labels = labels.to(torch.long)
            outputs = modele(words)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print("[Entraînement terminé !]")
    return modele, tous_les_mots, tags, intents

# ==========================================
# 4. BOUCLE PRINCIPALE AVEC CONTEXTE
# ==========================================

def demarrer_chat():
    fichier_json = "intents.json"
    initialiser_json_si_besoin(fichier_json)
    
    modele, tous_les_mots, tags, intents = entrainer_modele(fichier_json)
    modele.eval()

    # Variable pour stocker le contexte actuel
    contexte_actuel = ""

    print("Discutons ! (tapez 'quitter' ou 'apprendre')")
    
    while True:
        phrase = input("Vous : ")
        if phrase.lower() == "quitter":
            break
            
        # Logique de prédiction
        phrase_tokenisee = tokenize(phrase)
        X = bag_of_words(phrase_tokenisee, tous_les_mots)
        X = torch.from_numpy(X.reshape(1, X.shape[0]))

        output = modele(X)
        probs = torch.softmax(output, dim=1)
        prob, predicted = torch.max(probs, dim=1)
        
        tag_predit = tags[predicted.item()]

        if prob.item() > 0.75:
            trouve = False
            for i in intents['intents']:
                if i['tag'] == tag_predit:
                    # Vérification du filtre de contexte
                    # Si l'intention a un filtre, il doit correspondre au contexte actuel
                    if 'context_filter' not in i or i['context_filter'] == contexte_actuel:
                        reponse = random.choice(i['responses'])
                        print(f"Bot : {reponse}")
                        
                        # Mise à jour du contexte pour le prochain tour
                        if 'context_set' in i:
                            contexte_actuel = i['context_set']
                        else:
                            contexte_actuel = "" # Reset si pas de nouveau contexte
                        
                        trouve = True
                        break
            
            if not trouve:
                # Si on a prédit un tag mais que le contexte ne correspondait pas
                print("Bot : Je ne suis pas sûr de comprendre par rapport à ce qu'on disait...")
        else:
            print("Bot : Je n'ai pas compris. Voulez-vous m'apprendre cette phrase ?")

if __name__ == "__main__":
    demarrer_chat()
