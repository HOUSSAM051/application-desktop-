import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from docx import Document
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from collections import Counter
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import KFold
from PIL import Image, ImageTk
import os
import sys
import traceback
from io import StringIO
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Configuration des couleurs et styles modernes
COLORS = {
    'primary': '#1a237e',      # Bleu foncé professionnel
    'secondary': '#3949ab',    # Bleu indigo
    'accent': '#d32f2f',       # Rouge accent
    'background': '#f5f5f5',   # Gris très clair
    'text': '#212121',         # Texte presque noir
    'button': '#3949ab',       # Bleu pour les boutons
    'button_hover': '#1a237e', # Bleu plus foncé pour hover
    'success': '#2e7d32',      # Vert pour les succès
    'warning': '#f57c00',      # Orange pour les avertissements
    'card_bg': '#ffffff',      # Blanc pour les cartes
    'border': '#e0e0e0'        # Gris clair pour les bordures
}

# Style des polices
FONTS = {
    'title': ('Helvetica', 24, 'bold'),
    'subtitle': ('Helvetica', 18, 'bold'),
    'button': ('Helvetica', 12, 'bold'),
    'text': ('Helvetica', 11),
    'small': ('Helvetica', 10)
}

def create_styled_button(parent, text, command, width=20, height=2):
    button = tk.Button(
        parent,
        text=text,
        command=command,
        width=width,
        height=height,
        bg=COLORS['button'],
        fg='white',
        font=FONTS['button'],
        relief='flat',
        borderwidth=0,
        cursor='hand2',
        activebackground=COLORS['button_hover'],
        activeforeground='white'
    )
    
    def on_enter(e):
        button['bg'] = COLORS['button_hover']
    
    def on_leave(e):
        button['bg'] = COLORS['button']
    
    button.bind("<Enter>", on_enter)
    button.bind("<Leave>", on_leave)
    return button

def create_styled_label(parent, text, font_size=20, is_title=False):
    return tk.Label(
        parent,
        text=text,
        font=FONTS['title'] if is_title else FONTS['subtitle'],
        fg=COLORS['text'],
        bg=COLORS['background'],
        pady=10 if is_title else 5
    )

def create_card(parent):
    card = tk.Frame(
        parent,
        bg=COLORS['card_bg'],
        relief='flat',
        borderwidth=1,
        highlightbackground=COLORS['border'],
        highlightthickness=1
    )
    return card

# Fonction pour la régression linéaire
def regression_lineaire():
    if 'donnees_importees' not in globals():
        messagebox.showerror("Erreur", "Veuillez d'abord importer un fichier de données")
        return
        
    df = donnees_importees
    X = df.iloc[:, :-1].values  # Toutes les colonnes sauf la dernière
    y = df.iloc[:, -1].values   # Dernière colonne
    
    # Modèle
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    # Coefficients
    print("✅ Coefficients de la régression :")
    for i, col in enumerate(df.columns[:-1]):
        print(f"{col}: {model.coef_[i]:.2f}")
    print(f"Biais (intercept) : {model.intercept_:.2f}")
    
    # Marges de variation
    print("\n✅ Marges de variation :")
    for col in df.columns:
        print(f"{col}: {df[col].min():.2f} à {df[col].max():.2f}")
    
    # Affichage graphique
    plt.figure(figsize=(8, 5))
    plt.scatter(X[:, 0], y, color='blue', label='Données réelles')
    plt.scatter(X[:, 0], y_pred, color='red', label='Prédictions', alpha=0.5)
    plt.title("Régression Linéaire")
    plt.xlabel(df.columns[0])
    plt.ylabel(df.columns[-1])
    plt.legend()
    plt.grid(True)
    plt.show()


def clustering_kmeans():
    if 'donnees_importees' not in globals():
        messagebox.showerror("Erreur", "Veuillez d'abord importer un fichier de données")
        return
        
    df = donnees_importees
    X = df.iloc[:, :-1].values  # Toutes les colonnes sauf la dernière
    
    # Applique KMeans
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)
    
    # Affichage
    plt.figure(figsize=(8, 5))
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                s=200, c='red', marker='x', label='Centres')
    plt.title("Clustering K-Means sur vos données")
    plt.xlabel(df.columns[0])
    plt.ylabel(df.columns[1])
    plt.legend()
    plt.grid(True)
    plt.show()

def modele_arima():
    if 'donnees_importees' not in globals():
        messagebox.showerror("Erreur", "Veuillez d'abord importer un fichier de données")
        return
        
    df = donnees_importees
    serie = df.iloc[:, -1].values  # Utilise la dernière colonne comme série temporelle
    
    # Ajustement du modèle ARIMA(p=1,d=1,q=1)
    model = ARIMA(serie, order=(1, 1, 1))
    model_fit = model.fit()
    
    # Prédiction
    forecast = model_fit.predict(start=0, end=len(serie)-1)
    
    # Affichage
    plt.figure(figsize=(9, 5))
    plt.plot(range(len(serie)), serie, label="Données réelles", color="blue")
    plt.plot(range(len(forecast)), forecast, label="Prévision ARIMA", color="red")
    plt.title("Prévision ARIMA sur vos données")
    plt.xlabel("Temps")
    plt.ylabel(df.columns[-1])
    plt.legend()
    plt.grid(True)
    plt.show()

def random_forest_iris():
    if 'donnees_importees' not in globals():
        messagebox.showerror("Erreur", "Veuillez d'abord importer un fichier de données")
        return
        
    df = donnees_importees
    X = df.iloc[:, :-1].values  # Toutes les colonnes sauf la dernière
    y = df.iloc[:, -1].values   # Dernière colonne
    
    # Réduction à 2 dimensions pour affichage
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Entraîner le modèle
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    y_pred = model.predict(X)
    
    # Affichage
    plt.figure(figsize=(8, 5))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred, cmap='viridis')
    plt.title("Classification Random Forest sur vos données")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.grid(True)
    plt.show()

def validation_croisee():
    if 'donnees_importees' not in globals():
        messagebox.showerror("Erreur", "Veuillez d'abord importer un fichier de données")
        return
        
    try:
        df = donnees_importees
        
        # Afficher les informations de débogage
        print("\n=== Informations de débogage ===")
        print(f"Forme des données: {df.shape}")
        print(f"Types des colonnes:\n{df.dtypes}")
        print(f"Valeurs manquantes:\n{df.isnull().sum()}")
        print("==============================\n")
        
        # Vérification des données
        if df.empty:
            messagebox.showerror("Erreur", "Le fichier de données est vide")
            return
            
        # Vérification des types de données
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric_cols) > 0:
            messagebox.showerror("Erreur", f"Les colonnes suivantes ne sont pas numériques: {', '.join(non_numeric_cols)}")
            return
            
        # Vérification des valeurs manquantes
        if df.isnull().any().any():
            messagebox.showerror("Erreur", "Le fichier contient des valeurs manquantes")
            return
            
        X = df.iloc[:, :-1].values  # Toutes les colonnes sauf la dernière
        y = df.iloc[:, -1].values   # Dernière colonne
        
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        
        # Vérification de la taille des données
        if len(X) < 5:
            messagebox.showerror("Erreur", "Il faut au moins 5 observations pour la validation croisée")
            return
        
        # Modèle de régression
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Validation croisée R²
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        r2_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
        neg_mse_scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
        mse_scores = -neg_mse_scores
        
        resultat = f"✅ Scores validation croisée (régression, 5 plis) :\n"
        for i, (r2, mse) in enumerate(zip(r2_scores, mse_scores), 1):
            resultat += f"Pli {i}: R² = {r2:.4f}, MSE = {mse:.2f}\n"
        resultat += f"\n🎯 Moyenne R² : {np.mean(r2_scores):.4f}\n"
        resultat += f"📉 Moyenne MSE : {np.mean(mse_scores):.2f}"
        
        messagebox.showinfo("Résultats de la validation croisée (régression)", resultat)
        
        # Affichage graphique des scores R²
        plt.figure(figsize=(7, 4))
        plt.plot(range(1, 6), r2_scores, marker='o', linestyle='-', label='Score R² par pli')
        plt.axhline(np.mean(r2_scores), color='red', linestyle='--', label='Moyenne R²')
        plt.title("Validation croisée (régression) sur vos données")
        plt.xlabel("Pli (Fold)")
        plt.ylabel("Score R²")
        plt.xticks(range(1, 6))
        plt.legend()
        plt.grid(True)
        plt.show()
        
    except Exception as e:
        print(f"\nErreur détaillée: {str(e)}")
        messagebox.showerror("Erreur", f"Une erreur est survenue lors de la validation croisée : {str(e)}")

def selectionner_fichier_word():
    # Ouvrir la boîte de dialogue pour sélectionner un fichier Word
    fichier = filedialog.askopenfilename(
        title="Sélectionner un fichier Word",
        filetypes=[("Fichiers Word", "*.docx"), ("Tous les fichiers", "*.*")]
    )
    
    if fichier:
        try:
            # Lire le fichier Word
            doc = Document(fichier)
            
            # Vérifier le format des données
            donnees = []
            en_tete = None
            
            for i, paragraph in enumerate(doc.paragraphs):
                if paragraph.text.strip():
                    if i == 0:  # Première ligne = en-tête
                        en_tete = paragraph.text.strip().split(',')
                        if len(en_tete) != 3:
                            raise ValueError("L'en-tête doit contenir exactement 3 colonnes séparées par des virgules")
                    else:  # Données
                        ligne = paragraph.text.strip().split(',')
                        if len(ligne) != 3:
                            raise ValueError(f"La ligne {i+1} ne contient pas 3 valeurs")
                        try:
                            # Convertir les valeurs en nombres
                            donnees.append([float(x.strip()) for x in ligne])
                        except ValueError:
                            raise ValueError(f"La ligne {i+1} contient des valeurs non numériques")
            
            if len(donnees) < 10:
                raise ValueError("Le fichier doit contenir au moins 10 lignes de données")
            
            # Convertir en DataFrame pandas
            df = pd.DataFrame(donnees, columns=en_tete)
            
            # Stocker les données pour une utilisation ultérieure
            global donnees_importees
            donnees_importees = df
            
            # Afficher un message de confirmation avec les statistiques
            stats = f"Fichier chargé avec succès!\n\n"
            stats += f"Nombre de lignes: {len(donnees)}\n"
            stats += f"Colonnes: {', '.join(en_tete)}\n\n"
            stats += "Statistiques des données:\n"
            stats += df.describe().to_string()
            
            messagebox.showinfo("Succès", stats)
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de la lecture du fichier: {str(e)}")

def afficher_page_algorithmes():
    # Créer une nouvelle fenêtre pour les algorithmes
    algo_window = tk.Toplevel()
    algo_window.title("Algorithmes d'IA")
    algo_window.geometry("800x600")
    algo_window.configure(bg=COLORS['background'])
    
    # Titre
    titre = create_styled_label(algo_window, "Choisissez l'algorithme d'IA", 24, True)
    titre.pack(pady=20)
    
    # Frame pour les boutons avec un style moderne
    boutons_frame = tk.Frame(algo_window, bg=COLORS['background'])
    boutons_frame.pack(pady=20)
    
    # Boutons pour chaque algorithme avec le nouveau style
    create_styled_button(boutons_frame, "Régression Linéaire", regression_lineaire).pack(pady=10)
    create_styled_button(boutons_frame, "Clustering K-Means", clustering_kmeans).pack(pady=10)
    create_styled_button(boutons_frame, "Modèle ARIMA", modele_arima).pack(pady=10)
    create_styled_button(boutons_frame, "Random Forest", random_forest_iris).pack(pady=10)
    create_styled_button(boutons_frame, "Validation Croisée", validation_croisee).pack(pady=10)
    
    # Bouton retour avec un style différent
    retour_button = create_styled_button(algo_window, "Retour au menu", algo_window.destroy, 15, 2)
    retour_button.configure(bg=COLORS['accent'])
    retour_button.pack(pady=20)

def get_resource_path(relative_path):
    """Trouve le chemin du fichier, compatible PyInstaller"""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), relative_path)

class AIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Application d'Analyse de Données")
        self.root.geometry("1000x700")  # Fenêtre plus grande
        self.root.configure(bg=COLORS['background'])

        # Créer un frame pour le header avec une ombre
        self.header_frame = tk.Frame(self.root, bg=COLORS['card_bg'], relief='flat', borderwidth=1)
        self.header_frame.pack(fill='x', pady=(15, 5), padx=20)
        
        # Ajouter une bordure subtile au header
        self.header_frame.configure(highlightbackground=COLORS['border'], highlightthickness=1)

        # Remplacer l'obtention du chemin du logo par la fonction compatible PyInstaller
        logo_path = get_resource_path("emsi_logo.png")

        try:
            if os.path.exists(logo_path):
                logo_img = Image.open(logo_path)
                logo_img = logo_img.resize((300, 90), Image.Resampling.LANCZOS)
                logo_photo = ImageTk.PhotoImage(logo_img)
                self.logo_label = tk.Label(self.header_frame, image=logo_photo, bg=COLORS['card_bg'])
                self.logo_label.image = logo_photo
                self.logo_label.pack(side='left', padx=20, pady=10)
            else:
                print(f"Logo non trouvé à l'emplacement : {logo_path}")
        except Exception as e:
            print(f"Erreur lors du chargement du logo : {e}")

        # Affichage du nom complet à droite
        self.nom_label = tk.Label(
            self.header_frame, 
            text="Houssam Nassih", 
            font=FONTS['subtitle'],
            fg=COLORS['accent'], 
            bg=COLORS['card_bg']
        )
        self.nom_label.pack(side='right', padx=20, pady=10)

        # Titre principal avec style
        self.titre = create_styled_label(self.root, "Application d'Analyse de Données", is_title=True)
        self.titre.pack(pady=20)
        
        # Frame pour les boutons avec style
        self.boutons_frame = create_card(self.root)
        self.boutons_frame.pack(pady=30, padx=40, fill='x')
        
        # Titre de la section
        self.section_title = tk.Label(
            self.boutons_frame,
            text="Sélectionnez une action",
            font=FONTS['subtitle'],
            fg=COLORS['text'],
            bg=COLORS['card_bg'],
            pady=20
        )
        self.section_title.pack()
        
        # Frame pour les boutons
        self.buttons_container = tk.Frame(self.boutons_frame, bg=COLORS['card_bg'])
        self.buttons_container.pack(pady=20, padx=40)
        
        # Bouton pour importer les données
        self.import_button = create_styled_button(self.buttons_container, "Importer des données", self.importer_donnees)
        self.import_button.pack(pady=15, fill='x')
        
        # Bouton principal avec le nouveau style
        self.algo_button = create_styled_button(self.buttons_container, "Algorithmes d'IA", self.afficher_page_algorithmes)
        self.algo_button.pack(pady=15, fill='x')
        
        # Footer
        self.footer = tk.Label(
            self.root,
            text="© 2024 Application d'Analyse de Données",
            font=FONTS['small'],
            fg=COLORS['text'],
            bg=COLORS['background']
        )
        self.footer.pack(side='bottom', pady=10)

        # Initialisation des composants pour l'affichage des graphiques
        self.figure = plt.Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Zone de texte pour les résultats
        self.text_output = tk.Text(self.root, height=10, width=80, font=FONTS['text'])
        self.text_output.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Barre de statut
        self.status_var = tk.StringVar()
        self.status_var.set("Prêt")
        self.status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Variables pour les données
        self.X = None
        self.y = None
        
        # Vérification de la disponibilité des bibliothèques
        self.SKLEARN_AVAILABLE = True
        self.MATPLOTLIB_AVAILABLE = True
        self.STATSMODELS_AVAILABLE = True

        # Générer des données automatiquement au démarrage
        self.generer_donnees_test()

    def generer_donnees_test(self):
        """Génère des données de test pour les algorithmes"""
        try:
            # Générer des données de régression
            X, y = make_regression(
                n_samples=100,
                n_features=3,
                noise=10,
                random_state=42
            )
            
            # Créer un DataFrame avec des noms de colonnes explicites
            self.X = pd.DataFrame(
                X,
                columns=['Puissance (ch)', 'Poids (kg)', 'Vitesse max (km/h)']
            )
            self.y = pd.Series(y, name='Consommation (L/100km)')
            
            # Afficher les statistiques des données générées
            self.clear_output()
            self.text_output.insert(tk.END, "=== DONNÉES DE TEST GÉNÉRÉES ===\n\n")
            self.text_output.insert(tk.END, f"Nombre de lignes: {len(self.X)}\n")
            self.text_output.insert(tk.END, f"Nombre de colonnes: {len(self.X.columns)}\n\n")
            self.text_output.insert(tk.END, "Statistiques des données:\n")
            self.text_output.insert(tk.END, self.X.describe().to_string())
            
            self.status_var.set("Données de test générées avec succès")
            
        except Exception as e:
            self.show_error(e)
            self.status_var.set("Erreur lors de la génération des données de test")

    def importer_donnees(self):
        """Importe des données depuis un fichier CSV ou Excel"""
        fichier = filedialog.askopenfilename(
            title="Sélectionner un fichier de données",
            filetypes=[
                ("Fichiers CSV", "*.csv"),
                ("Fichiers Excel", "*.xlsx;*.xls"),
                ("Tous les fichiers", "*.*")
            ]
        )
        
        if fichier:
            try:
                # Lire le fichier selon son extension
                if fichier.endswith('.csv'):
                    df = pd.read_csv(fichier)
                elif fichier.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(fichier)
                else:
                    raise ValueError("Format de fichier non supporté")
                
                # Vérifier que le fichier contient au moins 2 colonnes
                if len(df.columns) < 2:
                    raise ValueError("Le fichier doit contenir au moins 2 colonnes")
                
                # Stocker les données
                self.X = df.iloc[:, :-1]  # Toutes les colonnes sauf la dernière
                self.y = df.iloc[:, -1]   # Dernière colonne
                
                # Afficher les statistiques
                self.clear_output()
                self.text_output.insert(tk.END, "=== DONNÉES IMPORTÉES ===\n\n")
                self.text_output.insert(tk.END, f"Nombre de lignes: {len(df)}\n")
                self.text_output.insert(tk.END, f"Nombre de colonnes: {len(df.columns)}\n\n")
                self.text_output.insert(tk.END, "Statistiques des données:\n")
                self.text_output.insert(tk.END, df.describe().to_string())
                
                self.status_var.set("Données importées avec succès")
                messagebox.showinfo("Succès", "Les données ont été importées avec succès")
                
            except Exception as e:
                self.show_error(e)
                self.status_var.set("Erreur lors de l'importation des données")

    def show_error(self, exception):
        """Affiche les erreurs de manière claire"""
        error_msg = f"Une erreur est survenue:\n{str(exception)}\n\n"
        error_msg += "Traceback:\n" + "\n".join(traceback.format_tb(exception._traceback_))
        messagebox.showerror("Erreur", error_msg)
        self.status_var.set("Erreur lors de l'exécution")
        self.text_output.insert(tk.END, f"\n\n=== ERREUR ===\n{error_msg}")

    def afficher_page_algorithmes(self):
        # Créer une nouvelle fenêtre pour les algorithmes
        algo_window = tk.Toplevel()
        algo_window.title("Algorithmes d'IA")
        algo_window.geometry("800x600")
        algo_window.configure(bg=COLORS['background'])
        
        # Titre
        titre = create_styled_label(algo_window, "Choisissez l'algorithme d'IA", 24, True)
        titre.pack(pady=20)
        
        # Frame pour les boutons avec un style moderne
        boutons_frame = tk.Frame(algo_window, bg=COLORS['background'])
        boutons_frame.pack(pady=20)
        
        # Boutons pour chaque algorithme avec le nouveau style
        create_styled_button(boutons_frame, "Régression Linéaire", lambda: self.show_linear_regression()).pack(pady=10)
        create_styled_button(boutons_frame, "Clustering K-Means", lambda: self.show_clustering()).pack(pady=10)
        create_styled_button(boutons_frame, "Modèle ARIMA", lambda: self.show_arima()).pack(pady=10)
        create_styled_button(boutons_frame, "Random Forest", lambda: self.show_random_forest()).pack(pady=10)
        create_styled_button(boutons_frame, "Validation Croisée", lambda: self.show_cross_validation()).pack(pady=10)
        
        # Bouton retour avec un style différent
        retour_button = create_styled_button(algo_window, "Retour au menu", algo_window.destroy, 15, 2)
        retour_button.configure(bg=COLORS['accent'])
        retour_button.pack(pady=20)
    
    def show_linear_regression(self):
        """Affiche les résultats de la régression linéaire"""
        self.clear_output()
        self.status_var.set("Exécution de la régression linéaire...")
        self.root.update()
        
        try:
            if not self.SKLEARN_AVAILABLE:
                raise ImportError("Scikit-learn n'est pas disponible")
            
            # Créer une nouvelle fenêtre pour les entrées
            input_window = tk.Toplevel(self.root)
            input_window.title("Paramètres de la régression linéaire")
            input_window.geometry("400x500")
            input_window.configure(bg=COLORS['background'])
            
            # Titre
            tk.Label(
                input_window,
                text="Entrez les valeurs min et max",
                font=FONTS['subtitle'],
                bg=COLORS['background'],
                fg=COLORS['text']
            ).pack(pady=20)
            
            # Frame pour les entrées
            input_frame = tk.Frame(input_window, bg=COLORS['background'])
            input_frame.pack(pady=20, padx=20)
            
            # Variables pour stocker les entrées
            min_values = {}
            max_values = {}
            
            # Créer les champs de saisie pour chaque caractéristique
            for i, col in enumerate(self.X.columns):
                tk.Label(
                    input_frame,
                    text=f"{col}:",
                    font=FONTS['text'],
                    bg=COLORS['background'],
                    fg=COLORS['text']
                ).grid(row=i*2, column=0, pady=10, sticky='w')
                
                # Frame pour min et max
                value_frame = tk.Frame(input_frame, bg=COLORS['background'])
                value_frame.grid(row=i*2+1, column=0, pady=5)
                
                # Min
                tk.Label(value_frame, text="Min:", bg=COLORS['background']).pack(side='left', padx=5)
                min_values[col] = tk.Entry(value_frame, width=10)
                min_values[col].pack(side='left', padx=5)
                min_values[col].insert(0, str(self.X[col].min()))
                
                # Max
                tk.Label(value_frame, text="Max:", bg=COLORS['background']).pack(side='left', padx=5)
                max_values[col] = tk.Entry(value_frame, width=10)
                max_values[col].pack(side='left', padx=5)
                max_values[col].insert(0, str(self.X[col].max()))
            
            def calculer_regression():
                try:
                    # Récupérer les valeurs min et max
                    min_vals = {col: float(min_values[col].get()) for col in self.X.columns}
                    max_vals = {col: float(max_values[col].get()) for col in self.X.columns}
                    
                    # Entraînement du modèle
                    model = LinearRegression()
                    model.fit(self.X, self.y)
                    
                    # Créer des données de test avec les valeurs min et max
                    test_data = pd.DataFrame({
                        col: np.linspace(min_vals[col], max_vals[col], 100)
                        for col in self.X.columns
                    })
                    
                    # Prédictions
                    y_pred = model.predict(test_data)
                    
                    # Affichage des résultats
                    self.clear_output()
                    self.text_output.insert(tk.END, "=== RÉGRESSION LINÉAIRE ===\n\n")
                    self.text_output.insert(tk.END, f"Coefficients: {model.coef_}\n")
                    self.text_output.insert(tk.END, f"Intercept: {model.intercept_}\n")
                    self.text_output.insert(tk.END, f"Score R²: {model.score(self.X, self.y):.3f}\n\n")
                    
                    # Graphique
                    if self.MATPLOTLIB_AVAILABLE:
                        ax = self.figure.add_subplot(111)
                        ax.scatter(self.X.iloc[:, 0], self.y, color='blue', label='Données réelles', alpha=0.5)
                        ax.plot(test_data.iloc[:, 0], y_pred, color='red', linewidth=2, label='Régression')
                        ax.set_xlabel(self.X.columns[0])
                        ax.set_ylabel('Consommation (L/100km)')
                        ax.set_title('Régression Linéaire')
                        ax.legend()
                        ax.grid(True)
                        self.canvas.draw()
                    
                    self.status_var.set("Régression linéaire terminée")
                    input_window.destroy()
                    
                except ValueError as e:
                    messagebox.showerror("Erreur", "Veuillez entrer des valeurs numériques valides")
                except Exception as e:
                    self.show_error(e)
            
            # Bouton pour calculer
            create_styled_button(
                input_window,
                "Calculer la régression",
                calculer_regression
            ).pack(pady=20)
            
        except Exception as e:
            self.show_error(e)
    
    def show_clustering(self):
        """Affiche les résultats du clustering"""
        self.clear_output()
        self.status_var.set("Exécution du clustering...")
        self.root.update()
        
        try:
            if not self.SKLEARN_AVAILABLE:
                raise ImportError("Scikit-learn n'est pas disponible")
            
            # Clustering K-means
            kmeans = KMeans(n_clusters=3, random_state=42)
            clusters = kmeans.fit_predict(self.X)
            
            # Affichage des résultats
            self.text_output.insert(tk.END, "=== CLUSTERING (K-means) ===\n\n")
            self.text_output.insert(tk.END, f"Centroïdes:\n{kmeans.cluster_centers_}\n\n")
            self.text_output.insert(tk.END, f"Répartition des clusters:\n{pd.Series(clusters).value_counts().to_string()}\n\n")
            
            # Graphique
            if self.MATPLOTLIB_AVAILABLE:
                ax = self.figure.add_subplot(111)
                scatter = ax.scatter(self.X.iloc[:, 0], self.X.iloc[:, 1], c=clusters, cmap='viridis')
                ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                         s=200, c='red', marker='X', label='Centroïdes')
                ax.set_xlabel('Puissance (ch)')
                ax.set_ylabel('Poids (kg)')
                ax.set_title('Clustering K-means')
                ax.legend()
                ax.grid(True)
                self.canvas.draw()
            
            self.status_var.set("Clustering terminé")
        except Exception as e:
            self.show_error(e)
    
    def show_arima(self):
        """Affiche les résultats des séries temporelles ARIMA"""
        self.clear_output()
        self.status_var.set("Exécution de l'analyse ARIMA...")
        self.root.update()
        
        try:
            if not self.STATSMODELS_AVAILABLE:
                raise ImportError("Statsmodels n'est pas disponible")
            
            # Séries temporelles (simulées)
            ts_data = self.y.rolling(window=5).mean().dropna()
            
            # Modèle ARIMA
            model = ARIMA(ts_data, order=(1, 0, 1))
            model_fit = model.fit()
            
            # Prédictions
            predictions = model_fit.predict()
            
            # Affichage des résultats
            self.text_output.insert(tk.END, "=== SÉRIES TEMPORELLES (ARIMA) ===\n\n")
            
            # Capture du summary dans une variable
            old_stdout = sys.stdout
            sys.stdout = mystdout = StringIO()
            print(model_fit.summary())
            sys.stdout = old_stdout
            self.text_output.insert(tk.END, mystdout.getvalue())
            
            # Graphique
            if self.MATPLOTLIB_AVAILABLE:
                ax = self.figure.add_subplot(111)
                ax.plot(ts_data.index, ts_data, label='Données originales')
                ax.plot(predictions.index, predictions, label='Prédictions ARIMA', linestyle='--')
                ax.set_xlabel('Index')
                ax.set_ylabel('Consommation (L/100km)')
                ax.set_title('Analyse des Séries Temporelles (ARIMA)')
                ax.legend()
                ax.grid(True)
                self.canvas.draw()
            
            self.status_var.set("Analyse ARIMA terminée")
        except Exception as e:
            self.show_error(e)
    
    def show_random_forest(self):
        """Affiche les résultats de la forêt aléatoire"""
        self.clear_output()
        self.status_var.set("Exécution de la forêt aléatoire...")
        self.root.update()
        
        try:
            if not self.SKLEARN_AVAILABLE:
                raise ImportError("Scikit-learn n'est pas disponible")
            
            # Modèle Random Forest
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(self.X, self.y)
            
            # Importance des caractéristiques
            importances = model.feature_importances_
            
            # Affichage des résultats
            self.text_output.insert(tk.END, "=== RANDOM FOREST ===\n\n")
            self.text_output.insert(tk.END, f"Score R²: {model.score(self.X, self.y):.3f}\n\n")
            self.text_output.insert(tk.END, "Importance des caractéristiques:\n")
            
            for name, importance in zip(self.X.columns, importances):
                self.text_output.insert(tk.END, f"{name}: {importance:.3f}\n")
            
            # Graphique
            if self.MATPLOTLIB_AVAILABLE:
                ax = self.figure.add_subplot(111)
                ax.barh(self.X.columns, importances)
                ax.set_xlabel('Importance')
                ax.set_title('Importance des Caractéristiques (Random Forest)')
                ax.grid(True)
                self.canvas.draw()
            
            self.status_var.set("Forêt aléatoire terminée")
        except Exception as e:
            self.show_error(e)
    
    def show_cross_validation(self):
        """Affiche les résultats de la validation croisée"""
        self.clear_output()
        self.status_var.set("Exécution de la validation croisée...")
        self.root.update()
        
        try:
            if not self.SKLEARN_AVAILABLE:
                raise ImportError("Scikit-learn n'est pas disponible")
            
            # Modèle pour la validation croisée
            model = LinearRegression()
            
            # Validation croisée
            scores = cross_val_score(model, self.X, self.y, cv=5, scoring='r2')
            
            # Affichage des résultats
            self.text_output.insert(tk.END, "=== VALIDATION CROISÉE ===\n\n")
            self.text_output.insert(tk.END, f"Scores R² pour chaque pli: {scores}\n")
            self.text_output.insert(tk.END, f"Score moyen: {scores.mean():.3f} ± {scores.std():.3f}\n\n")
            
            # Graphique
            if self.MATPLOTLIB_AVAILABLE:
                ax = self.figure.add_subplot(111)
                ax.plot(range(1, 6), scores, marker='o')
                ax.axhline(scores.mean(), color='red', linestyle='--', label='Moyenne')
                ax.set_xlabel('Pli')
                ax.set_ylabel('Score R²')
                ax.set_title('Validation Croisée (5 plis)')
                ax.legend()
                ax.grid(True)
                self.canvas.draw()
            
            self.status_var.set("Validation croisée terminée")
        except Exception as e:
            self.show_error(e)
    
    def clear_output(self):
        """Efface les résultats précédents"""
        self.text_output.delete(1.0, tk.END)
        if self.MATPLOTLIB_AVAILABLE:
            self.figure.clf()
            self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = AIApp(root)
    root.mainloop()