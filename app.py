import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Chargement des données ---
title_list = pd.read_csv('user_ratings_genres_mov.csv')
movie_titles = title_list['title'].unique()

# --- Fonction pour injecter du CSS personnalisé ---
def inject_custom_css():
    custom_css = """
    <style>
    body { background-color: #f4f6f8; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; color: #333333; }
    h1, h2, h3, h4, h5, h6 { color: #333333; }
    .css-1d391kg { background-color: #1a1a1a; color: #ffffff; }
    .css-1d391kg a { color: #4fd3c4; }
    div.stButton > button { background-color: #4fd3c4; color: white; border: none; border-radius: 4px; padding: 0.5em 1em; font-size: 1em; cursor: pointer; }
    div.stButton > button:hover { background-color: #3bb8a1; }
    .stSelectbox select { background-color: #ffffff; border: 1px solid #cccccc; border-radius: 4px; }
    .main { padding: 20px; }
    .css-1y4k81r { padding-top: 15px; }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

# --- Création de la matrice utilisateur-film ---
def create_user_movie_matrix(df):
    """Crée une matrice utilisateur-film à partir du dataframe."""
    return df.pivot_table(index='userId', columns='title', values='rating').fillna(0)

# --- Recommandations collaboratives basées sur la mémoire ---
def recommend_item_user(new_user_id, user_movie_matrix, top_n=5):
    """Approche Item-User : calcule une note prédite pour chaque film non noté par le nouvel utilisateur."""
    item_similarity = cosine_similarity(user_movie_matrix.T)
    item_similarity_df = pd.DataFrame(item_similarity, index=user_movie_matrix.columns, columns=user_movie_matrix.columns)
    user_ratings = user_movie_matrix.loc[new_user_id]
    pred_ratings = {}
    for movie in user_movie_matrix.columns:
        if user_ratings[movie] == 0:
            rated_movies = user_ratings[user_ratings > 0].index
            numer = 0
            denom = 0
            for rated in rated_movies:
                numer += item_similarity_df.loc[movie, rated] * user_ratings[rated]
                denom += abs(item_similarity_df.loc[movie, rated])
            if denom > 0:
                pred_ratings[movie] = numer / denom
    top_movies = sorted(pred_ratings.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return top_movies

def recommend_user_item(new_user_id, user_movie_matrix, top_n=5):
    """
    Approche User-Item : prédit une note pour chaque film non noté par le nouvel utilisateur
    en utilisant une moyenne pondérée basée sur la similarité entre utilisateurs.
    
    Améliorations :
      - Utilisation d'une petite valeur epsilon pour éviter la division par zéro.
      - Utilisation de DataFrame.at pour un accès explicite et rapide aux valeurs.
    
    Paramètres:
      - new_user_id : l'identifiant du nouvel utilisateur.
      - user_movie_matrix : DataFrame où les lignes représentent les utilisateurs et les colonnes les films.
      - top_n : nombre de films à recommander (par défaut 5).
      
    Retourne:
      - Une liste de tuples (titre du film, note prédite) triée par ordre décroissant de note prédite.
    """
    epsilon = 1e-8  # Pour éviter la division par zéro
    
    # Calcul de la similarité entre utilisateurs
    user_similarity = cosine_similarity(user_movie_matrix)
    user_similarity_df = pd.DataFrame(user_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)
    
    # Récupérer les notes du nouvel utilisateur
    user_ratings = user_movie_matrix.loc[new_user_id]
    pred_ratings = {}
    
    # Pour chaque film non noté par le nouvel utilisateur
    for movie in user_movie_matrix.columns:
        if user_ratings[movie] == 0:
            numer = 0.0
            denom = 0.0
            # Parcourir les autres utilisateurs
            for other_user in user_movie_matrix.index:
                if other_user != new_user_id and user_movie_matrix.at[other_user, movie] > 0:
                    similarity = user_similarity_df.at[new_user_id, other_user]
                    rating = user_movie_matrix.at[other_user, movie]
                    numer += similarity * rating
                    denom += abs(similarity)
            # Si au moins un voisin a noté le film, calculer la note prédite
            if denom > epsilon:
                pred_ratings[movie] = numer / (denom + epsilon)
    
    # Sélectionner les top_n films par note prédite décroissante
    top_movies = sorted(pred_ratings.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return top_movies


# --- Recommandations collaboratives basées sur un modèle ---
def recommend_nmf(new_user_id, user_movie_matrix, top_n=5):
    """
    Utilise NMF pour factoriser la matrice de notes et reconstruire les notes.
    Recommande les films non notés par le nouvel utilisateur en utilisant la reconstruction.
    
    Améliorations :
      - Utilisation de l'initialisation 'nndsvda' pour une meilleure convergence.
      - Augmentation du nombre d'itérations (max_iter) pour assurer la convergence.
    
    Paramètres:
      - new_user_id : l'identifiant du nouvel utilisateur.
      - user_movie_matrix : DataFrame des notes avec utilisateurs en lignes et films en colonnes.
      - top_n : nombre de films à recommander (par défaut 5).
      
    Retourne:
      - Une liste de tuples (titre du film, note prédite) triée par ordre décroissant.
    """
    nmf_model = NMF(n_components=20, init='nndsvda', max_iter=500, random_state=42)
    W = nmf_model.fit_transform(user_movie_matrix)
    H = nmf_model.components_
    R_pred = np.dot(W, H)
    R_pred_df = pd.DataFrame(R_pred, index=user_movie_matrix.index, columns=user_movie_matrix.columns)
    
    # Récupérer les notes actuelles de l'utilisateur et les prédictions pour les films non notés
    user_ratings = user_movie_matrix.loc[new_user_id]
    predictions = R_pred_df.loc[new_user_id]
    pred_ratings = {movie: predictions[movie] for movie in user_movie_matrix.columns if user_ratings[movie] == 0}
    
    # Sélectionner les films avec les scores les plus élevés
    top_movies = sorted(pred_ratings.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return top_movies


def recommend_svd(new_user_id, user_movie_matrix, top_n=5):
    """
    Utilise SVD (TruncatedSVD) pour reconstruire la matrice de notes et recommande les films non notés par le nouvel utilisateur.
    
    Améliorations :
      - Centrage des notes en soustrayant la note moyenne de chaque utilisateur avant la factorisation.
      - Ajout de la moyenne après la reconstruction pour obtenir les notes prédites.
    
    Paramètres:
      - new_user_id : l'identifiant du nouvel utilisateur.
      - user_movie_matrix : DataFrame des notes où chaque ligne correspond à un utilisateur et chaque colonne à un film.
      - top_n : nombre de films à recommander (par défaut 5).
      
    Retourne:
      - Une liste de tuples (titre du film, note prédite) triée par ordre décroissant de pertinence.
    """
    # Calculer la note moyenne de chaque utilisateur (en ignorant les zéros)
    user_means = user_movie_matrix.replace(0, np.nan).mean(axis=1).fillna(0)
    
    # Centrer la matrice en soustrayant la moyenne de chaque utilisateur
    centered_matrix = user_movie_matrix.sub(user_means, axis=0)
    
    # Appliquer SVD sur la matrice centrée
    from sklearn.decomposition import TruncatedSVD
    svd_model = TruncatedSVD(n_components=20, random_state=42)
    U = svd_model.fit_transform(centered_matrix)
    S = svd_model.singular_values_
    VT = svd_model.components_
    
    # Reconstruction de la matrice centrée
    R_pred_centered = np.dot(U, np.dot(np.diag(S), VT))
    
    # Ajouter la moyenne pour obtenir la matrice de notes prédite
    R_pred = R_pred_centered + user_means.values.reshape(-1, 1)
    R_pred_df = pd.DataFrame(R_pred, index=user_movie_matrix.index, columns=user_movie_matrix.columns)
    
    # Récupérer les prédictions pour le nouvel utilisateur pour les films non notés
    user_ratings = user_movie_matrix.loc[new_user_id]
    predictions = R_pred_df.loc[new_user_id]
    pred_ratings = {movie: predictions[movie] for movie in user_movie_matrix.columns if user_ratings[movie] == 0}
    
    # Retourner les top_n films triés par note prédite décroissante
    top_movies = sorted(pred_ratings.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return top_movies


def recommend_knn(new_user_id, user_movie_matrix, top_n=5, n_neighbors=5):
    """
    Utilise KNN pour recommander les films non notés par le nouvel utilisateur en s'appuyant sur les voisins les plus proches,
    avec une pondération des notes par l'inverse de la distance.
    
    Paramètres:
      - new_user_id: l'identifiant du nouvel utilisateur.
      - user_movie_matrix: DataFrame où les lignes représentent les utilisateurs et les colonnes les films.
      - top_n: nombre de recommandations à retourner.
      - n_neighbors: nombre de voisins à considérer (excluant l'utilisateur lui-même).
      
    Retourne:
      - Une liste de tuples (titre du film, note prédite) triée par ordre décroissant de pertinence.
    """
    from sklearn.neighbors import NearestNeighbors
    
    # Initialisation et entraînement du modèle KNN
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    model_knn.fit(user_movie_matrix)
    
    # Recherche des voisins pour le nouvel utilisateur
    distances, indices = model_knn.kneighbors([user_movie_matrix.loc[new_user_id]], n_neighbors=n_neighbors+1)
    distances = distances.flatten()[1:]  # Exclure l'utilisateur lui-même
    similar_users = indices.flatten()[1:]
    
    user_ratings = user_movie_matrix.loc[new_user_id]
    pred_ratings = {}
    
    # Pour chaque film non noté par le nouvel utilisateur
    for movie in user_movie_matrix.columns:
        if user_ratings[movie] == 0:
            weighted_sum = 0
            weight_sum = 0
            # Parcours des voisins
            for dist, neighbor in zip(distances, similar_users):
                neighbor_rating = user_movie_matrix.iloc[neighbor][movie]
                # Considérer uniquement les voisins ayant noté le film
                if neighbor_rating > 0:
                    # Pondération : plus la distance est faible, plus le poids est élevé
                    weight = 1 / (dist + 1e-8)
                    weighted_sum += neighbor_rating * weight
                    weight_sum += weight

            if weight_sum > 0:
                pred_ratings[movie] = weighted_sum / weight_sum

    top_movies = sorted(pred_ratings.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return top_movies

# --- Recommandation basée sur le contenu ---
def recommend_content(best_film, movies_df, top_n=5):
       # Transformation des genres en représentation TF-IDF
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    tfidf_matrix = vectorizer.fit_transform(movies_df['genres'])
    
    # Calcul de la similarité cosine entre tous les films
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Récupération de l'index du film choisi
    idx = movies_df[movies_df['title'] == best_film].index[0]
    
    # Calcul des scores de similarité pour le film choisi
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Tri des films par ordre décroissant de similarité
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Exclure le film lui-même (premier élément) et prendre les top_n films
    sim_scores = sim_scores[1:top_n+1]
    
    # Récupérer les titres et les scores des films recommandés
    recommendations = [(movies_df.iloc[i]['title'], score) for i, score in sim_scores]
    return recommendations

# --- Interface Streamlit ---
def main():
    inject_custom_css()
    
    st.sidebar.image("movie-critic.webp", use_container_width=True)
    page_options = ["Recommender System", "Solution Overview", "Project Documentation", "About"]
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    
    if page_selection == "Recommender System":
        st.title("🎬 Movie Recommender Engine")
        
        # Sélection de l'algorithme
        st.markdown("### 💡 Choisissez un algorithme de recommandation")
        algo = st.selectbox("Algorithm:", (
            'Collaborative Filtering - Memory-Based (Item-User)',
            'Collaborative Filtering - Memory-Based (User-Item)',
            'Collaborative Filtering - Model-Based (NMF)',
            'Collaborative Filtering - Model-Based (SVD)',
            'Collaborative Filtering - Model-Based (KNN)',
            'Content-Based Filtering'
        ))
        
        st.markdown("### 🎥 Sélectionnez vos 3 films préférés et évaluez-les")
        # Saisie des 3 films et leurs notes
        movie_1 = st.selectbox('Premier film', movie_titles, key='movie_1')
        rating_1 = st.slider(f"Évaluez {movie_1} (1-5)", 1, 5, 3, key=f"rating_1_{movie_1}")
        
        movie_2 = st.selectbox('Deuxième film', movie_titles, key='movie_2')
        rating_2 = st.slider(f"Évaluez {movie_2} (1-5)", 1, 5, 3, key=f"rating_2_{movie_2}")
        
        movie_3 = st.selectbox('Troisième film', movie_titles, key='movie_3')
        rating_3 = st.slider(f"Évaluez {movie_3} (1-5)", 1, 5, 3, key=f"rating_3_{movie_3}")
        
        # Affichage des préférences de l'utilisateur
        st.markdown("### Vos films sélectionnés et vos évaluations")
        st.write(f"{movie_1} : {rating_1} ⭐")
        st.write(f"{movie_2} : {rating_2} ⭐")
        st.write(f"{movie_3} : {rating_3} ⭐")
        
        # --- Intégration temporaire dans le dataset ---
        # On attribue un nouvel identifiant utilisateur (par exemple 9999)
        user_preferences = pd.DataFrame([
            {"userId": 9999, "title": movie_1, "rating": rating_1},
            {"userId": 9999, "title": movie_2, "rating": rating_2},
            {"userId": 9999, "title": movie_3, "rating": rating_3},
        ])
        df_original = title_list.copy()
        df_temp = pd.concat([df_original, user_preferences], ignore_index=True)
        
        # Création de la matrice utilisateur-film
        user_movie_matrix = create_user_movie_matrix(df_temp)
        
        # --- Calcul des recommandations ---
        if st.button("Obtenir des recommandations"):
            st.markdown("### 🔥 Films recommandés")
            
            if algo == 'Collaborative Filtering - Memory-Based (Item-User)':
                recs = recommend_item_user(new_user_id=9999, user_movie_matrix=user_movie_matrix)
                st.markdown("#### Recommandations Item-User:")
                for movie, score in recs:
                    st.write(f"{movie} - Score prédit: {score:.2f}")
            
            elif algo == 'Collaborative Filtering - Memory-Based (User-Item)':
                recs = recommend_user_item(new_user_id=9999, user_movie_matrix=user_movie_matrix)
                st.markdown("#### Recommandations User-Item:")
                for movie, score in recs:
                    st.write(f"{movie} - Score prédit: {score:.2f}")
            
            elif algo == 'Collaborative Filtering - Model-Based (NMF)':
                recs = recommend_nmf(new_user_id=9999, user_movie_matrix=user_movie_matrix)
                st.markdown("#### Recommandations NMF:")
                for movie, score in recs:
                    st.write(f"{movie} - Score prédit: {score:.2f}")
            
            elif algo == 'Collaborative Filtering - Model-Based (SVD)':
                recs = recommend_svd(new_user_id=9999, user_movie_matrix=user_movie_matrix)
                st.markdown("#### Recommandations SVD:")
                for movie, score in recs:
                    st.write(f"{movie} - Score prédit: {score:.2f}")
            
            elif algo == 'Collaborative Filtering - Model-Based (KNN)':
                recs = recommend_knn(new_user_id=9999, user_movie_matrix=user_movie_matrix)
                st.markdown("#### Recommandations KNN:")
                for movie, score in recs:
                    st.write(f"{movie} - Score prédit: {score:.2f}")
            
            elif algo == 'Content-Based Filtering':
                # Pour la recommandation basée sur le contenu, on détermine le film le mieux évalué parmi les 3
                user_films = {movie_1: rating_1, movie_2: rating_2, movie_3: rating_3}
                best_film = max(user_films, key=user_films.get)
                st.markdown(f"#### Film le mieux évalué: {best_film}")
                recs = recommend_content(best_film, title_list, top_n=5)
                st.markdown("#### Recommandations Content-Based:")
                for movie, score in recs:
                    st.write(f"{movie} - Similarité: {score:.2f}")
    
    elif page_selection == "Solution Overview":
        st.title("📊 Solution Overview")
        st.markdown("""
        **Project Goal:**  
        Build a robust movie recommendation system using both collaborative and content-based approaches.
        
        **Techniques Implemented:**
        - Content-Based Filtering
        - Collaborative Filtering (Memory-Based and Model-Based: NMF, SVD, KNN)
        
        **Technologies Used:**
        - Python, Pandas, Numpy
        - Streamlit for the GUI
        - Machine Learning libraries such as Scikit-learn and Surprise
        """)
    
    elif page_selection == "Project Documentation":
        st.title("📚 Project Documentation")
        st.markdown("""
        **Project Goal:**  
        Develop a robust movie recommendation system.
        
        **Techniques:**  
        - Content-Based Filtering  
        - Collaborative Filtering (Memory-Based and Model-Based)
        
        **Algorithms:**  
        - Memory-based filtering  
        - Model-based filtering: NMF, SVD, KNN
        
        **Technologies:**  
        - Python, Pandas, Numpy  
        - Streamlit, Scikit-learn, Surprise
        
        **Deployment:**  
        Streamlit web application.
        """)
    
    elif page_selection == "About":
        st.title("ℹ️ About")
        st.markdown("Developed as part of a data science project for EXPLORE Data Science Academy.")
        st.markdown("Created by: Your Name")
        st.markdown("Year: 2025")

if __name__ == '__main__':
    main()
