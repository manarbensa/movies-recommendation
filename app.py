import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer


title_list = pd.read_csv('user_ratings_genres_mov.csv')
movie_titles = title_list['title'].unique()


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

# --- Fonction d'affichage du carousel d'images ---
def header_carousel():
    carousel_html = """
    <style>
      /* Conteneur du carousel */
      .carousel {
        position: relative;
        width: 100%;
        overflow: hidden;
        margin-bottom: 20px;
      }
      /* Conteneur intérieur avec animation */
      .carousel-inner {
        display: flex;
        width: 500%; /* 5 images : 5 x 100% */
        animation: slide 15s infinite;
      }
      .carousel-inner img {
        width: 8%;  /* Ajustement de la taille des images */
        object-fit: cover;
        border-radius: 8px;
        margin: 0 5px;
      }
      /* Animation de défilement */
      @keyframes slide {
        0% { transform: translateX(0); }
        20% { transform: translateX(0); }
        25% { transform: translateX(-10%); }
        45% { transform: translateX(-10%); }
        50% { transform: translateX(-20%); }
        70% { transform: translateX(-20%); }
        75% { transform: translateX(-30%); }
        95% { transform: translateX(-30%); }
        100% { transform: translateX(0); }
      }
    </style>
    <div class="carousel">
      <div class="carousel-inner">
         <img src="https://m.media-amazon.com/images/M/MV5BOGNkODRkNzUtNjk5My00MmQxLWFkNjQtZmUwNzg5YjBhNWIwXkEyXkFqcGc@._V1_.jpg" alt="Jurassic Park">
         <img src="https://play-lh.googleusercontent.com/ibLPWUi77ykXK8Km_8I3rTLYUEFVpqtDhH4dWGVz3xY5fH2zq4q47xa5xtYvGI_BIFBNxMezb9YEvD452TA" alt="Film 2">
         <img src="https://upload.wikimedia.org/wikipedia/fr/thumb/6/6a/Jurassic_Park_logo.svg/1200px-Jurassic_Park_logo.svg.png" alt="Film 3">
         <img src="https://m.media-amazon.com/images/M/MV5BMjAxMzY3NjcxNF5BMl5BanBnXkFtZTcwNTI5OTM0Mw@@._V1_FMjpg_UX1000_.jpg" alt="Film 4">
         <img src="https://fr.web.img6.acsta.net/pictures/14/09/24/12/08/158828.jpg" alt="Film 5">
      </div>
    </div>
    """
    st.markdown(carousel_html, unsafe_allow_html=True)

# --- Création de la matrice utilisateur-film ---
def create_user_movie_matrix(df):
    """Crée une matrice utilisateur-film à partir du dataframe."""
    return df.pivot_table(index='userId', columns='title', values='rating').fillna(0)

# --- Recommandations collaboratives basées sur la mémoire (Item-User) ---
def recommend_item_user(new_user_id, user_movie_matrix, top_n=5):
    """Approche Item-User : calcule une note prédite pour chaque film non noté par le nouvel utilisateur."""
    item_similarity = cosine_similarity(user_movie_matrix.T)
    item_similarity_df = pd.DataFrame(item_similarity, index=user_movie_matrix.columns, columns=user_movie_matrix.columns)
    user_ratings = user_movie_matrix.loc[new_user_id]
    pred_ratings = {}
    for movie in user_movie_matrix.columns:
        if user_ratings[movie] == 0:
            rated_movies = user_ratings[user_ratings > 0].index
            numer = 0.0
            denom = 0.0
            for rated in rated_movies:
                numer += item_similarity_df.loc[movie, rated] * user_ratings[rated]
                denom += abs(item_similarity_df.loc[movie, rated])
            if denom > 0:
                pred_ratings[movie] = numer / denom
    top_movies = sorted(pred_ratings.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return top_movies

# --- Recommandations collaboratives basées sur la mémoire (User-Item) ---
def recommend_user_item(new_user_id, user_movie_matrix, top_n=5):
    """
    Approche User-Item : prédit une note pour chaque film non noté par le nouvel utilisateur
    en utilisant une moyenne pondérée basée sur la similarité entre utilisateurs.
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
            for other_user in user_movie_matrix.index:
                if other_user != new_user_id and user_movie_matrix.at[other_user, movie] > 0:
                    similarity = user_similarity_df.at[new_user_id, other_user]
                    rating = user_movie_matrix.at[other_user, movie]
                    numer += similarity * rating
                    denom += abs(similarity)
            if denom > epsilon:
                pred_ratings[movie] = numer / (denom + epsilon)
    
    top_movies = sorted(pred_ratings.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return top_movies

# --- Recommandations collaboratives basées sur un modèle (NMF) ---
def recommend_nmf(new_user_id, user_movie_matrix, top_n=5):
    """
    Utilise NMF pour factoriser la matrice de notes et reconstruire les notes.
    Recommande les films non notés par le nouvel utilisateur en utilisant la reconstruction.
    """
    nmf_model = NMF(n_components=20, init='nndsvda', max_iter=500, random_state=42)
    W = nmf_model.fit_transform(user_movie_matrix)
    H = nmf_model.components_
    R_pred = np.dot(W, H)
    R_pred_df = pd.DataFrame(R_pred, index=user_movie_matrix.index, columns=user_movie_matrix.columns)
    
    user_ratings = user_movie_matrix.loc[new_user_id]
    predictions = R_pred_df.loc[new_user_id]
    pred_ratings = {movie: predictions[movie] for movie in user_movie_matrix.columns if user_ratings[movie] == 0}
    top_movies = sorted(pred_ratings.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return top_movies

# --- Recommandations collaboratives basées sur un modèle (SVD) ---
def recommend_svd(new_user_id, user_movie_matrix, top_n=5):
    """
    Utilise SVD (TruncatedSVD) pour reconstruire la matrice de notes et recommande les films non notés par le nouvel utilisateur.
    """
    user_means = user_movie_matrix.replace(0, np.nan).mean(axis=1).fillna(0)
    centered_matrix = user_movie_matrix.sub(user_means, axis=0)
    svd_model = TruncatedSVD(n_components=20, random_state=42)
    U = svd_model.fit_transform(centered_matrix)
    S = svd_model.singular_values_
    VT = svd_model.components_
    R_pred_centered = np.dot(U, np.dot(np.diag(S), VT))
    R_pred = R_pred_centered + user_means.values.reshape(-1, 1)
    R_pred_df = pd.DataFrame(R_pred, index=user_movie_matrix.index, columns=user_movie_matrix.columns)
    
    user_ratings = user_movie_matrix.loc[new_user_id]
    predictions = R_pred_df.loc[new_user_id]
    pred_ratings = {movie: predictions[movie] for movie in user_movie_matrix.columns if user_ratings[movie] == 0}
    top_movies = sorted(pred_ratings.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return top_movies

# --- Recommandations collaboratives basées sur un modèle (KNN) ---
def recommend_knn(new_user_id, user_movie_matrix, top_n=5, n_neighbors=5):
    """
    Utilise KNN pour recommander les films non notés par le nouvel utilisateur en s'appuyant sur les voisins les plus proches,
    avec une pondération des notes par l'inverse de la distance.
    """
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    model_knn.fit(user_movie_matrix)
    distances, indices = model_knn.kneighbors([user_movie_matrix.loc[new_user_id]], n_neighbors=n_neighbors+1)
    distances = distances.flatten()[1:]
    similar_users = indices.flatten()[1:]
    
    user_ratings = user_movie_matrix.loc[new_user_id]
    pred_ratings = {}
    
    for movie in user_movie_matrix.columns:
        if user_ratings[movie] == 0:
            weighted_sum = 0.0
            weight_sum = 0.0
            for dist, neighbor in zip(distances, similar_users):
                neighbor_rating = user_movie_matrix.iloc[neighbor][movie]
                if neighbor_rating > 0:
                    weight = 1 / (dist + 1e-8)
                    weighted_sum += neighbor_rating * weight
                    weight_sum += weight
            if weight_sum > 0:
                pred_ratings[movie] = weighted_sum / weight_sum
                
    top_movies = sorted(pred_ratings.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return top_movies

# --- Recommandation basée sur le contenu ---
def recommend_content(best_film, movies_df, top_n=5):
    """
    Recommande des films similaires au film 'best_film' en se basant sur la similarité des genres.
    Cette version élimine les doublons pour éviter de recommander le même film plusieurs fois.
    """
    # Créer un DataFrame avec des films uniques (en se basant sur le titre)
    unique_movies_df = movies_df.drop_duplicates(subset='title').reset_index(drop=True)
    
    # Transformation des genres en représentation TF-IDF
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    tfidf_matrix = vectorizer.fit_transform(unique_movies_df['genres'])
    
    # Calcul de la similarité cosine entre tous les films
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Récupération de l'index du film 'best_film'
    idx_list = unique_movies_df[unique_movies_df['title'].str.lower() == best_film.lower()].index
    if len(idx_list) == 0:
        return []  # Si le film n'est pas trouvé, retourner une liste vide
    idx = idx_list[0]
    
    # Calcul des scores de similarité pour le film choisi
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Tri des films par ordre décroissant de similarité
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Parcourir les scores, exclure le film lui-même et collecter les top_n recommandations uniques
    recommendations = []
    for i, score in sim_scores:
        # Exclure le film lui-même
        if unique_movies_df.iloc[i]['title'].lower() == best_film.lower():
            continue
        recommendations.append((unique_movies_df.iloc[i]['title'], score))
        if len(recommendations) == top_n:
            break
    
    return recommendations


# --- Interface Streamlit ---
def main():
    inject_custom_css()
    header_carousel()  # Affichage du carousel en haut de la page
    
    st.sidebar.image("movie-critic.webp", use_container_width=True)
    page_options = ["Recommender System", "Solution Overview", "Project Documentation", "About"]
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    
    if page_selection == "Recommender System":
        st.title("🎬 Movie Recommender Engine")
        
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
        movie_1 = st.selectbox('Premier film', movie_titles, key='movie_1')
        rating_1 = st.slider(f"Évaluez {movie_1} (1-5)", 1, 5, 3, key=f"rating_1_{movie_1}")
        
        movie_2 = st.selectbox('Deuxième film', movie_titles, key='movie_2')
        rating_2 = st.slider(f"Évaluez {movie_2} (1-5)", 1, 5, 3, key=f"rating_2_{movie_2}")
        
        movie_3 = st.selectbox('Troisième film', movie_titles, key='movie_3')
        rating_3 = st.slider(f"Évaluez {movie_3} (1-5)", 1, 5, 3, key=f"rating_3_{movie_3}")
        
        st.markdown("### Vos films sélectionnés et vos évaluations")
        st.write(f"{movie_1} : {rating_1} ⭐")
        st.write(f"{movie_2} : {rating_2} ⭐")
        st.write(f"{movie_3} : {rating_3} ⭐")
        
        user_preferences = pd.DataFrame([
            {"userId": 9999, "title": movie_1, "rating": rating_1, "genres": ""},
            {"userId": 9999, "title": movie_2, "rating": rating_2, "genres": ""},
            {"userId": 9999, "title": movie_3, "rating": rating_3, "genres": ""},
        ])
        df_original = title_list.copy()
        df_temp = pd.concat([df_original, user_preferences], ignore_index=True)
        
        user_movie_matrix = create_user_movie_matrix(df_temp)
        
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
        st.title("ℹ About")
        st.markdown("Developed with ❤ by [Bensaada Manar](https://github.com/bensaadam) and [Aponi Felicien](https://github.com/felicienaponi).")
        st.markdown("Year: 2025")

if __name__ == '__main__':
    main()

