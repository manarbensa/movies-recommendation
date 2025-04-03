import streamlit as st
import pandas as pd

# Set the page configuration as the first Streamlit command
st.set_page_config(page_title="Movie Recommender System", layout="wide")

# Load data
title_list = pd.read_csv('user_ratings_genres_mov.csv')
movie_titles = title_list['title'].unique()
movie_genres = title_list['genres'].str.replace('|', ', ').unique()

# Function to inject custom CSS for modern design
def inject_custom_css():
    custom_css = """
    <style>
    /* Global body styling */
    body {
        background-color: #f4f6f8;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #333333;
    }

    /* Header and Title Styling */
    h1, h2, h3, h4, h5, h6 {
        color: #333333;
    }

    /* Sidebar customization */
    .css-1d391kg {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    .css-1d391kg a {
        color: #4fd3c4;
    }

    /* Button Styling */
    div.stButton > button {
        background-color: #4fd3c4;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.5em 1em;
        font-size: 1em;
        cursor: pointer;
    }

    div.stButton > button:hover {
        background-color: #3bb8a1;
    }

    /* Selectbox Styling */
    .stSelectbox select {
        background-color: #ffffff;
        border: 1px solid #cccccc;
        border-radius: 4px;
    }

    /* Main content padding */
    .main {
        padding: 20px;
    }

    /* Custom background for the sidebar images */
    .css-1y4k81r {
        padding-top: 15px;
    }

    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

# Main application function
def main():
    inject_custom_css()
    
    # Display image in the sidebar (verify the path)
    st.sidebar.image("movie-critic.webp", use_container_width=True)
    
    # Sidebar menu options
    page_options = ["Recommender System", "Solution Overview", "Project Documentation", "About"]
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    
    # Page: Recommender System
    if page_selection == "Recommender System":
        st.title("🎬 Movie Recommender Engine")
        
        # Algorithm selection
        st.markdown("### 💡 Choose a Recommendation Algorithm")
        sys = st.selectbox("Algorithm:",
                           ('Content-Based Filtering',
                            'Collaborative Filtering - Memory-Based',
                            'Collaborative Filtering - Model-Based (NMF)',
                            'Collaborative Filtering - Model-Based (SVD)',
                            'Collaborative Filtering - Model-Based (KNN)'))
        
        # Favorite movies input
        st.markdown("### 🎥 Sélectionnez vos 3 films préférés, évaluez-les et ajoutez leurs genres")
        
        # Movie 1
        movie_1 = st.selectbox('Premier film', movie_titles, key='movie_1')
        rating_1 = st.slider(f"Évaluez {movie_1} (1-5)", 1, 5, 3, key=f"rating_1_{movie_1}")
        genre_1 = st.selectbox(f'Genre de {movie_1}', movie_genres, key=f'genre_1_{movie_1}')
        
        # Movie 2
        movie_2 = st.selectbox('Deuxième film', movie_titles, key='movie_2')
        rating_2 = st.slider(f"Évaluez {movie_2} (1-5)", 1, 5, 3, key=f"rating_2_{movie_2}")
        genre_2 = st.selectbox(f'Genre de {movie_2}', movie_genres, key=f'genre_2_{movie_2}')
        
        # Movie 3
        movie_3 = st.selectbox('Troisième film', movie_titles, key='movie_3')
        rating_3 = st.slider(f"Évaluez {movie_3} (1-5)", 1, 5, 3, key=f"rating_3_{movie_3}")
        genre_3 = st.selectbox(f'Genre de {movie_3}', movie_genres, key=f'genre_3_{movie_3}')

        # Display the selected movies, ratings, and genres
        st.markdown("### Vos films sélectionnés, vos évaluations et leurs genres")
        st.write(f"{movie_1} : {rating_1} ⭐ | Genre: {genre_1}")
        st.write(f"{movie_2} : {rating_2} ⭐ | Genre: {genre_2}")
        st.write(f"{movie_3} : {rating_3} ⭐ | Genre: {genre_3}")

        #--------------------partie de felicien------------------------------------------------
        #---------------------------------------------------
        #------------------------------------------------------------
        
        '''if sys == 'Content Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies,
                                                            top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")'''
#----------------------------------------------------------------------------------

        '''if sys == 'Collaborative Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = collab_model(movie_list=fav_movies,
                                                           top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")'''


    # Page: Solution Overview
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
    
    # Page: Project Documentation
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
    
    # Page: About
    elif page_selection == "About":
        st.title("ℹ️ About")
        st.markdown("Developed as part of a data science project for EXPLORE Data Science Academy.")
        st.markdown("Created by: Your Name")
        st.markdown("Year: 2025")

if __name__ == '__main__':
    main()
