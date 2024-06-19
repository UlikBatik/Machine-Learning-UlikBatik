import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedRecommender:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
        self.vectorizer = TfidfVectorizer()
        self.user_profiles = None
        self.tfidf_matrix = None
        self.user_post_set = None
        self.unique_posts = None
        self._prepare_data()

    def _prepare_data(self):
        # Create user profiles based on the labels of the posts they have liked
        self.user_profiles = self.data.groupby('USERID')['BATIKID'].apply(lambda x: ' '.join(x)).reset_index()
        self.user_profiles.columns = ['USERID', 'Profile']
        
        # Mengambil setiap jenis label
        all_labels = self.data['BATIKID'].unique()
        
        # Mengubah BATIKID menjadi TF-IDF matrix
        self.tfidf_matrix = self.vectorizer.fit_transform(all_labels)
        
        # Get a list of all unique post IDs and labels
        self.unique_posts = self.data[['POSTID', 'BATIKID']].drop_duplicates()
        
        # Membuat variable untuk mengecek apakah postingan belum pernah disukai user
        self.user_post_set = set(zip(self.data['USERID'], self.data['POSTID']))

        # Membuat variable untuk mengecek apakah postingan 
        self.user_own_posts = set(zip(self.data['USERID'], self.data['POSTID']))


    def get_recommendations(self, user_id, top_n=10):
        user_row = self.user_profiles[self.user_profiles['USERID'] == user_id]
        if user_row.empty:
            return f"No data available for USERID {user_id}"
        
        user_profile = user_row['Profile'].values[0]

        # Mengubah history like menjadi TF-IDF matrix
        user_tfidf = self.vectorizer.transform([user_profile])

        # Menyiapkan list kosong untuk menyimpan simillarity scores untuk postingan yang belum pernah user sukai
        similarity_scores = []

        # Iterate over all unique posts
        for post_id, post_label in self.unique_posts.itertuples(index=False):
            # Mengecek apakah user belum pernah menyukai postingan tersebut
            if (user_id, post_id) not in self.user_post_set and (user_id, post_id) not in self.user_own_posts:
                # Transform the post label using the same TF-IDF vectorizer
                post_tfidf = self.vectorizer.transform([post_label])

                # Menghitung cosine similarity antara user profile dan post label(BATIKID)
                similarity = cosine_similarity(user_tfidf, post_tfidf).flatten()[0]

                # memasukan konten yang terkait
                similarity_scores.append((post_id, similarity))

        # Sort the similarity scores in descending order and get the top N posts
        top_posts = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[:top_n]
        top_post_ids = [post_id for post_id, score in top_posts]

        return top_post_ids

# Example usage
file_path = 'detail_like_with_label.csv'  
recommender = ContentBasedRecommender(file_path)

user_id_to_check = 1
recommendations = recommender.get_recommendations(user_id_to_check)
print(f"Recommendations for USERID {user_id_to_check}: POSTID{recommendations}")
