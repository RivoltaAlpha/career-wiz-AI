from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
# import logging

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

app = FastAPI()

# Load data
students_df = pd.read_csv('student_data.csv')
catalog_df = pd.read_csv('course_catalog.csv')

# Initialize the vectorizer
vectorizer = CountVectorizer(tokenizer=lambda x: x.split(', '))

# Create feature vectors for both students and courses
student_vectors = vectorizer.fit_transform(students_df['subjects'] + ', ' + students_df['interests'])
course_vectors = vectorizer.transform(catalog_df['subjects'] + ', ' + catalog_df['skills'])

# Placeholder for our ML model
model = None

# Example data model for student input
class StudentData(BaseModel):
    subjects: list[str]
    interests: list[str]

# Train the model using course catalog data
def train_model():
    global model
    
    # Vectorization of subjects and skills
    # Combine subjects and skills into a single feature for each course
    course_features = catalog_df['subjects'] + ', ' + catalog_df['skills']
    
    # Vectorize the course features
    course_vectors = vectorizer.transform(course_features)
    
    # Fit a Nearest Neighbors model on the course catalog
    model = NearestNeighbors(n_neighbors=5)  # Get top 5 neighbors (courses)
    model.fit(course_vectors)

# Vectorize student input for predictions
def vectorize_student_input(subjects, interests):
    # Combine the subjects and interests
    combined_input = ', '.join(subjects + interests)
    
    # Transform using the same vectorizer trained on the course data
    student_vector = vectorizer.transform([combined_input])
    return student_vector

# Display recommendations for each student
def generate_recommendations():
    recommendations = {}
    similarity_matrix = cosine_similarity(student_vectors, course_vectors)
    
    for i, student_id in enumerate(students_df['student_id']):
        similar_courses = similarity_matrix[i].argsort()[::-1]  # Sort courses by similarity
        recommended_courses = catalog_df.iloc[similar_courses[:5]]['course_name'].tolist()  # Top 5 courses
        recommendations[student_id] = recommended_courses
    return recommendations

@app.get("/")
def read_root():
    return {"message" : "Hello ML World"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Train the model at startup
@app.on_event("startup")
def startup_event():
    train_model()

# Prediction Endpoint
@app.post("/predict_career")
def predict_career(student_data: StudentData):
    # logger.info("Received request for predict_career")

    # Combine student subjects and interests
    student_vector = vectorize_student_input(student_data.subjects, student_data.interests)
    
    # Find the nearest courses using the trained model
    distances, indices = model.kneighbors(student_vector)
    
    # Get the top recommended courses
    recommended_courses = catalog_df.iloc[indices[0]]['course_name'].tolist()
    
    # Return the recommended courses as a response
    # logger.info("Completed predict_career request")

    return {"recommended_courses": recommended_courses}

