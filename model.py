from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import logging
import os
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configured CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://careerwiz-frontend.vercel.app"],  
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    expose_headers=["Content-Type"],
)

# Global variables
students_df = None
catalog_df = None
vectorizer = None
model = None

class StudentData(BaseModel):
    subjects: List[str]
    interests: List[str]

def load_data():
    global students_df, catalog_df
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        students_path = os.path.join(base_dir, 'student_data.csv')
        courses_path = os.path.join(base_dir, 'Courses.csv')
        students_df = pd.read_csv(students_path)
        catalog_df = pd.read_csv(courses_path)
        logger.info("Data loaded successfully")
    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        raise HTTPException(status_code=500, detail="Data files not found")
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise HTTPException(status_code=500, detail="Error loading data")

def initialize_vectorizer():
    global vectorizer, student_vectors, course_vectors
    try:
        vectorizer = CountVectorizer(tokenizer=lambda x: x.split(', '))
        student_vectors = vectorizer.fit_transform(students_df['subjects'] + ', ' + students_df['interests'])
        course_vectors = vectorizer.transform(catalog_df['subjects'] + ', ' + catalog_df['skills'])
        logger.info("Vectorizer initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing vectorizer: {str(e)}")
        raise HTTPException(status_code=500, detail="Error initializing vectorizer")

def train_model():
    global model
    try:
        course_features = catalog_df['subjects'] + ', ' + catalog_df['skills']
        course_vectors = vectorizer.transform(course_features)
        # The model finds the 5 course vectors that are most similar to the student's interests/subjects vector.
        model = NearestNeighbors(n_neighbors=5)
        model.fit(course_vectors)
        logger.info("Model trained successfully")
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise HTTPException(status_code=500, detail="Error training model")

def vectorize_student_input(subjects: List[str], interests: List[str]):
    try:
        combined_input = ', '.join(subjects + interests)
        return vectorizer.transform([combined_input])
    except Exception as e:
        logger.error(f"Error vectorizing student input: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing student data")

@app.get("/")
def read_root():
    return {"message": "Hello ML World"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.on_event("startup")
async def startup_event():
    try:
        logger.info("Starting up application...")
        load_data()
        logger.info("Data loaded successfully")
        initialize_vectorizer()
        logger.info("Vectorizer initialized successfully")
        train_model()
        logger.info("Model trained successfully")
        logger.info("Application startup complete")
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise

@app.post("/predict_career")
def predict_career(student_data: StudentData):
    logger.info("Received request for predict_career")
    try:
        student_vector = vectorize_student_input(student_data.subjects, student_data.interests)
        distances, indices = model.kneighbors(student_vector)
        recommended_courses = catalog_df.iloc[indices[0]]['course_name'].tolist()
        logger.info("Completed predict_career request")
        return {"recommended_courses": recommended_courses}
    except Exception as e:
        logger.error(f"Error in predict_career: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating recommendations")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)